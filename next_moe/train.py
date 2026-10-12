"""Single-GPU test trainer with autocast, checkpointing, evaluation and resume."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import time
import torch
from .data import TokenBatches
from .model import ModelConfig, NextMoE


@dataclass
class TrainConfig:
    steps: int = 300  # total optimizer updates, including resumed updates
    sequence_length: int = 128
    microbatch: int = 1
    accumulation: int = 4
    learning_rate: float = 2e-4
    warmup: int = 20
    auxiliary_weight: float = .01
    weight_decay: float = .1
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 4
    save_every: int = 100
    log_every: int = 10
    optimizer: str = 'adamw'  # optional 'adamw8bit'; no CPU offload
    seed: int = 1234


class Trainer:
    def __init__(self, model_config, train_config, data_dir, output_dir, device=None, resume=None):
        self.cfg, self.model_config = train_config, model_config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if min(train_config.steps, train_config.accumulation, train_config.microbatch,
               train_config.sequence_length, train_config.eval_batches, train_config.log_every,
               train_config.eval_every, train_config.save_every) < 1:
            raise ValueError('Step/batch/interval values must be positive.')
        if train_config.optimizer not in ('adamw', 'adamw8bit'):
            raise ValueError('Unknown optimizer.')
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)
        meta = json.loads((self.data_dir / 'metadata.json').read_text())
        if meta['actual_vocab'] > model_config.vocab_size:
            raise ValueError('Tokenizer vocabulary exceeds model embedding size.')
        if meta['special_ids']['<eos>'] != model_config.eos_id or meta['special_ids']['<pad>'] != model_config.pad_id:
            raise ValueError('Tokenizer special IDs do not match model configuration.')
        self.data_signature = self._data_signature()
        torch.manual_seed(train_config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(train_config.seed)
        self.batches = TokenBatches(data_dir, train_config.sequence_length,
                                   train_config.microbatch, train_config.seed)
        self.model = NextMoE(model_config).to(self.device)
        # FP32 resident parameters/grads; autocast computes activations in BF16/FP16.
        # Standard AdamW has the same 16 bytes/parameter storage as BF16 + FP32 master.
        if train_config.optimizer == 'adamw8bit':
            if self.device.type != 'cuda':
                raise ValueError('8-bit optimizer needs a supported CUDA GPU.')
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(),
                                                lr=train_config.learning_rate,
                                                weight_decay=train_config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.learning_rate,
                                               weight_decay=train_config.weight_decay, foreach=False)
        self.dtype = (torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
                      else torch.float16)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=4096,
                                          enabled=self.device.type == 'cuda' and self.dtype == torch.float16)
        self.step, self.tokens_seen = 0, 0
        self.last_counts = None
        if resume:
            self.load(resume)
        (self.output / 'model_config.json').write_text(json.dumps(model_config.to_dict(), indent=2))
        (self.output / 'train_config.json').write_text(json.dumps(asdict(train_config), indent=2))
        import shutil
        shutil.copy2(self.data_dir / 'tokenizer.json', self.output / 'tokenizer.json')
        shutil.copy2(self.data_dir / 'metadata.json', self.output / 'data_metadata.json')
        print(f'Parameters: {sum(p.numel() for p in self.model.parameters()):,}; device={self.device}; '
              f'autocast={self.dtype if self.device.type == "cuda" else "off"}; step={self.step}')

    def _data_signature(self):
        digest = hashlib.sha256()
        for name in ['tokenizer.json', 'metadata.json', 'train_uk.bin', 'train_en.bin', 'val_uk.bin', 'val_en.bin']:
            with (self.data_dir / name).open('rb') as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
        return digest.hexdigest()

    def autocast(self):
        return (torch.autocast('cuda', dtype=self.dtype) if self.device.type == 'cuda' else nullcontext())

    def learning_rate(self):
        c = self.cfg
        if self.step < c.warmup:
            return c.learning_rate * (self.step + 1) / max(c.warmup, 1)
        progress = (self.step - c.warmup) / max(c.steps - c.warmup, 1)
        return c.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1))))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        result = {}
        rng = torch.Generator().manual_seed(self.cfg.seed + 999)  # same windows each evaluation
        for language in ('uk', 'en'):
            losses = []
            for _ in range(self.cfg.eval_batches):
                x, y = self.batches.get('val', self.device, language, rng)
                with self.autocast():
                    losses.append(float(self.model(x, y)['ce']))
            result[f'val_ce_{language}'] = sum(losses) / len(losses)
            result[f'val_ppl_{language}'] = math.exp(min(result[f'val_ce_{language}'], 20))
        self.model.train()
        return result

    def save(self, filename='last.pt'):
        # Atomic replace: an interruption during writing preserves the previous file.
        destination = self.output / filename
        temporary = destination.with_suffix('.pt.tmp')
        checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                      'scaler': self.scaler.state_dict(), 'step': self.step, 'tokens_seen': self.tokens_seen,
                      'model_config': self.model_config.to_dict(), 'train_config': asdict(self.cfg),
                      'data_signature': self.data_signature, 'batch_rng': self.batches.rng.get_state(),
                      'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []}
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)
        print(f'Saved {destination} ({destination.stat().st_size / 2**30:.2f} GiB)')
        return destination

    def load(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if saved['model_config'] != self.model_config.to_dict():
            raise ValueError('Resume requires identical model configuration.')
        if saved['data_signature'] != self.data_signature:
            raise ValueError('Resume requires exactly the same tokenizer and token corpus.')
        # Strict resume: only output intervals and total steps may change.
        old = saved['train_config']
        for name in ('optimizer', 'sequence_length', 'microbatch', 'accumulation', 'learning_rate',
                     'warmup', 'weight_decay', 'grad_clip', 'auxiliary_weight', 'seed'):
            if old[name] != getattr(self.cfg, name):
                raise ValueError(f'Resume changed {name}. Start a new run instead.')
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scaler.load_state_dict(saved['scaler'])
        self.step, self.tokens_seen = saved['step'], saved['tokens_seen']
        self.batches.rng.set_state(saved['batch_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if self.device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'Resumed optimizer step {self.step}; changing total steps also changes the future cosine schedule.')

    def run(self):
        c = self.cfg
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        last_log, window_tokens = time.perf_counter(), 0
        overflow_retries = 0
        try:
            while self.step < c.steps:
                lr = self.learning_rate()
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                ce_total, aux_total, counts = 0.0, 0.0, None
                batch_rng_start = self.batches.rng.get_state()
                for _ in range(c.accumulation):
                    x, y = self.batches.get('train', self.device)
                    with self.autocast():
                        out = self.model(x, y)
                        loss = (out['ce'] + c.auxiliary_weight * out['aux']) / c.accumulation
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Non-finite loss. Check data, LR and precision.')
                    self.scaler.scale(loss).backward()
                    ce_total += float(out['ce'].detach()) / c.accumulation
                    aux_total += float(out['aux'].detach()) / c.accumulation
                    now_counts = self.model.last_counts
                    counts = now_counts.clone() if counts is None else counts + now_counts
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                if not bool(torch.isfinite(norm)):
                    if self.scaler.is_enabled() and overflow_retries < 8:
                        # unscale_ registered infs: scaler.step skips the optimizer update.
                        # Restore sampler so the smaller scale retries the same microbatches.
                        old_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.batches.rng.set_state(batch_rng_start)
                        overflow_retries += 1
                        print(f'FP16 overflow: retry {overflow_retries}, scale {old_scale} -> {self.scaler.get_scale()}')
                        continue
                    raise FloatingPointError('Persistent non-finite gradients. Reduce LR or use BF16 if available.')
                overflow_retries = 0
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                seen = c.microbatch * c.sequence_length * c.accumulation
                self.tokens_seen += seen
                window_tokens += seen
                self.last_counts = counts
                record = {'step': self.step, 'train_ce': ce_total, 'router_aux': aux_total,
                          'lr': lr, 'tokens_seen': self.tokens_seen, 'grad_norm': float(norm),
                          'unselected_expert_fraction_this_step': float((counts == 0).float().mean()),
                          'max_expert_to_mean_load': float((counts.max(-1).values / counts.mean(-1).clamp_min(1)).max())}
                if self.step % c.log_every == 0 or self.step == 1:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                        record['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30
                        record['peak_reserved_GiB'] = torch.cuda.max_memory_reserved() / 2**30
                    record['tokens_per_second_including_interval_overhead'] = window_tokens / max(time.perf_counter() - last_log, 1e-6)
                    print(json.dumps(record))
                    last_log, window_tokens = time.perf_counter(), 0
                if self.step % c.eval_every == 0:
                    record.update(self.evaluate())
                    print(json.dumps({'step': self.step, **{k: v for k, v in record.items() if k.startswith('val_')}}))
                with (self.output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                if self.step % c.save_every == 0:
                    self.save()
        except KeyboardInterrupt:
            # If interrupt occurred inside accumulation, partial gradients are not saved.
            self.optimizer.zero_grad(set_to_none=True)
            print('Interrupted: saving the last completed optimizer step; partial microbatch RNG may have advanced.')
        except torch.cuda.OutOfMemoryError:
            print('CUDA OOM: reduce sequence_length/microbatch and start a NEW run; leave checkpointing enabled.')
            raise
        self.save()
        return self.evaluate()

    def export_weights(self):
        path = self.output / 'model.pt'
        torch.save({'model_config': self.model_config.to_dict(), 'model': self.model.state_dict()}, path)
        return path
