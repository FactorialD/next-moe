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
from .precision import MasterWeights


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
    precision: str = 'fp32'  # fp32 / fp16_master / bf16_master
    restart_warmup: int = 100


class Trainer:
    def __init__(self, model_config, train_config, data_dir, output_dir, device=None, resume=None,
                 resume_mode='exact', resume_tokenizer=None, checkpoint_dir=None,
                 persistent_weights=False, disk_budget_gib=None):
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
        self.checkpoint_dir = Path(checkpoint_dir or output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_weights = persistent_weights
        self.disk_budget_gib = disk_budget_gib
        self.log_path = self.checkpoint_dir / 'metrics.jsonl'
        import shutil
        if self.log_path != self.output / 'metrics.jsonl' and not self.log_path.exists() and (self.output / 'metrics.jsonl').exists():
            shutil.copy2(self.output / 'metrics.jsonl', self.log_path)
        self.data_dir = Path(data_dir)
        meta = json.loads((self.data_dir / 'metadata.json').read_text())
        if meta['actual_vocab'] > model_config.vocab_size:
            raise ValueError('Tokenizer vocabulary exceeds model embedding size.')
        if meta['special_ids']['<eos>'] != model_config.eos_id or meta['special_ids']['<pad>'] != model_config.pad_id:
            raise ValueError('Tokenizer special IDs do not match model configuration.')
        self.data_signature = self._data_signature()
        self.tokenizer_signature = hashlib.sha256((self.data_dir / 'tokenizer.json').read_bytes()).hexdigest()
        torch.manual_seed(train_config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(train_config.seed)
        self.batches = TokenBatches(data_dir, train_config.sequence_length,
                                   train_config.microbatch, train_config.seed)
        if train_config.precision not in ('fp32', 'fp16_master', 'bf16_master'):
            raise ValueError('Unknown parameter precision.')
        self.dtype = (torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
                      else torch.float16)
        parameter_dtype = {'fp32': torch.float32, 'fp16_master': torch.float16,
                           'bf16_master': torch.bfloat16}[train_config.precision]
        if parameter_dtype == torch.bfloat16 and self.device.type == 'cuda' and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('This CUDA GPU does not support BF16. Use fp16_master or fp32.')
        if parameter_dtype != torch.float32:
            self.dtype = parameter_dtype
        self.model = NextMoE(model_config).to(self.device)
        self.masters = MasterWeights(self.model.parameters()) if parameter_dtype != torch.float32 else None
        if self.masters:
            self.model.to(dtype=parameter_dtype)
            self.masters.model = list(self.model.parameters())
        self.optim_parameters = self.masters.master if self.masters else list(self.model.parameters())
        if train_config.optimizer == 'adamw8bit':
            if self.device.type != 'cuda':
                raise ValueError('8-bit optimizer needs a supported CUDA GPU.')
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.optim_parameters,
                                                lr=train_config.learning_rate,
                                                weight_decay=train_config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.optim_parameters, lr=train_config.learning_rate,
                                               weight_decay=train_config.weight_decay, foreach=False)
        self.scaler = torch.amp.GradScaler(self.device.type, init_scale=4096,
                                          enabled=self.dtype == torch.float16 and
                                          (self.device.type == 'cuda' or self.masters is not None))
        self.restart_step = None
        self.last_saved_step = -1
        self.routing_total = torch.zeros(model_config.layers, model_config.experts, device=self.device)
        self.step, self.tokens_seen = 0, 0
        self.last_counts = None
        if resume:
            self.load(resume, mode=resume_mode, tokenizer_path=resume_tokenizer)
        (self.output / 'model_config.json').write_text(json.dumps(model_config.to_dict(), indent=2))
        (self.output / 'train_config.json').write_text(json.dumps(asdict(train_config), indent=2))
        import shutil
        shutil.copy2(self.data_dir / 'tokenizer.json', self.output / 'tokenizer.json')
        shutil.copy2(self.data_dir / 'metadata.json', self.output / 'data_metadata.json')
        print(f'Parameters: {sum(p.numel() for p in self.model.parameters()):,}; device={self.device}; '
              f'parameters={parameter_dtype}; autocast={self.dtype if self.device.type == "cuda" else "off"}; step={self.step}')

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
            lr = c.learning_rate * (self.step + 1) / max(c.warmup, 1)
        else:
            progress = (self.step - c.warmup) / max(c.steps - c.warmup, 1)
            lr = c.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1))))
        if self.restart_step is not None and c.restart_warmup > 0:
            lr *= min(1.0, (self.step - self.restart_step + 1) / c.restart_warmup)
        return lr

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        result = {}
        for index, language in enumerate(('uk', 'en')):
            rng = torch.Generator().manual_seed(self.cfg.seed + 999 + index)
            losses = []
            for _ in range(self.cfg.eval_batches):
                x, y = self.batches.get('val', self.device, language, rng)
                with self.autocast():
                    losses.append(float(self.model(x, y)['ce']))
            result[f'val_ce_{language}'] = sum(losses) / len(losses)
            result[f'val_ppl_{language}'] = math.exp(min(result[f'val_ce_{language}'], 20))
        self.model.train()
        return result

    @staticmethod
    def payload_bytes(payload):
        seen = set()
        def size(value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                if key in seen:
                    return 0
                seen.add(key)
                return storage.nbytes()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(size(v) for v in value)
            return 0
        return size(payload)

    def _write(self, payload, destination, budget=None):
        temporary = destination.with_suffix('.pt.tmp')
        estimated = int(self.payload_bytes(payload) * 1.05) + (1 << 20)
        if budget is not None:
            used = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
            if used + estimated > budget * 2**30:
                raise OSError(f'Run disk budget exceeded: existing {used/2**30:.2f} GiB + temporary '
                              f'{estimated/2**30:.2f} GiB > {budget:.2f} GiB. Previous snapshot is preserved.')
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f'Saved {destination} ({destination.stat().st_size / 2**30:.2f} GiB)')
        return destination

    def snapshot(self):
        return {'kind': 'weights', 'model': self.model.state_dict(),
                'step': self.step, 'tokens_seen': self.tokens_seen,
                'model_config': self.model_config.to_dict(), 'train_config': asdict(self.cfg),
                'data_signature': self.data_signature, 'tokenizer_signature': self.tokenizer_signature}

    def save(self, filename='last.pt'):
        checkpoint = {**self.snapshot(), 'kind': 'full',
                      'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(),
                      'restart_step': self.restart_step, 'routing_total': self.routing_total,
                      'master_weights': [p.detach() for p in self.masters.master] if self.masters else None,
                      'batch_rng': self.batches.rng.get_state(), 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []}
        destination = self._write(checkpoint, self.checkpoint_dir / filename,
                                  self.disk_budget_gib if self.checkpoint_dir == self.output else None)
        if self.persistent_weights:
            self._write(self.snapshot(), self.output / 'weights.pt', self.disk_budget_gib)
        if self.log_path != self.output / 'metrics.jsonl' and self.log_path.exists():
            import shutil
            shutil.copy2(self.log_path, self.output / 'metrics.jsonl')
        self.last_saved_step = self.step
        return destination

    def load(self, path, mode='exact', tokenizer_path=None):
        if mode not in ('exact', 'warm'):
            raise ValueError('Resume mode must be exact or warm.')
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if saved['model_config'] != self.model_config.to_dict():
            raise ValueError('Resume requires identical model configuration.')
        if mode == 'exact':
            if 'optimizer' not in saved:
                raise ValueError('Weights-only snapshot requires explicit warm restart.')
            if saved['data_signature'] != self.data_signature:
                raise ValueError('Exact resume requires exactly the same tokenizer and corpus.')
            old = saved['train_config']
            for name in ('optimizer', 'sequence_length', 'microbatch', 'accumulation', 'learning_rate',
                         'warmup', 'weight_decay', 'grad_clip', 'auxiliary_weight', 'seed', 'precision', 'restart_warmup'):
                if old.get(name, getattr(TrainConfig(), name)) != getattr(self.cfg, name):
                    raise ValueError(f'Exact resume changed {name}. Use a new phase with explicit warm restart.')
        else:
            signature = saved.get('tokenizer_signature')
            if signature is None and tokenizer_path is not None:
                signature = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
            if signature is None and saved.get('data_signature') == self.data_signature:
                signature = self.tokenizer_signature  # legacy checkpoint on identical corpus
            if signature != self.tokenizer_signature:
                raise ValueError('Warm restart requires the SAME tokenizer; do not retrain BPE.')
        self.model.load_state_dict(saved['model'])
        if self.masters:
            with torch.no_grad():
                values = saved.get('master_weights') if mode == 'exact' else None
                if values is None:
                    # Use full saved FP32 values when migrating a legacy checkpoint.
                    values = [saved['model'][name] for name, _ in self.model.named_parameters()]
                for master, value in zip(self.masters.master, values):
                    master.copy_(value)
            self.masters.copy_to_model()
        self.step, self.tokens_seen = saved['step'], saved['tokens_seen']
        if mode == 'exact':
            self.optimizer.load_state_dict(saved['optimizer'])
            self.scaler.load_state_dict(saved['scaler'])
            self.batches.rng.set_state(saved['batch_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if self.device.type == 'cuda' and saved['cuda_rng']:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            self.restart_step = saved.get('restart_step')
            if 'routing_total' in saved:
                self.routing_total.copy_(saved['routing_total'])
            self.last_saved_step = self.step
            print(f'Exact resume at step {self.step}; a changed total steps changes future cosine schedule.')
        else:
            self.restart_step = self.step
            print(f'WARM RESTART at step {self.step}: optimizer/RNG reset, '
                  f'{self.cfg.restart_warmup}-update LR ramp; not an exact continuation.')

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        if self.masters:
            self.masters.zero_grad()

    def run(self, max_updates=None):
        c = self.cfg
        self.model.train()
        self.zero_grad()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        last_log, window_tokens = time.perf_counter(), 0
        overflow_retries = 0
        stop_step = min(c.steps, self.step + max_updates) if max_updates is not None else c.steps
        try:
            while self.step < stop_step:
                update_started = time.perf_counter()
                data_seconds = 0.0
                lr = self.learning_rate()
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                ce_total, aux_total, counts = 0.0, 0.0, None
                batch_rng_start = self.batches.rng.get_state()
                for _ in range(c.accumulation):
                    data_started = time.perf_counter()
                    x, y = self.batches.get('train', self.device)
                    data_seconds += time.perf_counter() - data_started
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
                if self.masters:
                    self.masters.copy_gradients()
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.optim_parameters, c.grad_clip)
                if not bool(torch.isfinite(norm)):
                    if self.scaler.is_enabled() and overflow_retries < 8:
                        # unscale_ registered infs: scaler.step skips the optimizer update.
                        # Restore sampler so the smaller scale retries the same microbatches.
                        old_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.zero_grad()
                        self.batches.rng.set_state(batch_rng_start)
                        overflow_retries += 1
                        print(f'FP16 overflow: retry {overflow_retries}, scale {old_scale} -> {self.scaler.get_scale()}')
                        continue
                    raise FloatingPointError('Persistent non-finite gradients. Reduce LR or use BF16 if available.')
                overflow_retries = 0
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.masters:
                    self.masters.copy_to_model()
                self.zero_grad()
                self.step += 1
                seen = c.microbatch * c.sequence_length * c.accumulation
                self.tokens_seen += seen
                window_tokens += seen
                self.last_counts = counts
                self.routing_total += counts
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                update_seconds = time.perf_counter() - update_started
                record = {'step': self.step, 'train_ce': ce_total, 'router_aux': aux_total,
                          'lr': lr, 'tokens_seen': self.tokens_seen, 'grad_norm': float(norm),
                          'tokens_per_second_training_step': seen / max(update_seconds, 1e-6),
                          'data_loading_seconds': data_seconds,
                          'unselected_expert_fraction_phase': float((self.routing_total == 0).float().mean()),
                          'max_expert_to_mean_load_phase': float((self.routing_total.max(-1).values / self.routing_total.mean(-1).clamp_min(1)).max()),
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
                with self.log_path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                if self.step % c.save_every == 0:
                    self.save()
        except KeyboardInterrupt:
            # If interrupt occurred inside accumulation, partial gradients are not saved.
            self.zero_grad()
            print('Interrupted: saving the last completed optimizer step; partial microbatch RNG may have advanced.')
        except torch.cuda.OutOfMemoryError:
            print('CUDA OOM: reduce sequence_length/microbatch and start a NEW run; leave checkpointing enabled.')
            raise
        if self.last_saved_step != self.step:
            self.save()
        return self.evaluate()

    def export_weights(self):
        path = self.output / 'model.pt'
        return self._write(self.snapshot(), path, self.disk_budget_gib)
