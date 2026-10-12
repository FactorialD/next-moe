"""Live Colab efficiency tools. No model reload, external services or multi-GPU.

Benchmark executes short optimizer updates then restores model/master/optimizer,
progress and RNG after every candidate. CPU backup uses ~full-checkpoint RAM.
"""
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
import statistics
import tempfile
import torch


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def fix_evaluation_microbatch(trainer, batch=1):
    """Keep fixed validation windows when changing the TRAIN microbatch."""
    if getattr(trainer, '_fixed_evaluation_installed', False):
        trainer._evaluation_batch = batch
        return
    original = trainer.evaluate
    trainer._evaluation_batch = batch
    def fixed_evaluate():
        previous = trainer.batches.batch
        trainer.batches.batch = trainer._evaluation_batch
        try:
            return original()
        finally:
            trainer.batches.batch = previous
    trainer.evaluate = fixed_evaluate
    trainer._fixed_evaluation_installed = True


def configure_live_batch(trainer, microbatch, accumulation, checkpoint_blocks=None):
    """Repartition the SAME tokens/update, retaining optimizer and LR schedule."""
    if min(microbatch, accumulation) < 1:
        raise ValueError('Batch sizes must be positive integers.')
    if microbatch * accumulation != trainer.cfg.microbatch * trainer.cfg.accumulation:
        raise ValueError('This helper preserves effective batch; microbatch × accumulation must stay equal.')
    fix_evaluation_microbatch(trainer, 1)
    trainer.cfg.microbatch, trainer.cfg.accumulation = microbatch, accumulation
    trainer.batches.batch = microbatch
    if checkpoint_blocks is not None:
        trainer.model.config.checkpoint_blocks = checkpoint_blocks
    (trainer.output/'train_config.json').write_text(json.dumps(asdict(trainer.cfg), indent=2))
    (trainer.output/'model_config.json').write_text(json.dumps(trainer.model.config.to_dict(), indent=2))


def benchmark_batch_layout(trainer, units_per_hour=8.9, warmup_updates=2,
                           measured_updates=10, apply_best=True):
    """Non-destructive timing comparison with equal effective batch.

    Uses host RAM for a backup; short candidate updates are rolled back. No Drive
    writes or validation during timing. Full Adam states must already be allocated.
    Absolute CUDA speed must be measured on the user's actual GPU.
    """
    if units_per_hour <= 0 or measured_updates < 1 or warmup_updates < 0:
        raise ValueError('Invalid benchmark settings.')
    updates = warmup_updates + measured_updates
    if trainer.cfg.steps - trainer.step < updates:
        raise ValueError('Too close to the current training goal for this benchmark. Finish the stage first.')
    trainer.zero_grad()
    if not trainer.optimizer.state:
        raise ValueError('Run at least one real update first so Adam states are allocated.')
    fix_evaluation_microbatch(trainer, 1)
    base_cfg = copy.deepcopy(trainer.cfg)
    base_checkpoint = trainer.model.config.checkpoint_blocks
    saved = {'model': _cpu_copy(trainer.model.state_dict()),
             'master': _cpu_copy([p for p in trainer.masters.master]) if trainer.masters else None,
             'optimizer': _cpu_copy(trainer.optimizer.state_dict()),
             'scaler': copy.deepcopy(trainer.scaler.state_dict()),
             'step': trainer.step, 'tokens': trainer.tokens_seen,
             'batch_rng': trainer.batches.rng.get_state().clone(),
             'torch_rng': torch.get_rng_state().clone(),
             'cuda_rng': torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else [],
             'routing': _cpu_copy(trainer.routing_total),
             'last_counts': _cpu_copy(trainer.last_counts),
             'last_saved_step': trainer.last_saved_step,
             'restart_step': trainer.restart_step, 'training': trainer.model.training}
    original_save, original_evaluate, original_log = trainer.save, trainer.evaluate, trainer.log_path
    effective = base_cfg.microbatch * base_cfg.accumulation
    microbatches = sorted(set([base_cfg.microbatch] + [v for v in (1, 2, 4) if effective % v == 0]))
    candidates = [(m, effective // m, base_checkpoint) for m in microbatches]
    if base_checkpoint:
        candidates.append((max(microbatches), effective // max(microbatches), False))
    results = []

    def restore():
        trainer.zero_grad()
        trainer.cfg = copy.deepcopy(base_cfg)
        trainer.batches.batch = base_cfg.microbatch
        trainer.model.config.checkpoint_blocks = base_checkpoint
        trainer.model.load_state_dict(saved['model'])
        if trainer.masters:
            with torch.no_grad():
                for parameter, value in zip(trainer.masters.master, saved['master']):
                    parameter.copy_(value)
        # load_state_dict may reuse CPU tensors (including Adam's step scalar).
        # Give every candidate its own copy so updates cannot mutate the backup.
        trainer.optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        trainer.scaler.load_state_dict(saved['scaler'])
        trainer.step, trainer.tokens_seen = saved['step'], saved['tokens']
        trainer.last_saved_step, trainer.restart_step = saved['last_saved_step'], saved['restart_step']
        trainer.batches.rng.set_state(saved['batch_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        trainer.routing_total.copy_(saved['routing'])
        trainer.last_counts = saved['last_counts'].to(trainer.device) if saved['last_counts'] is not None else None
        trainer.model.train(saved['training'])
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    try:
        with tempfile.TemporaryDirectory(prefix='next-moe-benchmark-') as directory:
            trainer.save = lambda *args, **kwargs: None
            trainer.evaluate = lambda: {}
            for index, (microbatch, accumulation, checkpointing) in enumerate(candidates):
                restore()
                trainer.cfg.microbatch, trainer.cfg.accumulation = microbatch, accumulation
                trainer.batches.batch = microbatch
                trainer.model.config.checkpoint_blocks = checkpointing
                trainer.log_path = Path(directory)/f'{index}.jsonl'
                print(f'BENCHMARK: microbatch={microbatch}, accumulation={accumulation}, '
                      f'block_checkpointing={checkpointing}; updates will be rolled back.')
                trainer.run(max_updates=updates)
                records = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
                rates = [r['tokens_per_second_training_step'] for r in records[warmup_updates:]]
                speed = statistics.median(rates)
                result = {'microbatch': microbatch, 'accumulation': accumulation,
                          'checkpoint_blocks': checkpointing,
                          'effective_tokens_per_update': microbatch*accumulation*trainer.cfg.sequence_length,
                          'median_training_tokens_per_second': speed,
                          'units_per_hour_assumed': units_per_hour,
                          'training_tokens_per_unit': speed*3600/units_per_hour}
                if trainer.device.type == 'cuda':
                    result['peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/2**30
                    result['peak_reserved_GiB'] = torch.cuda.max_memory_reserved()/2**30
                results.append(result)
                print(json.dumps(result))
    finally:
        restore()
        trainer.save, trainer.evaluate, trainer.log_path = original_save, original_evaluate, original_log
    if apply_best and results:
        winner = max(results, key=lambda r: r['median_training_tokens_per_second'])
        configure_live_batch(trainer, winner['microbatch'], winner['accumulation'], winner['checkpoint_blocks'])
        print('Applied fastest measured layout; model/optimizer/progress/RNG restored. '
              'Update MICROBATCH, GRAD_ACCUM and CHECKPOINT_BLOCKS in notebook settings before a future reload.')
    report = {'device': str(trainer.device), 'rolled_back_to_step': trainer.step,
              'tokens_seen': trainer.tokens_seen, 'results': results,
              'scope': 'Training-step timing; exclude eval/save overhead. Not a GPU utilization or quality benchmark.'}
    (trainer.output/'efficiency-benchmark.json').write_text(json.dumps(report, indent=2))
    return results


def extend_token_budget(trainer, target_tokens, ramp_updates=100):
    """Extend goal in the live session, preserving optimizer; ramp avoids LR jump."""
    if target_tokens <= trainer.tokens_seen or ramp_updates < 1:
        raise ValueError('Target must exceed processed tokens; use a positive ramp.')
    batch_tokens = trainer.cfg.sequence_length * trainer.cfg.microbatch * trainer.cfg.accumulation
    trainer.cfg.steps = trainer.step + math.ceil((target_tokens-trainer.tokens_seen)/batch_tokens)
    trainer.restart_step = trainer.step  # LR ramp only; no optimizer reset
    trainer.cfg.restart_warmup = ramp_updates
    (trainer.output/'train_config.json').write_text(json.dumps(asdict(trainer.cfg), indent=2))
    print(f'New goal ~{target_tokens:,} total tokens / step {trainer.cfg.steps}; '
          f'optimizer retained, {ramp_updates}-update LR ramp.')
