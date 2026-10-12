"""Diagnostic CPU run; synthetic texts are NOT the selected Wikipedia corpus."""
from pathlib import Path
import argparse
import json
import resource
import time
import torch
from next_moe.data import prepare_data
from next_moe.model import ModelConfig
from next_moe.train import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/cloud-smoke-2026-10-12'))
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--sequence-length', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    source = args.output / 'synthetic-texts'
    texts = {
        'uk': ['Київ — місто в Україні. Дніпро протікає через Київ. Українська мова має літери і, ї, є та ґ. ',
               'Пам’ять моделі зберігає асоціації слів. Люди читають книжки, вивчають мови та ставлять запитання. '],
        'en': ['Kyiv is a city in Ukraine. The Dnipro flows through Kyiv. People read books and learn languages. ',
               'A model predicts the next token. Memory stores associations. Scientists study the world and ask questions. '],
    }
    for lang, passages in texts.items():
        (source / lang).mkdir(parents=True, exist_ok=True)
        for i in range(12):
            (source / lang / f'{i:02d}.txt').write_text((passages[i % 2] + f'Document {i}. ') * 12)
    data = args.output / 'data'
    if not (data / 'metadata.json').exists():
        prepare_data(data, 'local', source, documents_per_language=12,
                     validation_documents=2, vocab_size=65536)
    recipe = TrainConfig(steps=args.steps, sequence_length=args.sequence_length, microbatch=1,
                         accumulation=1, eval_every=50, eval_batches=2, save_every=100, log_every=10)
    trainer = Trainer(ModelConfig(), recipe, data, args.output / 'run', device='cpu')
    before = trainer.evaluate()
    start = time.perf_counter()
    after = trainer.run()
    elapsed = time.perf_counter() - start
    count = sum(p.numel() for p in trainer.model.parameters())
    # Reload all optimizer/model/RNG states, then execute another optimizer update.
    # Release the first instance to avoid keeping two full optimizer copies alive.
    after_weights = {name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()}
    expected_rng = trainer.batches.rng.get_state().clone()
    del trainer
    import gc
    gc.collect()
    resumed_recipe = TrainConfig(**{**recipe.__dict__, 'steps': args.steps + 1, 'eval_every': args.steps + 1,
                                   'save_every': args.steps + 1})
    resumed = Trainer(ModelConfig(), resumed_recipe, data, args.output / 'run', device='cpu',
                      resume=args.output / 'run/last.pt')
    for name, weight in resumed.model.state_dict().items():
        torch.testing.assert_close(weight, after_weights[name], atol=0, rtol=0)
    torch.testing.assert_close(resumed.batches.rng.get_state(), expected_rng)
    del after_weights
    resumed.run()
    report = {'kind': 'CPU full-profile diagnostic on synthetic local UK/EN; NOT Wikipedia pretraining',
              'parameters': count, 'device': 'cpu', 'threads': 4, 'sequence_length': args.sequence_length,
              'microbatch': 1, 'accumulation': 1, 'steps': args.steps,
              'processed_training_tokens': args.steps * args.sequence_length,
              'elapsed_seconds_including_eval_and_checkpoints': elapsed,
              'average_tokens_per_second': args.steps * args.sequence_length / elapsed,
              'peak_process_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              'before_validation': before, 'after_validation': after,
              'checkpoint_size_GiB': (args.output / 'run/last.pt').stat().st_size / 2**30,
              'resume_parameters_equal': True, 'resume_sampler_equal': True,
              'resume_completed_step': resumed.step,
              'note': 'Changed total steps by one for resume check; future cosine schedule changes.'}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('REPORT',json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
