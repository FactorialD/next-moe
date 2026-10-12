import json
from dataclasses import replace
import numpy as np
import torch
from next_moe.model import ModelConfig
from next_moe.data import prepare_data
from next_moe.train import TrainConfig, Trainer


def tiny_corpus(tmp_path):
    source = tmp_path / 'texts'
    for lang, base in [('uk', 'Україна має цікаву історію. Її міста і люди. '),
                       ('en', 'The world has a rich history. Cities and people. ')]:
        (source / lang).mkdir(parents=True)
        for i in range(6):
            (source / lang / f'{i}.txt').write_text((base + f'Document {i}. ') * 20)
    destination = tmp_path / 'data'
    prepare_data(destination, 'local', source, documents_per_language=6,
                 validation_documents=2, vocab_size=512)
    return destination


def test_prepare_split_roundtrip_and_resume(tmp_path):
    torch.set_num_threads(1)
    data = tiny_corpus(tmp_path)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    text = 'Україна: ї, і, й, є, ґ! Ukrainian English.'
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    metadata = json.loads((data / 'metadata.json').read_text())
    train_hashes = {h for lang in ('uk', 'en') for h in metadata['document_hashes'][lang]['train']}
    val_hashes = {h for lang in ('uk', 'en') for h in metadata['document_hashes'][lang]['val']}
    assert train_hashes.isdisjoint(val_hashes)
    cfg = TrainConfig(steps=2, sequence_length=8, microbatch=1, accumulation=1,
                      warmup=1, eval_every=2, eval_batches=1, save_every=2, log_every=1)
    trainer = Trainer(ModelConfig.tiny(), cfg, data, tmp_path / 'run', device='cpu')
    metrics = trainer.run()
    assert all(np.isfinite(v) for v in metrics.values())
    weights = trainer.export_weights()
    assert weights.exists()
    resumed = Trainer(ModelConfig.tiny(), cfg, data, tmp_path / 'resume',
                      device='cpu', resume=tmp_path / 'run/last.pt')
    assert resumed.step == 2 and resumed.tokens_seen == 16
    torch.testing.assert_close(trainer.batches.rng.get_state(), resumed.batches.rng.get_state())
    for a, b in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b)
    # With the same schedule, the next update must reproduce uninterrupted training.
    trainer.cfg = replace(cfg, steps=3)
    trainer.run()
    resumed.cfg = replace(cfg, steps=3)
    resumed.run()
    for a, b in zip(trainer.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    tokens = torch.tensor([[4, 5]])
    generated = resumed.model.generate(tokens, max_new_tokens=2, temperature=0, context=8, valid_vocab=512)
    assert generated.shape[1] > 2
