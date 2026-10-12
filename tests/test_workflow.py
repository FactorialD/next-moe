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


def test_fp16_master_session_exact_and_warm(tmp_path):
    torch.set_num_threads(1)
    data = tiny_corpus(tmp_path)
    cfg = TrainConfig(steps=3, sequence_length=8, accumulation=1, precision='fp16_master',
                      warmup=1, eval_batches=1, eval_every=3, save_every=3, log_every=1)
    persistent, local = tmp_path/'drive', tmp_path/'local'
    trainer = Trainer(ModelConfig.tiny(), cfg, data, persistent, device='cpu',
                      checkpoint_dir=local, persistent_weights=True, disk_budget_gib=.1)
    assert all(p.dtype == torch.float16 for p in trainer.model.parameters())
    trainer.run(max_updates=2)
    assert trainer.step == 2
    assert (persistent/'weights.pt').exists() and not (persistent/'last.pt').exists()
    assert (local/'last.pt').exists()
    assert all(p.dtype == torch.float32 for p in trainer.optim_parameters)
    assert all(s['exp_avg'].dtype == torch.float32 for s in trainer.optimizer.state.values())
    exact = Trainer(ModelConfig.tiny(), cfg, data, tmp_path/'exact', device='cpu', resume=local/'last.pt')
    for a,b in zip(trainer.masters.master,exact.masters.master):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    trainer.run(max_updates=1); exact.run(max_updates=1)
    for a,b in zip(trainer.masters.master,exact.masters.master):
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    warm_cfg = replace(cfg, steps=5, precision='fp32')
    warm = Trainer(ModelConfig.tiny(), warm_cfg, data, tmp_path/'warm', device='cpu',
                   resume=persistent/'weights.pt', resume_mode='warm')
    assert not warm.optimizer.state and warm.step == 3 and warm.restart_step == 3
    assert warm.learning_rate() < cfg.learning_rate/10
    warm.run(max_updates=1)
    assert warm.step == 4


def test_quota_preserves_previous_snapshot(tmp_path):
    import pytest
    data = tiny_corpus(tmp_path)
    cfg = TrainConfig(steps=1, sequence_length=8, accumulation=1, eval_batches=1)
    trainer = Trainer(ModelConfig.tiny(), cfg, data, tmp_path/'drive', device='cpu',
                      checkpoint_dir=tmp_path/'local', persistent_weights=True, disk_budget_gib=.1)
    trainer.run()
    path = tmp_path/'drive/weights.pt'
    previous = path.read_bytes()
    trainer.disk_budget_gib = path.stat().st_size / 2**30
    with pytest.raises(OSError,match='budget exceeded'):
        trainer.save()
    assert path.read_bytes() == previous
    assert not (tmp_path/'drive/weights.pt.tmp').exists()


def test_corpus_expansion_preserves_tokenizer_and_validation(tmp_path):
    data = tiny_corpus(tmp_path)
    source = tmp_path/'texts'
    for lang in ['uk','en']:
        for i in range(6,12):
            (source/lang/f'{i}.txt').write_text((f'New {lang} article number {i}. Україна and science. ')*20)
    expanded = tmp_path/'expanded'
    old = json.loads((data/'metadata.json').read_text())
    origin = json.loads((data/'documents.json').read_text())['en']['val'][0]['id']
    (source/'uk/00-alias-en-validation.txt').write_text((source/'en'/origin).read_text())
    new = prepare_data(expanded,'local',source,documents_per_language=12,validation_documents=2,
                       vocab_size=512,tokenizer_path=data/'tokenizer.json',
                       fixed_validation_metadata=data/'metadata.json')
    assert (expanded/'tokenizer.json').read_bytes() == (data/'tokenizer.json').read_bytes()
    for lang in ['uk','en']:
        assert new['document_hashes'][lang]['val'] == old['document_hashes'][lang]['val']
        assert set(new['document_hashes'][lang]['val']).isdisjoint(new['document_hashes'][lang]['train'])
        assert (expanded/f'val_{lang}.bin').read_bytes() == (data/f'val_{lang}.bin').read_bytes()
        assert new['document_counts'][lang]['train'] == 10


def test_legacy_fp32_checkpoint_warm_migration(tmp_path):
    data = tiny_corpus(tmp_path)
    cfg = TrainConfig(steps=1, sequence_length=8, accumulation=1, warmup=1, eval_batches=1)
    old = Trainer(ModelConfig.tiny(),cfg,data,tmp_path/'old',device='cpu')
    old.run()
    path = tmp_path/'old/last.pt'
    saved = torch.load(path,map_location='cpu',weights_only=True)
    for name in ('kind','tokenizer_signature','master_weights','restart_step','routing_total'):
        saved.pop(name,None)
    saved['train_config'].pop('precision');saved['train_config'].pop('restart_warmup')
    legacy = tmp_path/'legacy.pt';torch.save(saved,legacy)
    new = Trainer(ModelConfig.tiny(),replace(cfg,steps=2,precision='fp16_master'),data,
                  tmp_path/'new',device='cpu',resume=legacy,resume_mode='warm')
    for (name,_),master in zip(new.model.named_parameters(),new.masters.master):
        torch.testing.assert_close(master,saved['model'][name],atol=0,rtol=0)
    assert new.step == 1 and new.tokens_seen == 8
    new.run(max_updates=1)
    assert new.step == 2
