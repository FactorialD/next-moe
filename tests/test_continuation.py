import ast
import json
from pathlib import Path
import random
import hashlib
import numpy as np
import nbformat
import torch
import pytest
from next_moe.model import ModelConfig
from next_moe.train import Trainer, TrainConfig
from next_moe.corpus import prepare_continuation_data, NearDuplicates, install_phase_schedule
from test_workflow import tiny_corpus

ROOT=Path(__file__).parents[1]


def source_records():
    result={}
    for source_index, source in enumerate(('web_uk','pdf_uk','web_en','pdf_en')):
        rng=random.Random(500+source_index)
        docs=[]
        for i in range(60):
            words=[''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=10)) for _ in range(45)]
            docs.append({'text':' '.join(words), 'id':f'{source}/{i}', 'url':f'https://example.org/{source}/{i}'})
        result[source]=docs
    return result


def test_corpus_tokenizer_holdouts_and_cache(tmp_path):
    old=tiny_corpus(tmp_path)
    records=source_records()
    # Inject protected validation texts into every reader; they must be rejected.
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(old/'tokenizer.json'))
    ids=np.fromfile(old/'val_uk.bin',dtype=np.int32)
    end=int(np.flatnonzero(ids==tok.token_to_id('<eos>'))[0])
    protected=tok.decode(ids[:end].tolist())
    for docs in records.values(): docs.insert(0,{'text':protected,'id':'protected'})
    records['pdf_uk'].insert(0, {'text':records['pdf_uk'][-1]['text'],
                                'id':'wrong-language', 'per_page_languages':['eng_Latn']})
    out=tmp_path/'new'
    kwargs=dict(tokens_per_language=2000,validation_documents=2,record_sources=records)
    meta=prepare_continuation_data(out,old,**kwargs)
    assert (out/'tokenizer.json').read_bytes()==(old/'tokenizer.json').read_bytes()
    for lang in ('uk','en'):
        assert (out/f'val_{lang}.bin').read_bytes()==(old/f'val_{lang}.bin').read_bytes()
        assert meta['tokens'][f'train_{lang}']>=2000
        assert meta['document_counts'][lang]['val_new']==2
        assert meta['source_training_tokens'][lang]['web']>=1600
        assert meta['source_training_tokens'][lang]['pdf']>=400
        assert meta['rejected'][lang]['duplicate_or_holdout_match']>=2
    manifest=[json.loads(x) for x in (out/'documents.jsonl').read_text().splitlines()]
    assert not any(x['id']=='protected' for x in manifest)
    assert not any(x['id']=='wrong-language' for x in manifest)
    assert meta['rejected']['uk']['mixed_pdf_languages']==1
    assert prepare_continuation_data(out,old,**kwargs)==meta
    with pytest.raises(ValueError,match='recipe differs'):
        prepare_continuation_data(out,old,**{**kwargs,'tokens_per_language':3000})
    with (out/'train_en.bin').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='file changed'):
        prepare_continuation_data(out,old,**kwargs)


def test_phase_lr_uses_tokens_not_global_step(tmp_path):
    torch.set_num_threads(1)
    data=tiny_corpus(tmp_path)
    t=Trainer(ModelConfig.tiny(),TrainConfig(steps=500,warmup=1,learning_rate=1e-4),data,tmp_path/'run',device='cpu')
    t.tokens_seen=30_000_000;t.step=200;t.restart_step=200
    install_phase_schedule(t,30_000_000,50_000_000)
    assert t.learning_rate()==pytest.approx(1e-6)
    t.step=300;t.tokens_seen=40_000_000
    assert t.learning_rate()==pytest.approx(5.5e-5)
    t.tokens_seen=50_000_000
    assert t.learning_rate()==pytest.approx(1e-5)


def test_continuation_notebook_offline(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    old=tiny_corpus(tmp_path)
    storage=tmp_path/'drive';base=storage/'checkpoint-30m-frozen';base.mkdir(parents=True)
    import shutil,sys,math,time,gc,subprocess,os
    shutil.copytree(old,base/'data-v2')
    t=Trainer(ModelConfig.tiny(),TrainConfig(steps=1,sequence_length=8,accumulation=1,
               warmup=1,eval_batches=1,eval_every=1,save_every=1),old,tmp_path/'oldrun',device='cpu')
    t.run();t.tokens_seen=30_000_000;t.step=6000
    torch.save(t.snapshot(),base/'weights.pt')
    nb=nbformat.read(ROOT/'notebooks/next_moe_30m_to_50m_webpdf.ipynb',as_version=4)
    nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    for s in codes:ast.parse(s)
    env={'torch':torch,'Path':Path,'sys':sys,'json':json,'math':math,'time':time,'gc':gc,
         'shutil':shutil,'hashlib':hashlib,'os':os,'subprocess':subprocess}
    params=codes[1].replace('USE_DRIVE = True','USE_DRIVE = False')
    params=params.replace("'/content/drive/MyDrive/next-moe'",repr(str(storage)))
    params=params.replace("'/content/next-moe-local'",repr(str(tmp_path/'local')))
    params=params.replace("'/content/next-moe-continuation-code'",repr(str(tmp_path/'code')))
    exec(params,env)
    env.update(ADDITIONAL_TOKENS=64,TOKENS_PER_LANGUAGE=2000,NEW_VALIDATION_DOCUMENTS=2,
               SEQ_LEN=8,MICROBATCH=2,GRAD_ACCUM=2,RUN_EFFICIENCY_BENCHMARK=False)
    exec(codes[2],env)
    # Verify all embedded source files, not merely notebook syntax.
    for name,text in env['SOURCES'].items():assert text==(ROOT/name).read_text()
    import next_moe.corpus as corpus
    records=source_records()
    def stream(spec,seed,shuffle_buffer):
        kind='pdf' if 'pdf' in spec['dataset'] else 'web'
        lang='uk' if spec['config']=='ukr_Cyrl' else 'en'
        return iter(records[f'{kind}_{lang}'])
    monkeypatch.setattr(corpus,'stream_source',stream)
    for i in (3,4,5,6,7,8,9):exec(codes[i],env)
    assert env['trainer'].tokens_seen==30_000_064
    assert env['trainer'].step==6002
    assert set(env['final_metrics'])=={'val_ce_uk','val_ce_en','val_ppl_uk','val_ppl_en',
                                      'val_new_ce_uk','val_new_ce_en','val_new_ppl_uk','val_new_ppl_en'}
    assert (env['RUN_DIR']/'weights.pt').exists()
    # Reopening exactly preserves phase target, optimizer and LR schedule.
    for i in (3,4,5):exec(codes[i],env)
    assert env['trainer'].step==6002 and env['trainer'].optimizer.state
    assert env['phase']['target_tokens']==30_000_064
