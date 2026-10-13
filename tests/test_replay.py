import ast,json,hashlib
from pathlib import Path
import numpy as np
import torch
import pytest
import nbformat
from next_moe.model import ModelConfig
from next_moe.train import Trainer,TrainConfig
from next_moe.corpus import prepare_continuation_data
from next_moe.replay import prepare_replay_data,install_replay_sampling
from test_workflow import tiny_corpus
from test_continuation import source_records
ROOT=Path(__file__).parents[1]


def test_replay_sampler_restore_and_validation(tmp_path):
    torch.set_num_threads(1)
    wiki=tiny_corpus(tmp_path)
    web=tmp_path/'web';out=tmp_path/'replay'
    prepare_continuation_data(web,wiki,tokens_per_language=2000,validation_documents=2,record_sources=source_records())
    meta=prepare_replay_data(out,web,wiki)
    assert meta['sampling']['wikipedia']==.25
    assert prepare_replay_data(out,web,wiki)==meta
    for lang in ('uk','en'):
        for split in ('val','val_new'):
            assert (out/f'{split}_{lang}.bin').read_bytes()==(web/f'{split}_{lang}.bin').read_bytes()
    t=Trainer(ModelConfig.tiny(),TrainConfig(steps=2,sequence_length=8,microbatch=4,accumulation=1,
        warmup=1,eval_batches=1,eval_every=2,save_every=2),out,tmp_path/'run',device='cpu')
    baseline=t.evaluate();install_replay_sampling(t)
    assert t.evaluate()==baseline
    wiki_windows=set()
    for lang in ('uk','en'):
        ids=np.fromfile(out/f'train_wikipedia_{lang}.bin',dtype=np.int32)
        wiki_windows.update(tuple(row) for row in np.lib.stride_tricks.sliding_window_view(ids,8))
    sampled=[]
    for _ in range(128):
        rows,_=t.batches.get('train','cpu')
        sampled.extend(tuple(row.tolist()) in wiki_windows for row in rows)
    assert .17 < sum(sampled)/len(sampled) < .33
    state=t.batches.rng.get_state().clone()
    x,y=t.batches.get('train','cpu');t.batches.rng.set_state(state)
    x2,y2=t.batches.get('train','cpu');torch.testing.assert_close(x,x2);torch.testing.assert_close(y,y2)
    t.run(max_updates=1)
    clone=Trainer(ModelConfig.tiny(),t.cfg,out,tmp_path/'clone',device='cpu',resume=t.checkpoint_dir/'last.pt')
    install_replay_sampling(clone)
    a=t.batches.get('train','cpu');b=clone.batches.get('train','cpu')
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    with pytest.raises(ValueError,match='recipe changed'):prepare_replay_data(out,web,wiki,.3)


@pytest.mark.parametrize('notebook,starting_tokens,base_name,data_name', [
    ('next_moe_50m_wikipedia_replay.ipynb',50_001_920,'next-moe-30m-plus20m-webpdf-v1','data-stage3'),
    ('next_moe_55m_plus10m_replay.ipynb',55_001_920,'next-moe-50m-plus5m-replay-v1','data-replay'),
    ('next_moe_55m_plus10m_replay.ipynb',55_001_920,'checkpoint-55001920-tokens-frozen','data-replay'),
])
def test_replay_notebook_offline(tmp_path,notebook,starting_tokens,base_name,data_name):
    torch.set_num_threads(1)
    import shutil,sys,math,time,gc,subprocess,os
    wiki=tiny_corpus(tmp_path)
    storage=tmp_path/'drive';base=storage/base_name;base.mkdir(parents=True)
    raw_web=tmp_path/'web'
    prepare_continuation_data(raw_web,wiki,tokens_per_language=2000,validation_documents=2,record_sources=source_records())
    web=base/data_name
    if data_name=='data-replay':prepare_replay_data(web,raw_web,wiki)
    else:shutil.copytree(raw_web,web)
    t=Trainer(ModelConfig.tiny(),TrainConfig(steps=1,sequence_length=8,accumulation=1,warmup=1,
        eval_batches=1,eval_every=1,save_every=1),web,tmp_path/'oldrun',device='cpu')
    t.run();t.tokens_seen=starting_tokens;t.step=25915;torch.save(t.snapshot(),base/'weights.pt')
    nb=nbformat.read(ROOT/'notebooks'/notebook,as_version=4)
    nbformat.validate(nb);codes=[c.source for c in nb.cells if c.cell_type=='code']
    for s in codes:ast.parse(s)
    env={'torch':torch,'Path':Path,'sys':sys,'json':json,'math':math,'time':time,'gc':gc,
         'shutil':shutil,'hashlib':hashlib,'os':os,'subprocess':subprocess}
    params=codes[1].replace('USE_DRIVE = True','USE_DRIVE = False')
    params=params.replace("'/content/drive/MyDrive/next-moe'",repr(str(storage)))
    params=params.replace("'/content/next-moe-local'",repr(str(tmp_path/'local')))
    params=params.replace("'/content/next-moe-continuation-code'",repr(str(tmp_path/'code')))
    params=params.replace("WIKI_DATA_PATH = ''",'WIKI_DATA_PATH = '+repr(str(wiki)))
    exec(params,env)
    env.update(ADDITIONAL_TOKENS=64,SEQ_LEN=8,MICROBATCH=2,GRAD_ACCUM=2)
    exec(codes[2],env)
    for name,source in env['SOURCES'].items():assert source==(ROOT/name).read_text()
    for index in (3,4,5,6,7,8,9):exec(codes[index],env)
    assert env['trainer'].tokens_seen==starting_tokens+64
    assert env['trainer']._replay_sampling_installed
    assert (env['RUN_DIR']/'baseline-validation.json').exists()
    for index in (3,4,5):exec(codes[index],env)
    assert env['trainer'].optimizer.state and env['trainer']._replay_sampling_installed
    assert env['phase']['target_tokens']==starting_tokens+64
