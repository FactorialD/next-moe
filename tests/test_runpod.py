import ast,hashlib,json,sys,math,time,gc,shutil,subprocess,os,zipfile
from pathlib import Path
import nbformat
import torch
import pytest
from next_moe.train import Trainer,TrainConfig
from next_moe.model import ModelConfig
from next_moe.corpus import prepare_continuation_data
from next_moe.replay import prepare_replay_data,install_replay_sampling
from scripts.runpod_transfer import make_runpod_bundle
from test_workflow import tiny_corpus
from test_continuation import source_records
ROOT=Path(__file__).parents[1]

@pytest.mark.parametrize('full',[False,True])
def test_transfer_and_runpod_cells(tmp_path,full):
    torch.set_num_threads(1)
    wiki=tiny_corpus(tmp_path);web=tmp_path/'web';data=tmp_path/'replay'
    prepare_continuation_data(web,wiki,tokens_per_language=2000,validation_documents=2,record_sources=source_records())
    prepare_replay_data(data,web,wiki)
    cfg=TrainConfig(steps=3,sequence_length=8,microbatch=2,accumulation=2,warmup=1,
                   eval_batches=1,eval_every=3,save_every=3)
    t=Trainer(ModelConfig.tiny(),cfg,data,tmp_path/'colab',device='cpu')
    install_replay_sampling(t);t.run(max_updates=1)
    (t.output/'phase.json').write_text(json.dumps({'initial_tokens':0,'target_tokens':96}))
    bundle=make_runpod_bundle(t,tmp_path/'transfer.zip',include_optimizer=full)
    input_dir=tmp_path/'input'
    with zipfile.ZipFile(bundle) as z:z.extractall(input_dir)
    assert (input_dir/'last.pt').exists()==full
    nb=nbformat.read(ROOT/'notebooks/next_moe_runpod.ipynb',as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    for source in codes:ast.parse(source)
    env={'torch':torch,'native_bf16':False,'Path':Path,'sys':sys,'json':json,'math':math,
         'time':time,'gc':gc,'shutil':shutil,'hashlib':hashlib,'os':os,'subprocess':subprocess}
    params=codes[1].replace("'/workspace/next-moe-transfer'",repr(str(input_dir)))
    params=params.replace("'/workspace/next-moe-runs'",repr(str(tmp_path/'volume')))
    params=params.replace('if shutil.disk_usage(RUN_DIR).free<required_free_gib*2**30:', 'if False:')
    exec(params,env);env.update(SEQ_LEN=16,VALIDATION_SEQ_LEN=8,MICROBATCH=1,GRAD_ACCUM=2,
               RUN_BENCHMARK=False,RUN_HOURS=1 if full else None)
    exec(codes[2],env)
    for name,text in env['SOURCES'].items():assert text==(ROOT/name).read_text()
    for i in (3,4):exec(codes[i],env)
    assert env['mode']==('exact' if full else 'warm')
    if full:
        for a,b in zip(t.optim_parameters,env['trainer'].optim_parameters):torch.testing.assert_close(a,b)
        torch.testing.assert_close(t.batches.rng.get_state(),env['trainer'].batches.rng.get_state())
    for i in range(5,len(codes)):exec(codes[i],env)
    assert env['trainer'].tokens_seen==96 and env['phase']['target_tokens']==96
    assert (env['RUN_DIR']/'last.pt').exists()
    for i in (3,4):exec(codes[i],env)
    assert env['mode']=='exact' and env['trainer'].step==3
    assert env['phase']['target_tokens']==96
    # A changed bundle must be rejected before training.
    (input_dir/'weights.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='Transfer changed'):exec(codes[3],env)
