import hashlib
from pathlib import Path
import torch
from next_moe.train import TrainConfig,Trainer
from next_moe.model import ModelConfig
from test_workflow import tiny_corpus
from scripts.training_control import install_interrupt_safe_run,run_for_hours


def test_stop_saves_and_exits_outer_loop_without_changing_source(tmp_path):
    torch.set_num_threads(1)
    data=tiny_corpus(tmp_path)
    trainer=Trainer(ModelConfig.tiny(),TrainConfig(steps=5,sequence_length=8,accumulation=1,
        warmup=1,eval_batches=1,eval_every=5,save_every=5),data,tmp_path/'run',device='cpu')
    source=Path(__import__('next_moe.train',fromlist=['Trainer']).__file__)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    get=trainer.batches.get;calls=[]
    def interrupted(split,*args,**kwargs):
        if split=='train':
            calls.append(trainer.step)
            if trainer.step==1:raise KeyboardInterrupt
        return get(split,*args,**kwargs)
    trainer.batches.get=interrupted
    report=run_for_hours(trainer,1,chunk_updates=3)
    assert report['reason']=='interrupted' and trainer.step==1 and calls==[0,1]
    saved=torch.load(trainer.checkpoint_dir/'last.pt',map_location='cpu',weights_only=True)
    assert saved['step']==1 and saved['optimizer']['state']
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()
    # Patch idempotence and continuation after a deliberate Stop.
    method=trainer.run;install_interrupt_safe_run(trainer);assert trainer.run is method
    trainer.batches.get=get
    resumed=Trainer(ModelConfig.tiny(),trainer.cfg,data,tmp_path/'resume',device='cpu',resume=trainer.checkpoint_dir/'last.pt')
    a=run_for_hours(trainer,1,2);b=run_for_hours(resumed,1,2)
    assert a['reason']==b['reason']=='goal'
    for x,y in zip(trainer.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
