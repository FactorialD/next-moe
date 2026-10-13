import json
from dataclasses import replace
import torch
from next_moe.model import ModelConfig
from next_moe.train import TrainConfig,Trainer
from scripts.colab_efficiency import benchmark_batch_layout,configure_live_batch,extend_token_budget
from test_workflow import tiny_corpus


def test_non_destructive_benchmark_and_fixed_validation(tmp_path):
    torch.set_num_threads(1)
    data=tiny_corpus(tmp_path)
    cfg=TrainConfig(steps=30,sequence_length=8,accumulation=4,warmup=1,eval_batches=2,
                    eval_every=20,save_every=20,log_every=10)
    trainer=Trainer(ModelConfig.tiny(),cfg,data,tmp_path/'run',device='cpu')
    trainer.run(max_updates=1)
    model={k:v.clone() for k,v in trainer.model.state_dict().items()}
    moments={p:{k:v.clone() for k,v in s.items() if isinstance(v,torch.Tensor)}
             for p,s in trainer.optimizer.state.items()}
    rng=trainer.batches.rng.get_state().clone()
    before=trainer.evaluate()
    logs=trainer.log_path.read_bytes()
    results=benchmark_batch_layout(trainer,warmup_updates=1,measured_updates=2,apply_best=False)
    assert len(results)==4 and trainer.step==1 and trainer.tokens_seen==32
    assert all(r['effective_tokens_per_update']==32 for r in results)
    assert all(r['median_training_tokens_per_second']>0 for r in results)
    for k,v in trainer.model.state_dict().items():torch.testing.assert_close(v,model[k],atol=0,rtol=0)
    for p,s in trainer.optimizer.state.items():
        for k,v in moments[p].items():torch.testing.assert_close(v,s[k],atol=0,rtol=0)
    torch.testing.assert_close(rng,trainer.batches.rng.get_state())
    assert trainer.log_path.read_bytes()==logs
    configure_live_batch(trainer,4,1,False)
    after=trainer.evaluate()
    for k in before:assert abs(before[k]-after[k])<1e-4
    assert trainer.cfg.microbatch==4 and trainer.cfg.accumulation==1
    assert not trainer.model.config.checkpoint_blocks
    assert trainer.step==1 and trainer.tokens_seen==32
    optimizer_id=id(trainer.optimizer)
    extend_token_budget(trainer,3200)
    assert id(trainer.optimizer)==optimizer_id and trainer.optimizer.state
    assert trainer.cfg.steps==100 and trainer.restart_step==1


def test_benchmark_restores_master_weights(tmp_path):
    torch.set_num_threads(1)
    data=tiny_corpus(tmp_path)
    cfg=TrainConfig(steps=20,sequence_length=8,accumulation=2,warmup=1,eval_batches=1,
                    precision='fp16_master',eval_every=20,save_every=20)
    trainer=Trainer(ModelConfig.tiny(),cfg,data,tmp_path/'run',device='cpu')
    trainer.run(max_updates=1)
    values=[p.clone() for p in trainer.masters.master]
    scaler=trainer.scaler.state_dict().copy()
    results=benchmark_batch_layout(trainer,units_per_hour=None,warmup_updates=0,measured_updates=1,apply_best=False)
    assert all(r['training_tokens_per_unit'] is None for r in results)
    for p,v in zip(trainer.masters.master,values):torch.testing.assert_close(p,v,atol=0,rtol=0)
    assert trainer.scaler.state_dict()==scaler
