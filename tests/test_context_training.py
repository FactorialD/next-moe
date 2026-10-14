import json
import torch
from next_moe.train import Trainer,TrainConfig
from next_moe.model import ModelConfig
from next_moe.corpus import prepare_continuation_data,install_domain_evaluation
from next_moe.replay import prepare_replay_data,install_replay_sampling
from scripts.context_training import configure_training_context
from scripts.colab_efficiency import fix_evaluation_microbatch
from test_workflow import tiny_corpus
from test_continuation import source_records


def test_context_change_keeps_adam_rng_and_short_validation(tmp_path):
    torch.set_num_threads(1)
    wiki=tiny_corpus(tmp_path);web=tmp_path/'web';data=tmp_path/'replay'
    prepare_continuation_data(web,wiki,tokens_per_language=2000,validation_documents=2,record_sources=source_records())
    prepare_replay_data(data,web,wiki)
    cfg=TrainConfig(steps=3,sequence_length=8,microbatch=4,accumulation=1,
                   eval_batches=1,eval_every=3,save_every=3,warmup=1)
    t=Trainer(ModelConfig.tiny(),cfg,data,tmp_path/'run',device='cpu')
    install_replay_sampling(t);fix_evaluation_microbatch(t,1);install_domain_evaluation(t);t.run(max_updates=1)
    baseline=t.evaluate();rng=t.batches.rng.get_state().clone();optimizer=t.optimizer
    moments={p:{k:v.clone() for k,v in state.items() if isinstance(v,torch.Tensor)}
             for p,state in t.optimizer.state.items()}
    rate=t.learning_rate();step=t.step;tokens=t.tokens_seen
    configure_training_context(t,16,2,1,8)
    assert t.optimizer is optimizer and t.step==step and t.tokens_seen==tokens and t.learning_rate()==rate
    torch.testing.assert_close(t.batches.rng.get_state(),rng)
    for p,state in t.optimizer.state.items():
        for key,value in moments[p].items():torch.testing.assert_close(state[key],value,rtol=0,atol=0)
    metrics=t.evaluate()
    for key,value in baseline.items():assert abs(metrics[key]-value)<1e-6
    assert 'val_ctx16_ce_uk' in metrics and 'val_ctx16_new_ce_en' in metrics
    assert t.batches.length==16 and t.batches.batch==2
    t.run(max_updates=1)
    assert t.tokens_seen==tokens+32
    restored=Trainer(ModelConfig.tiny(),t.cfg,data,tmp_path/'restored',device='cpu',resume=t.checkpoint_dir/'last.pt')
    install_replay_sampling(restored);install_domain_evaluation(restored)
    configure_training_context(restored,16,2,1,8)
    t.run();restored.run()
    for a,b in zip(t.model.parameters(),restored.model.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
