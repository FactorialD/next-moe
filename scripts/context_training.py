"""Longer training context with preserved Adam, token batch and short validation."""
from dataclasses import asdict
import json


def configure_training_context(trainer, sequence_length=2048, microbatch=1,
                               accumulation=1, validation_length=512):
    """Restore the saved recipe FIRST, then make an explicit context transition."""
    if min(sequence_length,microbatch,accumulation,validation_length)<1:
        raise ValueError('Positive context/batch lengths required.')
    previous=trainer.cfg.sequence_length*trainer.cfg.microbatch*trainer.cfg.accumulation
    desired=sequence_length*microbatch*accumulation
    if previous!=desired:
        raise ValueError(f'Preserve tokens/update: saved {previous}, requested {desired}. Adjust batch settings.')
    for array in trainer.batches.arrays.values():
        if len(array)<=max(sequence_length,validation_length):raise ValueError('Corpus/holdout too short for context.')
    trainer.cfg.sequence_length=sequence_length
    trainer.cfg.microbatch=microbatch;trainer.cfg.accumulation=accumulation
    trainer.batches.length=sequence_length;trainer.batches.batch=microbatch
    trainer._validation_contexts=(validation_length,sequence_length)
    (trainer.output/'train_config.json').write_text(json.dumps(asdict(trainer.cfg),indent=2))
    (trainer.output/'context-phase.json').write_text(json.dumps({
        'training_sequence_length':sequence_length,'validation_sequence_lengths':list(dict.fromkeys((validation_length,sequence_length))),
        'tokens_per_update':desired,'optimizer_preserved':True,'target_steps':trainer.cfg.steps,
        'note':'Context transition is a recipe change; existing Adam/RNG/LR are preserved.'},indent=2))
    if not getattr(trainer,'_context_evaluation_installed',False):
        original=trainer.evaluate
        def evaluate():
            length,batch=trainer.batches.length,trainer.batches.batch
            short,long=trainer._validation_contexts
            try:
                trainer.batches.batch=1;trainer.batches.length=short
                result=original()
                if long!=short:
                    trainer.batches.length=long
                    result.update({key.replace('val_',f'val_ctx{long}_',1):value
                                   for key,value in original().items()})
                return result
            finally:
                trainer.batches.length=length;trainer.batches.batch=batch
        trainer.evaluate=evaluate;trainer._context_evaluation_installed=True
    print(f'Context transition: train={sequence_length}, validation={validation_length}/{sequence_length}, '
          f'tokens/update={desired}; Adam/RNG/LR/target retained.',flush=True)
