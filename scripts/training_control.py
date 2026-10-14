"""Interrupt-safe timed training for legacy Trainer.run without changing source hashes."""
import ast
import inspect
import textwrap
import time
import types


def install_interrupt_safe_run(trainer):
    """Keep legacy save-on-interrupt behavior but re-raise instead of returning.

    Patch only this instance's Python method in memory. Model/optimizer/RNG and
    repository/embedded source files remain unchanged, so old exact-resume hashes
    still match. Stop during evaluation/save outside the handler also propagates.
    """
    if getattr(trainer,'_interrupt_safe_run_installed',False):return
    original=type(trainer).run
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    handlers=[node for node in ast.walk(tree) if isinstance(node,ast.ExceptHandler)
              and isinstance(node.type,ast.Name) and node.type.id=='KeyboardInterrupt']
    if len(handlers)!=1:raise RuntimeError('Unexpected Trainer.run structure; no patch applied.')
    handlers[0].body.extend(ast.parse('if self.last_saved_step != self.step:\n    self.save()\nraise').body)
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<interrupt-safe Trainer.run>','exec'),namespace)
    trainer.run=types.MethodType(namespace[original.__name__],trainer)
    trainer._interrupt_safe_run_installed=True


def run_for_hours(trainer,hours,chunk_updates=1000):
    """Interrupt once to exit the whole timed loop after legacy checkpoint saving.

    Deadline checked between chunks, not inside a CUDA operation or save. Keep a
    reservation margin longer than one chunk plus validation/checkpoint time.
    """
    if hours<=0 or chunk_updates<1:raise ValueError('Positive hours and chunk_updates required.')
    install_interrupt_safe_run(trainer)
    deadline=time.monotonic()+hours*3600
    metrics={};reason='goal'
    try:
        while trainer.step<trainer.cfg.steps:
            if time.monotonic()>=deadline:
                reason='time_limit';break
            metrics=trainer.run(max_updates=chunk_updates)
    except KeyboardInterrupt:
        reason='interrupted'
        print('STOP: timed loop stopped; no next training block will start.',flush=True)
        print('Last saved step:',trainer.last_saved_step,'checkpoint:',trainer.checkpoint_dir/'last.pt',flush=True)
    return {'reason':reason,'step':trainer.step,'tokens_seen':trainer.tokens_seen,'last_metrics':metrics}
