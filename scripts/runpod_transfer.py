"""Build a local migration ZIP from a stopped Colab trainer; no Drive copies."""
from pathlib import Path
import hashlib,json,tempfile,zipfile
import torch


def make_runpod_bundle(trainer, destination='/content/next-moe-runpod-transfer.zip', include_optimizer=False):
    """Call after the training cell stops. Compact by default; full exact optional."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError(f'Bundle already exists: {destination}')
    files={}
    with tempfile.TemporaryDirectory(prefix='next-moe-transfer-') as folder:
        folder=Path(folder)
        weights=folder/'weights.pt';torch.save(trainer.snapshot(),weights);files['weights.pt']=weights
        for path in trainer.data_dir.rglob('*'):
            if path.is_file():files['data-replay/'+str(path.relative_to(trainer.data_dir))]=path
        for name in ('phase.json','baseline-validation.json','source_hashes.json','pip-freeze.txt'):
            if (trainer.output/name).exists():files[name]=trainer.output/name
        if trainer.log_path.exists():files['metrics.jsonl']=trainer.log_path
        if include_optimizer:
            # Same completed update for model, FP32 masters, Adam and RNG.
            # Persistent weights may be refreshed by trainer.save().
            files['last.pt']=trainer.save()
        manifest={'step':trainer.step,'tokens_seen':trainer.tokens_seen,
                  'optimizer_included':include_optimizer,'file_sha256':{}}
        with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_STORED) as bundle:
            for name,path in files.items():
                h=hashlib.sha256()
                with path.open('rb') as handle:
                    while chunk:=handle.read(1<<20):h.update(chunk)
                manifest['file_sha256'][name]=h.hexdigest()
                bundle.write(path,name)
            bundle.writestr('transfer-manifest.json',json.dumps(manifest,indent=2))
    print('Bundle:',destination,'GiB:',round(destination.stat().st_size/2**30,3))
    print('Tokens:',trainer.tokens_seen,'optimizer included:',include_optimizer)
    return destination
