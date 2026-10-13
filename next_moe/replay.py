"""Wikipedia replay for the 50M checkpoint; reuse cached web/PDF and both holdouts."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from tokenizers import Tokenizer
from .corpus import NearDuplicates, digest


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(1<<20):h.update(chunk)
    return h.hexdigest()


def prepare_replay_data(output, web_data, wiki_data, wikipedia_fraction=.25):
    output,web_data,wiki_data=map(Path,(output,web_data,wiki_data))
    if not 0 < wikipedia_fraction < 1:raise ValueError('Replay fraction must be between zero and one.')
    if (web_data/'tokenizer.json').read_bytes() != (wiki_data/'tokenizer.json').read_bytes():
        raise ValueError('Wikipedia tokenizer differs; keep original token IDs.')
    for lang in ('uk','en'):
        if file_hash(web_data/f'val_{lang}.bin') != file_hash(wiki_data/f'val_{lang}.bin'):
            raise ValueError('Wikipedia validation differs from the preserved old holdout.')
    base=json.loads((web_data/'metadata.json').read_text())
    files=['tokenizer.json','documents.jsonl']+[f'{split}_{lang}.bin'
        for lang in ('uk','en') for split in ('train','val','val_new')]
    recipe={'wikipedia_fraction':wikipedia_fraction,
            'web_metadata_sha256':file_hash(web_data/'metadata.json'),
            'web_files_sha256':{name:file_hash(web_data/name) for name in files},
            'wiki_metadata_sha256':file_hash(wiki_data/'metadata.json'),
            'wiki_training_sha256':{lang:file_hash(wiki_data/f'train_{lang}.bin') for lang in ('uk','en')}}
    if (output/'metadata.json').exists():
        meta=json.loads((output/'metadata.json').read_text())
        if meta.get('replay_recipe') != recipe:raise ValueError('Replay recipe changed; use a new RUN_NAME.')
        for name,expected in meta['file_sha256'].items():
            if file_hash(output/name)!=expected:raise ValueError(f'Replay cache changed: {name}')
        return meta
    if output.exists() and any(output.iterdir()):raise ValueError('Incomplete replay directory; use a new directory.')
    output.mkdir(parents=True,exist_ok=True)
    for name in files:shutil.copy2(web_data/name,output/name)
    # New holdouts may overlap documents from the original Wikipedia train pool.
    # Remove those documents before replay, using the same approximate filter.
    tokenizer=Tokenizer.from_file(str(web_data/'tokenizer.json'))
    eos=tokenizer.token_to_id('<eos>');protected=set();near=NearDuplicates()
    for lang in ('uk','en'):
        protected.update(base['document_hashes'][lang].get('val',[]))
        protected.update(base['document_hashes'][lang].get('val_new',[]))
        for split in ('val','val_new'):
            ids=np.fromfile(web_data/f'{split}_{lang}.bin',dtype=np.int32);start=0
            for end in np.flatnonzero(ids==eos):
                text=tokenizer.decode(ids[start:end].tolist(),skip_special_tokens=False).strip()
                if text:protected.add(digest(text));near.duplicate(text,add=True)
                start=end+1
    removed={}
    for lang in ('uk','en'):
        ids=np.fromfile(wiki_data/f'train_{lang}.bin',dtype=np.int32);start=0;removed[lang]=0
        with (output/f'train_wikipedia_{lang}.bin').open('wb') as handle:
            for end in np.flatnonzero(ids==eos):
                text=tokenizer.decode(ids[start:end].tolist(),skip_special_tokens=False).strip()
                if digest(text) in protected or near.duplicate(text):removed[lang]+=1
                else:ids[start:end+1].tofile(handle)
                start=end+1
        if start!=len(ids):raise ValueError('Wikipedia token pool must end in EOS.')
        if (output/f'train_wikipedia_{lang}.bin').stat().st_size==0:raise ValueError('No leak-free Wikipedia replay documents.')
    shutil.copy2(wiki_data/'metadata.json',output/'wikipedia_metadata.json')
    meta={**base,'source':'web-pdf-wikipedia-replay','replay_recipe':recipe,
          'sampling':{'languages':{'uk':.5,'en':.5},'wikipedia':wikipedia_fraction,
                      'web_pdf':1-wikipedia_fraction,
                      'effective_web_approx':(1-wikipedia_fraction)*.8,
                      'effective_pdf_approx':(1-wikipedia_fraction)*.2},
          'wikipedia_tokens':{lang:(output/f'train_wikipedia_{lang}.bin').stat().st_size//4 for lang in ('uk','en')},
          'wikipedia_holdout_overlap_documents_removed':removed}
    copied=files+['wikipedia_metadata.json']+[f'train_wikipedia_{lang}.bin' for lang in ('uk','en')]
    meta['file_sha256']={name:file_hash(output/name) for name in copied}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta


def install_replay_sampling(trainer):
    """One 25/75 source draw per sequence, same checkpointed RNG; val unchanged."""
    if getattr(trainer,'_replay_sampling_installed',False):return
    batches=trainer.batches
    meta=json.loads((trainer.data_dir/'metadata.json').read_text())
    fraction=meta['replay_recipe']['wikipedia_fraction']
    wiki={lang:np.memmap(trainer.data_dir/f'train_wikipedia_{lang}.bin',mode='r',dtype=np.int32)
          for lang in ('uk','en')}
    if any(len(a)<=batches.length for a in wiki.values()):raise ValueError('Wikipedia train pool too short.')
    original=batches.get
    def get(split,device,language=None,generator=None):
        if split!='train':return original(split,device,language,generator)
        torch=batches.torch; rng=generator if generator is not None else batches.rng
        xs,ys=[],[]
        for _ in range(batches.batch):
            lang=language or ('uk','en')[int(torch.randint(2,(1,),generator=rng))]
            use_wiki=bool(torch.rand((),generator=rng)<fraction)
            array=wiki[lang] if use_wiki else batches.arrays[('train',lang)]
            start=int(torch.randint(len(array)-batches.length,(1,),generator=rng))
            row=torch.from_numpy(np.array(array[start:start+batches.length+1],dtype=np.int64))
            xs.append(row[:-1]);ys.append(row[1:])
        return torch.stack(xs).to(device),torch.stack(ys).to(device)
    batches.get=get
    trainer._replay_sampling_installed=True
