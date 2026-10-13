"""Bounded, pinned UK/EN continuation corpus with unchanged BPE and holdouts."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import numpy as np
from tokenizers import Tokenizer

SOURCES = {
    'web_uk': {'dataset': 'HuggingFaceFW/fineweb-2', 'config': 'ukr_Cyrl',
               'revision': 'af9c13333eb981300149d5ca60a8e9d659b276b9', 'license': 'ODC-BY-1.0'},
    'web_en': {'dataset': 'HuggingFaceFW/fineweb-edu', 'config': 'CC-MAIN-2025-26',
               'revision': '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9', 'license': 'ODC-BY-1.0'},
    'pdf_uk': {'dataset': 'HuggingFaceFW/finepdfs-edu', 'config': 'ukr_Cyrl',
               'revision': '9cfabe2127faca99b3d5c4dc6d1fcb397399ebde', 'license': 'ODC-BY-1.0'},
    'pdf_en': {'dataset': 'HuggingFaceFW/finepdfs-edu', 'config': 'eng_Latn',
               'revision': '9cfabe2127faca99b3d5c4dc6d1fcb397399ebde', 'license': 'ODC-BY-1.0'},
}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class NearDuplicates:
    """Approximate SimHash on bounded word shingles; not a proof of no leakage."""
    def __init__(self):
        self.bands = defaultdict(list)

    @staticmethod
    def fingerprint(text):
        words = re.findall(r'\w+', text.casefold())
        shingles = {' '.join(words[i:i+5]) for i in range(max(len(words)-4, 1))}
        # Deterministic bounded sample; long texts cannot exhaust host RAM/CPU here.
        hashes = sorted(hashlib.blake2b(s.encode(), digest_size=8).digest() for s in shingles)[:128]
        bits = np.unpackbits(np.frombuffer(b''.join(hashes), dtype=np.uint8)).reshape(-1, 64)
        packed = np.packbits(bits.sum(axis=0)*2 >= len(hashes)).tobytes()
        return int.from_bytes(packed, 'big')

    def duplicate(self, text, add=False):
        value = self.fingerprint(text)
        keys = [(i, (value >> (i*16)) & 65535) for i in range(4)]
        candidates = {v for key in keys for v in self.bands[key]}
        match = any((value ^ other).bit_count() <= 3 for other in candidates)
        if add and not match:
            for key in keys:
                self.bands[key].append(value)
        return match


def stream_source(spec, seed, shuffle_buffer):
    from datasets import load_dataset
    print('Opening', spec['dataset'], spec['config'], spec['revision'], flush=True)
    dataset = load_dataset(spec['dataset'], name=spec['config'], revision=spec['revision'],
                           split='train', streaming=True)
    return iter(dataset.shuffle(seed=seed, buffer_size=shuffle_buffer))


def prepare_continuation_data(output, old_data, tokens_per_language=20_000_000,
                              validation_documents=128, pdf_fraction=.2, seed=1234,
                              shuffle_buffer=256, max_chars=20000, max_scanned=500000,
                              record_sources=None, progress_every=500):
    """80/20 web/PDF token pool, 50/50 languages in Trainer; no tokenizer training.

    Old val bins are byte-identical; new domain holdouts are separate. All source
    readers fail explicitly, never fall back to toy data or old Wikipedia train.
    record_sources is for offline fixtures only, keyed by web_uk/pdf_uk/etc.
    """
    if tokens_per_language < 1 or validation_documents < 2 or not 0 < pdf_fraction < 1:
        raise ValueError('Positive corpus limits and pdf_fraction strictly between 0 and 1 required.')
    output, old_data = Path(output), Path(old_data)
    recipe = {'tokens_per_language': tokens_per_language, 'validation_documents': validation_documents,
              'pdf_fraction': pdf_fraction, 'seed': seed, 'shuffle_buffer': shuffle_buffer,
              'max_chars': max_chars, 'max_scanned': max_scanned, 'sources': SOURCES,
              'old_tokenizer_sha256': hashlib.sha256((old_data/'tokenizer.json').read_bytes()).hexdigest(),
              'old_metadata_sha256': hashlib.sha256((old_data/'metadata.json').read_bytes()).hexdigest(),
              'old_validation_sha256': {lang: hashlib.sha256((old_data/f'val_{lang}.bin').read_bytes()).hexdigest()
                                        for lang in ('uk', 'en')}}
    if (output/'metadata.json').exists():
        meta = json.loads((output/'metadata.json').read_text())
        if meta.get('recipe') != recipe:
            raise ValueError('Existing corpus recipe differs. Use a NEW output directory.')
        for name, expected in meta['file_sha256'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Cached corpus file changed: {name}')
        print('Verified cached corpus:', output, flush=True)
        return meta
    if output.exists() and any(output.iterdir()):
        raise ValueError('Incomplete corpus directory exists. Use a new directory or remove the incomplete copy.')
    output.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(old_data/'tokenizer.json'))
    eos = tok.token_to_id('<eos>')
    old = json.loads((old_data/'metadata.json').read_text())
    blocked, near = set(), NearDuplicates()
    for lang in ('uk', 'en'):
        shutil.copy2(old_data/f'val_{lang}.bin', output/f'val_{lang}.bin')
        origins_path = old_data/'documents.json'
        if origins_path.exists():
            origins = json.loads(origins_path.read_text())[lang]['val']
            for origin in origins:
                blocked.update(origin[k] for k in ('sha256_used_text', 'sha256_full_stripped_text') if origin.get(k))
        blocked.update(old.get('document_hashes', {}).get(lang, {}).get('val', []))
        ids = np.fromfile(old_data/f'val_{lang}.bin', dtype=np.int32)
        start = 0
        for stop in np.flatnonzero(ids == eos):
            text = tok.decode(ids[start:stop].tolist(), skip_special_tokens=False).strip()
            if text:
                blocked.add(digest(text)); near.duplicate(text, add=True)
            start = stop+1
    shutil.copy2(old_data/'tokenizer.json', output/'tokenizer.json')
    counts, scanned, rejected, doc_counts, source_counts = {}, {}, {}, {}, {}
    hashes = {lang: {'val': old['document_hashes'][lang]['val'], 'train': [], 'val_new': []}
              for lang in ('uk', 'en')}
    with (output/'documents.jsonl').open('w') as manifest:
        for lang in ('uk', 'en'):
            specs = {kind: SOURCES[f'{kind}_{lang}'] for kind in ('web', 'pdf')}
            readers = {kind: (iter(record_sources[f'{kind}_{lang}']) if record_sources is not None
                              else stream_source(spec, seed, shuffle_buffer)) for kind, spec in specs.items()}
            train_counts, val_counts, scanned_lang = Counter(), Counter(), 0
            target = {'web': round(tokens_per_language*(1-pdf_fraction)),
                      'pdf': tokens_per_language-round(tokens_per_language*(1-pdf_fraction))}
            reject = Counter()
            with (output/f'train_{lang}.bin').open('wb') as train, (output/f'val_new_{lang}.bin').open('wb') as val:
                while any(train_counts[k] < target[k] for k in target):
                    kind = min((k for k in target if train_counts[k] < target[k]),
                               key=lambda k: train_counts[k]/max(target[k], 1))
                    try:
                        record = next(readers[kind])
                    except StopIteration as error:
                        raise ValueError(f'{lang}/{kind} exhausted before token goal; do not silently change mixture.') from error
                    scanned_lang += 1
                    if scanned_lang % progress_every == 0:
                        print(lang, 'scanned', scanned_lang, 'train tokens', sum(train_counts.values()),
                              '/', tokens_per_language, 'rejected', dict(reject), flush=True)
                    if scanned_lang > max_scanned:
                        raise ValueError(f'{lang} exceeded max_scanned before token goal.')
                    full = record.get('text', '').strip()
                    text = full[:max_chars]
                    page_languages = record.get('per_page_languages') or []
                    known = [value for value in page_languages if value not in ('unknown', 'und', None)]
                    expected_language = 'ukr_Cyrl' if lang == 'uk' else 'eng_Latn'
                    if known and sum(value == expected_language for value in known)/len(known) < .8:
                        reject['mixed_pdf_languages'] += 1; continue
                    if len(text) < 200 or sum(c.isalpha() for c in text)/max(len(text), 1) < .35:
                        reject['short_or_nonprose'] += 1; continue
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    if len(lines) > 10 and len(set(lines))/len(lines) < .5:
                        reject['repeated_lines'] += 1; continue
                    key, full_key = digest(text), digest(full)
                    if key in blocked or full_key in blocked or near.duplicate(text):
                        reject['duplicate_or_holdout_match'] += 1; continue
                    blocked.update((key, full_key)); near.duplicate(text, add=True)
                    ids = np.asarray(tok.encode(text).ids + [eos], dtype=np.int32)
                    # Holdouts per source ensure both web and PDF coverage.
                    pdf_val = min(validation_documents-1, max(1, round(validation_documents*pdf_fraction)))
                    val_goal = pdf_val if kind == 'pdf' else validation_documents-pdf_val
                    split = 'val_new' if val_counts[kind] < val_goal else 'train'
                    ids.tofile(val if split == 'val_new' else train)
                    if split == 'val_new':
                        val_counts[kind] += 1
                    else:
                        train_counts[kind] += len(ids)
                    hashes[lang][split].append(key)
                    manifest.write(json.dumps({'language': lang, 'split': split, 'source': kind, **specs[kind],
                        'id': record.get('id'), 'url': record.get('url'),
                        'crawl_date': record.get('date') or record.get('crawl_date'),
                        'sha256_used_text': key, 'sha256_full_stripped_text': full_key,
                        'tokens': len(ids), 'used_characters': len(text)}, ensure_ascii=False)+'\n')
            counts[f'train_{lang}'] = sum(train_counts.values())
            counts[f'val_{lang}'] = (output/f'val_{lang}.bin').stat().st_size//4
            counts[f'val_new_{lang}'] = (output/f'val_new_{lang}.bin').stat().st_size//4
            doc_counts[lang] = {'train': len(hashes[lang]['train']), 'val_new': sum(val_counts.values()),
                                'val': len(hashes[lang]['val'])}
            source_counts[lang], scanned[lang], rejected[lang] = dict(train_counts), scanned_lang, dict(reject)
            print(lang, 'COMPLETE:', counts[f'train_{lang}'], 'train tokens', flush=True)
    files = ['tokenizer.json', 'documents.jsonl'] + [f'{split}_{lang}.bin'
                 for lang in ('uk', 'en') for split in ('train', 'val', 'val_new')]
    meta = {'source': 'fineweb-finepdfs-continuation', 'source_revision': SOURCES, 'recipe': recipe,
            'actual_vocab': tok.get_vocab_size(), 'special_ids': {s: tok.token_to_id(s) for s in ('<pad>', '<eos>')},
            'tokens': counts, 'document_hashes': hashes, 'document_counts': doc_counts,
            'source_training_tokens': source_counts, 'scanned_documents': scanned, 'rejected': rejected,
            'tokenizer_reused': True, 'validation_preserved': True,
            'filters': 'min200 chars; alpha>=35%; known PDF page languages>=80% target; repeated-line filter; global exact hashes and approximate SimHash<=3',
            'max_chars_per_document': max_chars,
            'file_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files}}
    (output/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def install_domain_evaluation(trainer):
    """Report old Wikipedia and new-domain holdouts without modifying core trainer."""
    if getattr(trainer, '_domain_evaluation_installed', False):
        return
    original = trainer.evaluate
    newer = {('val', lang): np.memmap(trainer.data_dir/f'val_new_{lang}.bin', mode='r', dtype=np.int32)
             for lang in ('uk', 'en')}
    if any(len(a) <= trainer.cfg.sequence_length for a in newer.values()):
        raise ValueError('New holdouts are too short for evaluation.')
    def evaluate():
        result = original()
        previous = {key: trainer.batches.arrays[key] for key in newer}
        trainer.batches.arrays.update(newer)
        try:
            result.update({key.replace('val_', 'val_new_', 1): value for key, value in original().items()})
        finally:
            trainer.batches.arrays.update(previous)
        return result
    trainer.evaluate = evaluate
    trainer._domain_evaluation_installed = True


def install_phase_schedule(trainer, initial_tokens, target_tokens):
    """Cosine tracks THIS phase's token budget, independent of historical steps."""
    if target_tokens <= initial_tokens:
        raise ValueError('Phase target must exceed starting tokens.')
    def learning_rate():
        progress = min(1., max(0., (trainer.tokens_seen-initial_tokens)/(target_tokens-initial_tokens)))
        rate = trainer.cfg.learning_rate * (.1 + .9*.5*(1+math.cos(math.pi*progress)))
        if trainer.restart_step is not None and trainer.cfg.restart_warmup > 0:
            rate *= min(1., (trainer.step-trainer.restart_step+1)/trainer.cfg.restart_warmup)
        return rate
    trainer.learning_rate = learning_rate
