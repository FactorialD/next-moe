"""Bounded UK/EN document ingestion, leak-free split, tokenizer and token files."""
from pathlib import Path
import hashlib
import json
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

LANGUAGES = ('uk', 'en')
SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']
WIKIPEDIA_REVISION = 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'


def document_sources(source='wikipedia', local_dir=None):
    if source == 'wikipedia':
        from datasets import load_dataset
        # Public parquet dataset. No trust_remote_code; no full-dataset download.
        return {lang: iter(load_dataset(
            'wikimedia/wikipedia', f'20231101.{lang}', revision=WIKIPEDIA_REVISION,
            split='train', streaming=True))
                for lang in LANGUAGES}
    if source != 'local' or local_dir is None:
        raise ValueError('Use wikipedia or local with a directory containing uk/ and en/.')
    result = {}
    for lang in LANGUAGES:
        paths = sorted((Path(local_dir) / lang).glob('*.txt'))
        if len(paths) < 2:
            raise ValueError(f'Need at least two separate UTF-8 documents in {lang}/.')
        result[lang] = ({'text': path.read_text(encoding='utf-8'), 'id': path.name,
                         'url': None, 'title': path.name} for path in paths)
    return result


def prepare_data(output, source='wikipedia', local_dir=None, documents_per_language=1000,
                 validation_documents=32, vocab_size=65536, max_chars_per_document=20000,
                 seed=1234):
    if documents_per_language <= validation_documents or validation_documents < 1:
        raise ValueError('Training and validation must both contain documents.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = document_sources(source, local_dir)
    documents, seen, hashes, provenance, scanned = {}, set(), {}, {}, {}
    for lang in LANGUAGES:
        docs, ids, origins = [], [], []
        scanned[lang] = 0
        for record in raw[lang]:
            scanned[lang] += 1
            text = record['text'].strip()
            if len(text) < 200:
                continue
            full_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            text = text[:max_chars_per_document]
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            docs.append(text)
            ids.append(digest)
            origins.append({'id': record.get('id'), 'url': record.get('url'),
                            'title': record.get('title'), 'sha256_used_text': digest,
                            'sha256_full_stripped_text': full_digest, 'used_characters': len(text)})
            if len(docs) >= documents_per_language:
                break
        if len(docs) < validation_documents + 1:
            raise ValueError(f'Too few usable {lang} documents ({len(docs)}).')
        # Shuffle locally with fixed seed; then split by complete documents.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(docs))
        documents[lang] = {'val': [docs[i] for i in order[:validation_documents]],
                           'train': [docs[i] for i in order[validation_documents:]]}
        hashes[lang] = {'val': [ids[i] for i in order[:validation_documents]],
                        'train': [ids[i] for i in order[validation_documents:]]}
        provenance[lang] = {'val': [origins[i] for i in order[:validation_documents]],
                            'train': [origins[i] for i in order[validation_documents:]]}
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    # ByteLevel preserves Ukrainian letters, punctuation and case; no strip accents.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # Equal document budgets for both languages, validation excluded.
    tokenizer.train_from_iterator((doc for lang in LANGUAGES for doc in documents[lang]['train']),
                                  trainer=trainer)
    tokenizer.save(str(output / 'tokenizer.json'))
    metadata = {'source': source, 'source_config': '20231101.{uk,en}' if source == 'wikipedia' else None,
                'source_dataset': 'wikimedia/wikipedia' if source == 'wikipedia' else None,
                'source_revision': WIKIPEDIA_REVISION if source == 'wikipedia' else None,
                'requested_documents_per_language': documents_per_language,
                'validation_documents_per_language': validation_documents,
                'max_characters_per_document': max_chars_per_document,
                'min_characters_per_document': 200, 'scanned_documents': scanned,
                'document_counts': {lang: {split: len(documents[lang][split])
                                          for split in ('train', 'val')} for lang in LANGUAGES},
                'seed': seed, 'document_hashes': hashes, 'requested_vocab': vocab_size,
                'actual_vocab': tokenizer.get_vocab_size(), 'normalization': 'none; byte-level BPE',
                'special_ids': {s: tokenizer.token_to_id(s) for s in SPECIAL}, 'tokens': {}}
    for lang in LANGUAGES:
        for split in ('train', 'val'):
            ids = []
            for doc in documents[lang][split]:
                ids.extend(tokenizer.encode(doc).ids)
                ids.append(tokenizer.token_to_id('<eos>'))
            np.asarray(ids, dtype=np.int32).tofile(output / f'{split}_{lang}.bin')
            metadata['tokens'][f'{split}_{lang}'] = len(ids)
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (output / 'documents.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(json.dumps({key: metadata[key] for key in ('actual_vocab', 'tokens', 'special_ids')}, indent=2))
    return metadata


class TokenBatches:
    """50/50 UK/EN sampling, no padding; CPU RNG state is checkpointed."""
    def __init__(self, folder, sequence_length=128, microbatch=1, seed=1234):
        import torch
        self.torch = torch
        self.length, self.batch = sequence_length, microbatch
        self.rng = torch.Generator().manual_seed(seed)
        self.arrays = {(split, lang): np.memmap(Path(folder) / f'{split}_{lang}.bin',
                                              mode='r', dtype=np.int32)
                       for split in ('train', 'val') for lang in LANGUAGES}
        for key, array in self.arrays.items():
            if len(array) <= sequence_length:
                raise ValueError(f'{key} needs more than {sequence_length} tokens.')

    def get(self, split, device, language=None, generator=None):
        torch = self.torch
        gen = self.rng if generator is None else generator
        xs, ys = [], []
        for _ in range(self.batch):
            lang = language or LANGUAGES[int(torch.randint(2, (1,), generator=gen))]
            array = self.arrays[(split, lang)]
            start = int(torch.randint(len(array) - self.length, (1,), generator=gen))
            row = torch.from_numpy(np.array(array[start:start + self.length + 1], dtype=np.int64))
            xs.append(row[:-1]); ys.append(row[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)
