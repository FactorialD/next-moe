"""Standalone checkpoint 50M -> +5M, cached web/PDF plus 25% Wikipedia replay."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/next_moe_30m_to_50m_webpdf.ipynb').read_text())
c=nb['cells']
c[0]['source']='''# next-moe: checkpoint 50M → короткий Wikipedia replay етап

Продовжуємо з weights.pt 50M. За замовчуванням **ще 5M tokens**, приблизно до55M,
щоб перевірити відновлення Wikipedia UK без значної втрати нового домену.
Суміш за sampled sequences/tokens: **25% Wikipedia +75% cached web/PDF**,
тобто приблизно25% Wikipedia,60% web,15% PDF. UK/EN залишаються50/50.
Частки стохастичні, не жорсткі для кожного update. BPE незмінний.

Нові датасети з мережі **не завантажуються**: використовуємо вже збережені
FineWeb/FinePDFs data-stage3/ та початкові Wikipedia train token files.
Стара і нова validation byte-identical. Baseline evaluation перед навчанням.
Default LR5e-5, warm ramp100 і token-based cosine нового етапу.

Runtime → Change runtime type → L4. До зміни GPU впевніться, що weights.pt50M
збережено на Drive. Drive зберігає компактні weights; local last.pt містить Adam.
Окрема ячейка графіків після training cell; для перегляду посеред етапу поставте
SESSION_UPDATES=500, повторюйте training cell, потім графіки.
'''
c[2]['source']='''## Параметри та шляхи

BASE_RUN_NAME — каталог завершеного50M run. Не продовжуйте його старим notebook
під час replay: базовий SHA має залишатися незмінним. Новий run окремий.
WIKI_DATA_PATH='' знаходить data-v2 спочатку у frozen30M, потім у старому run.
Якщо вони розташовані інакше, вкажіть абсолютний шлях до Wikipedia data-v2.
COMPUTE_UNITS_PER_HOUR=1.54 — ваш останній тарифL4; оновіть, якщо він змінився.
'''
s=c[3]['source']
s=s.replace("BASE_RUN_NAME = 'checkpoint-30m-frozen'", "BASE_RUN_NAME = 'next-moe-30m-plus20m-webpdf-v1'")
s=s.replace("BASE_DATA_NAME = 'data-v2'", "BASE_DATA_NAME = 'data-stage3'\nWIKI_DATA_PATH = ''\nWIKIPEDIA_FRACTION = .25")
s=s.replace("\nRUN_NAME = 'next-moe-30m-plus20m-webpdf-v1'", "\nRUN_NAME = 'next-moe-50m-plus5m-replay-v1'")
s=s.replace('ADDITIONAL_TOKENS = 20_000_000','ADDITIONAL_TOKENS = 5_000_000')
s=s.replace('LEARNING_RATE = 1e-4','LEARNING_RATE = 5e-5')
s=s.replace('RUN_EFFICIENCY_BENCHMARK = True','RUN_EFFICIENCY_BENCHMARK = False')
s=s.replace('COMPUTE_UNITS_PER_HOUR = None','COMPUTE_UNITS_PER_HOUR = 1.54')
s=s.replace("DATA_DIR = RUN_DIR / 'data-stage3'", "DATA_DIR = RUN_DIR / 'data-replay'")
# Corpus limits belong to previous data preparation and are not controls in replay.
s='\n'.join(line for line in s.splitlines() if not line.startswith(('TOKENS_PER_LANGUAGE =','NEW_VALIDATION_DOCUMENTS =','PDF_FRACTION =','SHUFFLE_BUFFER =')))+'\n'
s+='''
if WIKI_DATA_PATH:
    WIKI_DATA = Path(WIKI_DATA_PATH)
else:
    candidates=[STORAGE/'checkpoint-30m-frozen/data-v2',
                STORAGE/'next-moe-0p2b-engram0p1b-10m-v2/data-v2']
    WIKI_DATA=next((p for p in candidates if all((p/name).exists() for name in
        ('tokenizer.json','metadata.json','train_uk.bin','train_en.bin','val_uk.bin','val_en.bin'))),None)
    if WIKI_DATA is None:
        raise FileNotFoundError('Set WIKI_DATA_PATH to the ORIGINAL Wikipedia data-v2 directory.')
print('Wikipedia replay source:',WIKI_DATA)
'''
c[3]['source']=s
sources={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'next_moe').glob('*.py'))}
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
c[5]['source']='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\n'+'\n'.join(c[5]['source'].splitlines()[2:])+'\nfrom next_moe.replay import prepare_replay_data, install_replay_sampling\n'
c[6]['source']='''## Фіксація checkpoint50M та бюджету нового етапу

Phase metadata фіксує фактичні starting tokens і target=start+5M, а також
WIKIPEDIA_FRACTION. Повторне відкриття не додає новий бюджет. Base SHA і
оригінальний tokenizer мають збігатися. Змінений recipe потребує нового RUN_NAME.
'''
c[7]['source']=c[7]['source'].replace('saved 30M run','saved 50M run').replace('27_000_000 <=','47_000_000 <=').replace('<= 32_000_000','<= 53_000_000').replace('checkpoint near 30M','checkpoint near 50M')
c[7]['source']=c[7]['source'].replace("if phase['additional_tokens'] != ADDITIONAL_TOKENS:","if phase['additional_tokens'] != ADDITIONAL_TOKENS or phase.get('wikipedia_fraction') != WIKIPEDIA_FRACTION:")
c[7]['source']=c[7]['source'].replace("'additional_tokens':ADDITIONAL_TOKENS,", "'additional_tokens':ADDITIONAL_TOKENS, 'wikipedia_fraction':WIKIPEDIA_FRACTION,")
c[8]['source']='''## Кешований корпус:25% Wikipedia replay,75% web/PDF

Копіюємо вже токенізовані training pools і зберігаємо їх fingerprints. Новий
sampler вибирає джерело для кожної sequence:25/75. Усі його draws використовують
той самий checkpointed RNG. Старі й нові holdout не додаються до навчання.
Не перевчати tokenizer. Старі corpus source/revision/counts залишаються в metadata.
'''
c[9]['source']='''
if (DATA_DIR/'metadata.json').exists():
    metadata=prepare_replay_data(DATA_DIR,OLD_DATA,WIKI_DATA,WIKIPEDIA_FRACTION)
    shutil.copytree(DATA_DIR,CACHE_DIR,dirs_exist_ok=True)
else:
    build_dir=LOCAL_DIR/'replay-build'
    metadata=prepare_replay_data(build_dir,OLD_DATA,WIKI_DATA,WIKIPEDIA_FRACTION)
    size=sum(p.stat().st_size for p in build_dir.rglob('*') if p.is_file())
    used=sum(p.stat().st_size for p in RUN_DIR.rglob('*') if p.is_file())
    if (size+used+1.2*2**30)/2**30>PERSISTENT_BUDGET_GIB:
        raise OSError('Replay data plus temporary weights snapshots exceed run budget.')
    pending=RUN_DIR/'data-replay.tmp'
    if pending.exists():raise FileExistsError('Incomplete data-replay.tmp exists; rename/remove before retry.')
    shutil.copytree(build_dir,pending);os.replace(pending,DATA_DIR)
    shutil.copytree(build_dir,CACHE_DIR,dirs_exist_ok=True)
tokenizer=Tokenizer.from_file(str(CACHE_DIR/'tokenizer.json'))
print(json.dumps({k:metadata[k] for k in ('tokens','wikipedia_tokens','sampling')},indent=2))
print('Cached sources reused; both validations unchanged; no remote corpus download.')
'''
c[10]['source']='''## Ваги та нова фаза

Base weights50M, новий Adam, LR5e-5 із ramp100. Local last.pt дає exact resume,
Drive weights.pt — warm restart. Token-based LR schedule лише для нового+5M.
Replay sampler встановлюється до baseline evaluation/benchmark/training що reload.
'''
c[11]['source']=c[11]['source'].replace('fix_evaluation_microbatch(trainer, 1)','install_replay_sampling(trainer)\nfix_evaluation_microbatch(trainer, 1)')
c[11]['source']+='''
baseline_path=RUN_DIR/'baseline-validation.json'
if not baseline_path.exists():
    baseline_path.write_text(json.dumps({'tokens_seen':trainer.tokens_seen,**trainer.evaluate()},indent=2))
'''
c[14]['source']=c[14]['source'].replace('+20M','+5M')
c[16]['source']+='''

Порівнюйте кінцеву evaluation з baseline цього run (checkpoint50M).
Мета: old UK CE знижується без суттєвого погіршення new UK/EN.
'''
c[17]['source']+='''
baseline=json.loads((RUN_DIR/'baseline-validation.json').read_text())
current=trainer.evaluate()
print('CE change vs checkpoint50M (negative = improvement):')
for key in ('val_ce_uk','val_ce_en','val_new_ce_uk','val_new_ce_en'):
    print(key,round(baseline[key],4),'->',round(current[key],4),'delta',round(current[key]-baseline[key],4))
'''
c[20]['source']='''## Збереження

Новий RUN_DIR: weights.pt, tokenizer.json, data-replay/, phase.json, configs,
source_hashes.json, metrics.jsonl, baseline-validation.json, pip-freeze.txt.
Local last.pt≈3.83GiB із optimizer; Drive weights≈0.55GiB. Початкові30M/50M run
не перезаписуються. Replay token pools≈210MB, точний обсяг залежить від actual counts.
Загальна квота Drive включає всі попередні run; залиште місце для temporary snapshot.

Інструкція: https://github.com/FactorialD/next-moe/blob/work/docs/replay-colab.md
'''
for i,cell in enumerate(c):cell['id']=f'replay-{i:02d}'
nb['metadata']['colab']['name']='next_moe_50m_wikipedia_replay.ipynb'
path=ROOT/'notebooks/next_moe_50m_wikipedia_replay.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Built',path.name,len(c),'cells')
