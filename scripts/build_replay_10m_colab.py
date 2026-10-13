"""Standalone +10M continuation of the completed ~55M replay checkpoint."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/next_moe_50m_wikipedia_replay.ipynb').read_text())
c=nb['cells']
for cell in c:
    if isinstance(cell['source'],list):cell['source']=''.join(cell['source'])
    cell['metadata']={}
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
c[0]['source']='''# next-moe: checkpoint ≈55M → ще10M, та сама replay суміш

Завантажуємо завершений checkpoint≈55M і продовжуємо до приблизно65M загалом.
**25% Wikipedia /75% web–PDF,50/50 UK/EN**, tokenizer та обидві validation незмінні.
Наявний filtered data-replay/ копіюється без повторного завантаження корпусів
і без повторної фільтрації. Default LR5e-5,100-update warm ramp, phase-token cosine.

Новий run окремий: попередні55M ваги не перезаписуються. Drive weights дають warm
restart із новим Adam; у межах runtime local last.pt дає exact resume.
Навчання за замовчуванням **до кінця додаткових10M**, без ліміту1000 updates.
Окрема ячейка графіків: train/old/new validation CE, routing, tokens/s, deltaCE
проти baseline55M. Для перегляду посеред етапу SESSION_UPDATES=500.
'''
c[2]['source']='''## Параметри та джерело checkpoint

BASE_RUN_NAME='' автоматично вибирає збережений frozen checkpoint≈55M з
data-replay/, інакше next-moe-50m-plus5m-replay-v1. Можна вказати каталог вручну.
BASE_DATA_NAME='data-replay'. Потрібна саме готова replay суміш із минулого етапу.
Не потрібно знову шукати початкові Wikipedia data-v2/ або web data-stage3/.
ТарифL4=1.54 — ваше минуле показання; оновіть за актуальною панеллю Colab.
'''
s=c[3]['source']
s=s.replace("BASE_RUN_NAME = 'next-moe-30m-plus20m-webpdf-v1'", "BASE_RUN_NAME = ''")
s=s.replace("BASE_DATA_NAME = 'data-stage3'", "BASE_DATA_NAME = 'data-replay'")
s=s.replace("RUN_NAME = 'next-moe-50m-plus5m-replay-v1'", "RUN_NAME = 'next-moe-55m-plus10m-replay-v1'")
s=s.replace('ADDITIONAL_TOKENS = 5_000_000','ADDITIONAL_TOKENS = 10_000_000')
s='\n'.join(line for line in s.splitlines() if not line.startswith('WIKI_DATA_PATH ='))+'\n'
s=s[:s.index('if WIKI_DATA_PATH:')]
s=s.replace('BASE_RUN = STORAGE / BASE_RUN_NAME', '''if not BASE_RUN_NAME:
    frozen=[]
    for folder in STORAGE.glob('checkpoint-*-tokens-frozen'):
        try: tokens=int(folder.name.split('-')[1])
        except (ValueError,IndexError): continue
        if 53_000_000 <= tokens <= 57_000_000 and (folder/'weights.pt').exists() and (folder/'data-replay/metadata.json').exists():
            frozen.append((tokens,folder.name))
    BASE_RUN_NAME=max(frozen)[1] if frozen else 'next-moe-50m-plus5m-replay-v1'
BASE_RUN = STORAGE / BASE_RUN_NAME''')
s=s.replace("LEARNING_RATE = 5e-5          # new phase, conservative LR compared with original 2e-4", "LEARNING_RATE = 5e-5          # same peak LR as previous replay phase")
s=s.replace('# None = train to +20M goal; 1000 = deliberately pause','# None = train to +10M goal; 500 = deliberately pause')
s+="print('Base checkpoint:', BASE_RUN)\n"
c[3]['source']=s
sources={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'next_moe').glob('*.py'))}
hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
c[5]['source']='SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\n'+'\n'.join(c[5]['source'].splitlines()[2:])+'\n'
c[6]['source']='''## Фіксація checkpoint55M та бюджету

Starting tokens беруться з ваг; target=start+10M. Phase metadata і baseSHA
зберігаються на Drive. Повторне відкриття не додає ще10M і не змінює target.
'''
c[7]['source']=c[7]['source'].replace('47_000_000 <=','53_000_000 <=').replace('<= 53_000_000','<= 57_000_000').replace('saved 50M run','saved 55M run').replace('checkpoint near 50M','checkpoint near 55M')
c[8]['source']='''## Незмінний кешований replay корпус

Перевіряємо fingerprints усіх файлів. Token IDs, Wikipedia pool і обидві
валідації залишаються незмінними. Новий sampler має ту саму частку25%.
Файли копіюються в новий run і local cache; попередній run не змінюється.
'''
c[9]['source']='''
def verify_replay_cache(folder):
    metadata=json.loads((folder/'metadata.json').read_text())
    if metadata.get('source')!='web-pdf-wikipedia-replay':
        raise ValueError('Expected data-replay from the completed55M phase.')
    if metadata['replay_recipe']['wikipedia_fraction']!=WIKIPEDIA_FRACTION:
        raise ValueError('Keep the same replay fraction as checkpoint55M.')
    for name,expected in metadata['file_sha256'].items():
        if file_hash(folder/name)!=expected:raise ValueError(f'Corpus file changed: {name}')
    return metadata

metadata=verify_replay_cache(OLD_DATA)
if (DATA_DIR/'metadata.json').exists():
    cached=verify_replay_cache(DATA_DIR)
    if file_hash(DATA_DIR/'metadata.json')!=file_hash(OLD_DATA/'metadata.json'):
        raise ValueError('Base corpus metadata differs from cached run; use new RUN_NAME.')
else:
    size=sum(p.stat().st_size for p in OLD_DATA.rglob('*') if p.is_file())
    used=sum(p.stat().st_size for p in RUN_DIR.rglob('*') if p.is_file())
    if (size+used+1.2*2**30)/2**30>PERSISTENT_BUDGET_GIB:
        raise OSError('Replay corpus and temporary weights exceed run budget.')
    pending=RUN_DIR/'data-replay.tmp'
    if pending.exists():raise FileExistsError('Incomplete data-replay.tmp; rename/remove before retry.')
    shutil.copytree(OLD_DATA,pending);os.replace(pending,DATA_DIR)
shutil.copytree(DATA_DIR,CACHE_DIR,dirs_exist_ok=True)
tokenizer=Tokenizer.from_file(str(CACHE_DIR/'tokenizer.json'))
print(json.dumps({k:metadata[k] for k in ('tokens','wikipedia_tokens','sampling')},indent=2))
print('Original replay corpus and both holdouts preserved; no corpus download.')
'''
c[5]['source']+='from next_moe.replay import file_hash\n'
c[10]['source']=c[10]['source'].replace('weights50M','weights55M').replace('нового+5M','нового+10M')
c[14]['source']=c[14]['source'].replace('+5M','+10M')
c[16]['source']=c[16]['source'].replace('checkpoint50M','checkpoint55M')
c[17]['source']=c[17]['source'].replace('checkpoint50M','checkpoint55M')
c[20]['source']='''## Збереження

Новий RUN_DIR: weights.pt, data-replay/, tokenizer.json, phase.json, baseline-validation.json,
configs, source_hashes.json, metrics.jsonl, pip-freeze.txt. Previous55M checkpoint
не перезаписується. Local full last.pt≈3.83GiB; Drive compact weights≈0.55GiB.
Перед запуском перевірте місце: новий corpus≈210MB, snapshots потребують тимчасово
ще≈0.55GiB. Run budget не дорівнює загальній квоті Drive.
Інструкція: https://github.com/FactorialD/next-moe/blob/work/docs/replay-10m-colab.md
'''
for i,cell in enumerate(c):cell['id']=f'replay-10m-{i:02d}'
nb['metadata']={'colab':{'name':'next_moe_55m_plus10m_replay.ipynb'},
                'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
p=ROOT/'notebooks/next_moe_55m_plus10m_replay.ipynb'
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Built',p.name,len(c),'cells')
