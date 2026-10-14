"""Build a standalone persistent-volume RunPod migration notebook."""
from pathlib import Path
import hashlib,json,textwrap
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s):cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':textwrap.dedent(s).strip()+'\n'})
md('''
# next-moe на RunPod: перенесення і продовження replay training

Один GPU Pod, PyTorch>=2.6 із CUDA, Jupyter. Каталог **/workspace має бути вашим
постійним volume**, а не ephemeral container disk. Перевірте mount перед запуском.
ZIP із Colab розпакуйте в /workspace/next-moe-transfer (без додаткової вкладеної папки).
Немає drive.mount, Colab APIs або повторного download корпусів.

Default: продовжити залишок phase.json. Якщо бюджет завершено, автоматично нові
tokens не додаються. Для нового етапу задайте ADDITIONAL_TOKENS, наприклад10_000_000,
і новий RUN_NAME. Sampler25%Wikipedia/75%web–PDF, мовний баланс50/50 збережений.
Compact weights — warm restart; last.pt за наявності — exact migration зі станом Adam.
На іншому GPU parameter precision має підтримуватися; інакше explicit warm mode.

**Training context2048, microbatch1/accumulation1: ті самі2048 tokens/update.**
Exact load спочатку відновлює старий recipe512 і Adam, після чого змінюємо context.
Ціль із phase.json і LR не перезапускаються. Validation512 зберігає старі ключі;
окремі val_ctx2048_* показують довший контекст. Це зміна recipe, не ідентичний
старому запуск. Stop тепер перериває також зовнішні цикли після checkpoint saving.
''')
code('''
import sys,subprocess,json,math,time,gc,shutil,hashlib,os
from pathlib import Path
import torch
if not torch.cuda.is_available():raise RuntimeError('GPU/CUDA unavailable. Use a PyTorch GPU Pod.')
try: native_bf16=torch.cuda.is_bf16_supported(including_emulation=False)
except TypeError:raise RuntimeError('Use PyTorch>=2.6 with CUDA; do not install a CPU Torch wheel.')
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('GPU:',torch.cuda.get_device_name(0),'VRAM GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
print('Native BF16:',native_bf16)
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'datasets==3.6.0','tokenizers==0.21.1','matplotlib>=3.8,<4'])
''')
md('''
## Шляхи, бюджет, ціна GPU

INPUT_DIR — каталог ZIP/перенесених файлів: weights.pt + data-replay/ + phase.json
якщо є незавершений бюджет. last.pt необов'язковий. RUN_NAME новий, input не
перезаписується. PRICE_PER_HOUR_USD — фактичний тариф вашого Pod, без storage fees.
LOCAL full optimizer checkpoints також зберігаються на volume: потрібне місце
для last.pt≈3.83GiB і його тимчасової копії. Volume20GB — практична стартова ємність.
''')
code('''
INPUT_PATH = '/workspace/next-moe-transfer'
STORAGE_PATH = '/workspace/next-moe-runs'
RUN_NAME = 'runpod-replay-v1'
ADDITIONAL_TOKENS = 0        # 0 preserves imported phase target; positive = NEW phase
RESUME_MODE = 'auto'         # auto / warm; different precision requires warm
SEQ_LEN = 2048
VALIDATION_SEQ_LEN = 512
MICROBATCH = 1
GRAD_ACCUM = 1
LEARNING_RATE = 5e-5
RESTART_WARMUP = 100
SESSION_UPDATES = None
RUN_HOURS = None            # optional time limit: remaining rental hours minus a margin
RUN_BENCHMARK = True
PRICE_PER_HOUR_USD = .24     # update to YOUR actual Pod GPU price
SEED = 1234
INPUT_DIR = Path(INPUT_PATH)
RUN_DIR = Path(STORAGE_PATH)/RUN_NAME
DATA_DIR = RUN_DIR/'data-replay'
CODE_DIR = Path(STORAGE_PATH)/'embedded-code'
if RUN_DIR.resolve()==INPUT_DIR.resolve():raise ValueError('Use a separate output run.')
for p in (RUN_DIR,CODE_DIR):p.mkdir(parents=True,exist_ok=True)
print('Storage free GiB:',shutil.disk_usage(RUN_DIR).free/2**30)
required_free_gib = 5 if (RUN_DIR/'last.pt').exists() else 9
if shutil.disk_usage(RUN_DIR).free<required_free_gib*2**30:
    raise OSError(f'Keep at least{required_free_gib}GiB free for full checkpoint + temporary copy and weights/data.')
''')
md('## Вбудований код')
sources={str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'next_moe').glob('*.py'))}
hashes={name:hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}
code('SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\n'+'''
for name,value in SOURCES.items():
    target=CODE_DIR/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(value)
if str(CODE_DIR) not in sys.path:sys.path.insert(0,str(CODE_DIR))
from next_moe.model import ModelConfig,NextMoE
from next_moe.train import Trainer,TrainConfig
from next_moe.corpus import install_domain_evaluation,install_phase_schedule
from next_moe.replay import install_replay_sampling,file_hash
from tokenizers import Tokenizer
'''+ 'exec('+repr((ROOT/'scripts/colab_efficiency.py').read_text())+')\n'
+ 'exec('+repr((ROOT/'scripts/context_training.py').read_text())+')\n'
+ 'exec('+repr((ROOT/'scripts/training_control.py').read_text())+')\n')
md('''
## Перевірка transfer і копіювання незмінного корпусу

Transfer manifest SHA перевіряє ZIP contents; для вручну переданого каталогу
без transfer manifest перевіряються fingerprints data-replay і checkpoint BPE.
Checkpoint.pt untrusted files не завантажувати: використовуйте свої weights.
Corpus не змінюється. Partial copy directory не приймається за готовий.
''')
code('''
manifest_path=INPUT_DIR/'transfer-manifest.json'
if manifest_path.exists():
    transfer=json.loads(manifest_path.read_text())
    for name,expected in transfer['file_sha256'].items():
        if file_hash(INPUT_DIR/name)!=expected:raise ValueError(f'Transfer changed: {name}')
for p in (INPUT_DIR/'weights.pt',INPUT_DIR/'data-replay/metadata.json'):
    if not p.exists():raise FileNotFoundError(f'{p}. Extract transfer ZIP into INPUT_DIR.')
input_data=INPUT_DIR/'data-replay'
meta=json.loads((input_data/'metadata.json').read_text())
if meta.get('source')!='web-pdf-wikipedia-replay':raise ValueError('Expected completed replay dataset.')
for name,expected in meta['file_sha256'].items():
    if file_hash(input_data/name)!=expected:raise ValueError(f'Corpus changed: {name}')
if (DATA_DIR/'metadata.json').exists():
    if file_hash(DATA_DIR/'metadata.json')!=file_hash(input_data/'metadata.json'):
        raise ValueError('Different input corpus for existing RUN_NAME.')
    for name,expected in meta['file_sha256'].items():
        if file_hash(DATA_DIR/name)!=expected:raise ValueError(f'Cached corpus changed: {name}')
else:
    pending=RUN_DIR/'data-replay.tmp'
    if pending.exists():raise FileExistsError('Incomplete copy exists; rename/remove data-replay.tmp before retry.')
    shutil.copytree(input_data,pending);os.replace(pending,DATA_DIR)
tokenizer=Tokenizer.from_file(str(DATA_DIR/'tokenizer.json'))
print('Corpus tokens:',meta['tokens'],'Wiki replay tokens:',meta['wikipedia_tokens'])
''')
md('''
## Checkpoint і фіксований target

Auto використовує спочатку власний volume last.pt, потім власні weights.pt,
потім імпортований last.pt або weights.pt. При ADDITIONAL_TOKENS>0 завжди нова
warm phase від input; повторне відкриття відновлює вже створений run.
Exact migration продовжує optimizer/RNG/LR recipe. Новий phase з weights —
warm restart. Target переноситься з input phase.json; якщо його немає і additional=0,
навчання не запускається автоматично. Введіть additional та новий run для нового бюджету.
''')
code('''
phase_path=RUN_DIR/'phase.json'
input_identity=file_hash(INPUT_DIR/'weights.pt')
if phase_path.exists():
    phase=json.loads(phase_path.read_text())
    if phase['input_weights_sha256']!=input_identity or phase['additional_tokens_requested']!=ADDITIONAL_TOKENS:
        raise ValueError('Input/budget changed; use a new RUN_NAME.')
else:
    initial_source=INPUT_DIR/'last.pt' if (INPUT_DIR/'last.pt').exists() else INPUT_DIR/'weights.pt'
    initial=torch.load(initial_source,map_location='cpu',weights_only=True,mmap=True)
    imported=json.loads((INPUT_DIR/'phase.json').read_text()) if (INPUT_DIR/'phase.json').exists() else {}
    start=initial['tokens_seen']
    phase={'initial_tokens':start if ADDITIONAL_TOKENS>0 else imported.get('initial_tokens',start),
           'target_tokens':start+ADDITIONAL_TOKENS if ADDITIONAL_TOKENS>0 else imported.get('target_tokens',start),
           'additional_tokens_requested':ADDITIONAL_TOKENS,'input_weights_sha256':input_identity}
    # A completed archive without a phase still needs a positive LR span for timing only.
    if phase['target_tokens']<=phase['initial_tokens']:phase['initial_tokens']=phase['target_tokens']-1
    del initial
    phase_path.write_text(json.dumps(phase,indent=2))
if RESUME_MODE=='auto' and (RUN_DIR/'last.pt').exists():resume_path,mode=RUN_DIR/'last.pt','exact'
elif (RUN_DIR/'weights.pt').exists():resume_path,mode=RUN_DIR/'weights.pt','warm'
elif RESUME_MODE=='auto' and ADDITIONAL_TOKENS==0 and (INPUT_DIR/'last.pt').exists():resume_path,mode=INPUT_DIR/'last.pt','exact'
else:resume_path,mode=INPUT_DIR/'weights.pt','warm'
header=torch.load(resume_path,map_location='cpu',weights_only=True,mmap=True)
model_cfg=ModelConfig(**header['model_config'])
if mode=='exact':
    hash_dir=RUN_DIR if resume_path.parent==RUN_DIR else INPUT_DIR
    if (hash_dir/'source_hashes.json').exists():
        old_hashes=json.loads((hash_dir/'source_hashes.json').read_text())
        if any(SOURCE_HASHES.get(name)!=value for name,value in old_hashes.items()):
            raise ValueError('Exact migration code changed. Use original code or explicit warm mode.')
    train_cfg=TrainConfig(**header['train_config'])
    # Restore saved recipe exactly, THEN explicitly change context while keeping Adam.
    if train_cfg.precision=='bf16_master' and not native_bf16:
        raise ValueError('Saved BF16 not supported on this GPU. Use RESUME_MODE=warm.')
else:
    precision='bf16_master' if native_bf16 else 'fp16_master'
    steps=header['step']+math.ceil(max(phase['target_tokens']-header['tokens_seen'],0)/(SEQ_LEN*MICROBATCH*GRAD_ACCUM))
    train_cfg=TrainConfig(steps=max(1,steps),sequence_length=SEQ_LEN,microbatch=MICROBATCH,
        accumulation=GRAD_ACCUM,learning_rate=LEARNING_RATE,warmup=1,restart_warmup=RESTART_WARMUP,
        precision=precision,seed=SEED,eval_batches=16,eval_every=500,save_every=500,log_every=10)
del header
trainer=Trainer(model_cfg,train_cfg,DATA_DIR,RUN_DIR,resume=resume_path,resume_mode=mode,
    checkpoint_dir=RUN_DIR,persistent_weights=True,disk_budget_gib=None)
(RUN_DIR/'source_hashes.json').write_text(json.dumps(SOURCE_HASHES,indent=2))
install_replay_sampling(trainer);fix_evaluation_microbatch(trainer,1);install_domain_evaluation(trainer)
install_phase_schedule(trainer,phase['initial_tokens'],phase['target_tokens'])
configure_training_context(trainer,SEQ_LEN,MICROBATCH,GRAD_ACCUM,VALIDATION_SEQ_LEN)
install_interrupt_safe_run(trainer)
print('Resume:',mode,'tokens:',trainer.tokens_seen,'target:',phase['target_tokens'])
print('Initial validation:',trainer.evaluate())
''')
md('''
## Короткий GPU benchmark і dollars

Тестові updates відкочуються; warm-up потребує одного справжнього update для
Adam allocation, лише якщо є незавершений budget. Для завершеного checkpoint
benchmark за замовчуванням пропускається, щоб не робити незапланований update.
PRICE_PER_HOUR_USD порівнює GPU compute cost, не додає storage/idle/setup overhead.
''')
code('''
if RUN_BENCHMARK and trainer.cfg.steps-trainer.step>=13:
    if not trainer.optimizer.state:trainer.run(max_updates=1)
    results=benchmark_batch_layout(trainer,units_per_hour=None)
    for row in results:
        speed=row['median_training_tokens_per_second']
        print('microbatch',row['microbatch'],'tokens/$',speed*3600/PRICE_PER_HOUR_USD)
    speed=max(r['median_training_tokens_per_second'] for r in results)
    hours=max(phase['target_tokens']-trainer.tokens_seen,0)/speed/3600
    print('Remaining training hours:',hours,'estimated GPU dollars:',hours*PRICE_PER_HOUR_USD)
else:print('Benchmark disabled or completed budget. Set an explicit new budget/run if needed.')
''')
md('''
## Навчання

Повторюйте лише цю ячейку для продовження поточного trainer. Не змінюйте phase
budget на кожному запуску. last.pt із optimizer та weights.pt зберігаються кожні500
updates й у кінці; після Stop перевірте повідомлення saved перед вимкненням Pod.
Обсяг останнього checkpoint і target друкуються. Completed goal не додає tokens.
''')
code('''
started=time.perf_counter()
if RUN_HOURS is None:
    final_metrics=trainer.run(max_updates=SESSION_UPDATES)
else:
    training_report=run_for_hours(trainer,hours=RUN_HOURS,chunk_updates=SESSION_UPDATES or 1000)
    final_metrics=training_report['last_metrics']
    print(training_report)
print('Minutes:',(time.perf_counter()-started)/60,'step:',trainer.step,'tokens:',trainer.tokens_seen)
print('Validation:',final_metrics)
print('Persistent run:',RUN_DIR)
_=(RUN_DIR/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
''')
md('## Графіки loss, експертів і швидкості')
base=json.loads((ROOT/'notebooks/next_moe_55m_plus10m_replay.ipynb').read_text())
graph=base['cells'][17]['source'].split('baseline=json.loads')[0]
graph=graph.replace("[('val_ce_', '-'), ('val_new_ce_', '--')]",
    "[('val_ce_', '-'), ('val_new_ce_', '--'), ('val_ctx2048_ce_', ':'), ('val_ctx2048_new_ce_', '-.')]")
# A zero-update migrated archive may have no training metrics yet.
code("if not trainer.log_path.exists():\n    print('No training metrics yet; run an unfinished training stage first.')\nelse:\n"+'\n'.join('    '+line for line in graph.splitlines()))
md('## Генерація')
code(base['cells'][19]['source'])
md('''
## Завершення

Перевірте last.pt/weights.pt на постійному mount, збережіть notebook і резервну
копію поза Pod. Stop/Terminate мають різні наслідки для різних storage types;
перевірте поточні RunPod terms. Container disk не вважати довготривалим сховищем.
Stop може залишати платне storage, запущений idle GPU теж коштує гроші.
Не відкривати Jupyter без пароля/token; користуйтеся Connect у RunPod.
''')
for i,cell in enumerate(cells):cell['id']=f'runpod-{i:02d}'
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
p=ROOT/'notebooks/next_moe_runpod.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Built',p.name,len(cells),'cells')
