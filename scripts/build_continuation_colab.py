"""Build self-contained 30M -> +20M UK/EN Colab continuation workflow."""
from pathlib import Path
import hashlib
import json
import textwrap
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip()+'\n'})
def code(s): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                           'source':textwrap.dedent(s).strip()+'\n'})
md('''
# next-moe: checkpoint 30M → ще 20M на нових UK/EN даних

**Спочатку Runtime → Change runtime type → L4.** Notebook автономний, код вбудований.
Модель/Engram/BPE беруться з вашого checkpoint; це continued pretraining, не chat SFT.
Новий corpus run має окремий каталог і не перезаписує checkpoint 30M.

80% web + 20% освітніх PDF за tokens, 50/50 UK/EN за sampled sequences.
UK: FineWeb2 `ukr_Cyrl`; EN: FineWeb-Edu `CC-MAIN-2025-26`; PDF: FinePDFs-Edu UK/EN.
Ревізії закріплені, corpus caps — **20M tokens/lang** (≈160MB int32 train files).
Старий Wikipedia validation залишається byte-identical; новий holdout окремий.

**Навчання за замовчуванням іде до кінця +20M**, без ліміту 1000 updates.
Завершення GPU runtime знищує локальний Adam checkpoint. Drive зберігає weights
≈0.55GiB, metadata і token files; після втрати runtime буде warm restart.
Підготовка даних також витрачає час GPU-сеансу; corpus cache повторно не збирається.
''')
code('''
import sys, subprocess, json, math, time, gc, shutil, hashlib, os
from pathlib import Path
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'datasets==3.6.0', 'tokenizers==0.21.1', 'matplotlib>=3.8,<4'])
import torch
print('PyTorch:', torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError('Оберіть GPU runtime; для L4: Runtime → Change runtime type.')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory/2**30)
print('Native BF16:', torch.cuda.is_bf16_supported(including_emulation=False))
''')
md('''
## Параметри та checkpoint

BASE_RUN — frozen checkpoint-30m-frozen із попередньої інструкції. Якщо його немає,
вкажіть каталог робочого checkpoint 30M. Він має містити weights.pt і data-v2/.
RUN_NAME нового етапу інший. COMPUTE_UNITS_PER_HOUR введіть із поточної панелі
Colab для **L4**: не використовуємо стару ціну G4 8.9 як тариф L4.
''')
code('''
USE_DRIVE = True
STORAGE_PATH = '/content/drive/MyDrive/next-moe'
BASE_RUN_NAME = 'checkpoint-30m-frozen'
BASE_DATA_NAME = 'data-v2'
RUN_NAME = 'next-moe-30m-plus20m-webpdf-v1'
ADDITIONAL_TOKENS = 20_000_000
TOKENS_PER_LANGUAGE = 20_000_000
NEW_VALIDATION_DOCUMENTS = 128
PDF_FRACTION = .2
SEQ_LEN = 512
MICROBATCH = 4
GRAD_ACCUM = 1
PRECISION = 'auto_master'     # L4 supports native BF16; FP32 master Adam
LEARNING_RATE = 1e-4          # new phase, conservative LR compared with original 2e-4
RESTART_WARMUP = 100
RUN_EFFICIENCY_BENCHMARK = True
COMPUTE_UNITS_PER_HOUR = None # enter actual L4 units/hour; None measures speed only
SESSION_UPDATES = None       # None = train to +20M goal; 1000 = deliberately pause
RESUME_MODE = 'auto'         # auto / warm (explicit optimizer reset)
PERSISTENT_BUDGET_GIB = 4.5   # NEW run folder only, not entire Drive quota
SEED = 1234
SHUFFLE_BUFFER = 256
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
STORAGE = Path(STORAGE_PATH)
BASE_RUN = STORAGE / BASE_RUN_NAME
OLD_DATA = BASE_RUN / BASE_DATA_NAME
RUN_DIR = STORAGE / RUN_NAME
LOCAL_DIR = Path('/content/next-moe-local') / RUN_NAME
CODE_DIR = Path('/content/next-moe-continuation-code')
DATA_DIR = RUN_DIR / 'data-stage3'
CACHE_DIR = LOCAL_DIR / 'data'
for folder in (RUN_DIR, LOCAL_DIR, CODE_DIR): folder.mkdir(parents=True, exist_ok=True)
if RUN_DIR.resolve() == BASE_RUN.resolve():
    raise ValueError('New run must not overwrite frozen base run.')
''')
md('## Вбудовані вихідні файли')
sources = {str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'next_moe').glob('*.py'))}
hashes = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
code('SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'''
for name, content in SOURCES.items():
    target = CODE_DIR/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content)
if str(CODE_DIR) not in sys.path: sys.path.insert(0, str(CODE_DIR))
from next_moe.model import ModelConfig, NextMoE
from next_moe.train import TrainConfig, Trainer
from next_moe.corpus import prepare_continuation_data, install_domain_evaluation, install_phase_schedule
from tokenizers import Tokenizer
'''+ 'exec('+repr((ROOT/'scripts/colab_efficiency.py').read_text())+')\n')
md('''
## Перевірка базової моделі та незмінного tokenizer

При першому запуску +20M відраховуються від фактичного tokens_seen checkpoint,
а не від припущення, що він рівно 30,000,000. Phase target зберігається на Drive.
Повторне відкриття не додає ще 20M і не повторює extend_token_budget.
''')
code('''
base_checkpoint = BASE_RUN/'weights.pt'
for path in (base_checkpoint, OLD_DATA/'tokenizer.json', OLD_DATA/'metadata.json',
             OLD_DATA/'val_uk.bin', OLD_DATA/'val_en.bin'):
    if not path.exists():
        raise FileNotFoundError(f'{path}. Set BASE_RUN_NAME/BASE_DATA_NAME to your saved 30M run.')
phase_path = RUN_DIR/'phase.json'
if phase_path.exists():
    phase = json.loads(phase_path.read_text())
    if phase['additional_tokens'] != ADDITIONAL_TOKENS:
        raise ValueError('Existing phase has a different budget. Use a new RUN_NAME.')
    base_sha = phase['base_weights_sha256']
    if str(BASE_RUN) != phase['base_run'] or hashlib.sha256(base_checkpoint.read_bytes()).hexdigest() != base_sha:
        raise ValueError('Frozen base checkpoint changed. Restore it or use a NEW RUN_NAME.')
else:
    base_sha = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    base_local = LOCAL_DIR/'base-weights.pt'
    shutil.copy2(base_checkpoint, base_local)
    header = torch.load(base_local, map_location='cpu', weights_only=True, mmap=True)
    if not 27_000_000 <= header['tokens_seen'] <= 32_000_000:
        raise ValueError('Expected checkpoint near 30M; verify BASE_RUN before continuing.')
    phase = {'initial_step':header['step'], 'initial_tokens':header['tokens_seen'],
             'additional_tokens':ADDITIONAL_TOKENS,
             'target_tokens':header['tokens_seen']+ADDITIONAL_TOKENS,
             'base_weights_sha256':base_sha, 'base_run':str(BASE_RUN),
             'tokenizer_sha256':header['tokenizer_signature']}
    del header
    phase_path.write_text(json.dumps(phase, indent=2))
if hashlib.sha256((OLD_DATA/'tokenizer.json').read_bytes()).hexdigest() != phase['tokenizer_sha256']:
    raise ValueError('Original tokenizer does not match checkpoint. Do not retrain BPE.')
print('Phase:', phase)
''')
md('''
## Новий корпус: потокове читання, кеш і походження

Перший запуск займає час: progress друкується під час сканування. Цілі — training
**tokens**, не documents. Частки 80/20 наближені з точністю до останнього документа.
Використовуємо bounded shuffle, exact hashes і approximate SimHash дедуплікацію;
це не гарантія повної відсутності близьких дублікатів. Фільтруються короткі,
непрозові та повторювані тексти. Весь пул — природні джерельні тексти, classifiers
якості авторів мають LLM-анотації; PDF/OCR може містити помилки.

metadata.json містить dataset/config/revision, exact counts, reject counters,
мікс за tokens і hashes. documents.jsonl зберігає URL та походження кожного документа.
Після збою неповний каталог не використовується як готовий корпус. Помилки мережі
не підміняються toy data. У разі зміни corpus settings використовуйте новий RUN_NAME.
''')
code('''
if (DATA_DIR/'metadata.json').exists():
    metadata = prepare_continuation_data(DATA_DIR, OLD_DATA,
        tokens_per_language=TOKENS_PER_LANGUAGE, validation_documents=NEW_VALIDATION_DOCUMENTS,
        pdf_fraction=PDF_FRACTION, seed=SEED, shuffle_buffer=SHUFFLE_BUFFER)
    shutil.copytree(DATA_DIR, CACHE_DIR, dirs_exist_ok=True)
else:
    build_dir = LOCAL_DIR/'corpus-build'
    metadata = prepare_continuation_data(build_dir, OLD_DATA,
        tokens_per_language=TOKENS_PER_LANGUAGE, validation_documents=NEW_VALIDATION_DOCUMENTS,
        pdf_fraction=PDF_FRACTION, seed=SEED, shuffle_buffer=SHUFFLE_BUFFER)
    estimated = sum(p.stat().st_size for p in build_dir.rglob('*') if p.is_file())
    used = sum(p.stat().st_size for p in RUN_DIR.rglob('*') if p.is_file())
    if (used+estimated+1.2*2**30)/2**30 > PERSISTENT_BUDGET_GIB:
        raise OSError('New run data plus two temporary weights snapshots exceed configured run budget.')
    pending = RUN_DIR/'data-stage3.tmp'
    if pending.exists():
        raise FileExistsError('Incomplete Drive corpus copy exists: rename/remove data-stage3.tmp before retry.')
    shutil.copytree(build_dir, pending)
    os.replace(pending, DATA_DIR)
    shutil.copytree(build_dir, CACHE_DIR, dirs_exist_ok=True)
print(json.dumps({k:metadata[k] for k in ('tokens','source_training_tokens','document_counts','rejected')}, indent=2))
tokenizer = Tokenizer.from_file(str(CACHE_DIR/'tokenizer.json'))
for lang in ('uk','en'):
    assert (CACHE_DIR/f'val_{lang}.bin').read_bytes() == (OLD_DATA/f'val_{lang}.bin').read_bytes()
print('Tokenizer reused; old validation byte-identical.')
''')
md('''
## Завантаження моделі та новий етап

З базового weights.pt — **warm restart**: ваги збережені, Adam новий, LR плавно
піднімається за 100 updates. LR ceiling 1e-4 для нового етапу (гіпотеза, стежте за validation).
Auto у поточному runtime обирає local last.pt з optimizer; після втрати runtime —
новий weights.pt на Drive. Читаємо model_config із checkpoint, не змінюємо архітектуру.
L4/BF16: weights і gradients моделі BF16, master weights/Adam FP32.
''')
code('''
if RESUME_MODE == 'auto' and (LOCAL_DIR/'last.pt').exists():
    resume_path, mode = LOCAL_DIR/'last.pt', 'exact'
elif (RUN_DIR/'weights.pt').exists():
    resume_path, mode = RUN_DIR/'weights.pt', 'warm'
else:
    resume_path, mode = base_checkpoint, 'warm'
if mode == 'warm':
    local_weights = LOCAL_DIR/'input-weights.pt'
    shutil.copy2(resume_path, local_weights)
    resume_path = local_weights
header = torch.load(resume_path, map_location='cpu', weights_only=True, mmap=True)
model_cfg = ModelConfig(**header['model_config'])
if mode == 'exact':
    hash_path = RUN_DIR/'source_hashes.json'
    if hash_path.exists() and json.loads(hash_path.read_text()) != SOURCE_HASHES:
        raise ValueError('Exact resume code changed. Use original notebook or explicit RESUME_MODE=warm.')
    train_cfg = TrainConfig(**header['train_config'])
    if train_cfg.sequence_length != SEQ_LEN:
        raise ValueError('Exact resume seq_len differs; keep SEQ_LEN or explicitly use warm.')
else:
    precision = ('bf16_master' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
                 else 'fp16_master' if torch.cuda.is_available() else 'fp32') if PRECISION=='auto_master' else PRECISION
    per_step = SEQ_LEN*MICROBATCH*GRAD_ACCUM
    steps = header['step'] + math.ceil(max(phase['target_tokens']-header['tokens_seen'],0)/per_step)
    train_cfg = TrainConfig(steps=max(steps,1), sequence_length=SEQ_LEN, microbatch=MICROBATCH,
        accumulation=GRAD_ACCUM, learning_rate=LEARNING_RATE, warmup=1,
        restart_warmup=RESTART_WARMUP, precision=precision, seed=SEED,
        eval_batches=16, eval_every=500, save_every=500, log_every=10)
del header
trainer = Trainer(model_cfg, train_cfg, CACHE_DIR, RUN_DIR, resume=resume_path,
    resume_mode=mode, checkpoint_dir=LOCAL_DIR, persistent_weights=True,
    disk_budget_gib=PERSISTENT_BUDGET_GIB)
(RUN_DIR/'source_hashes.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
fix_evaluation_microbatch(trainer, 1)
install_domain_evaluation(trainer)
install_phase_schedule(trainer, phase['initial_tokens'], phase['target_tokens'])
print('Resume:', mode, 'step:', trainer.step, 'tokens:', trainer.tokens_seen)
print('Goal:', phase['target_tokens'], 'remaining:', max(phase['target_tokens']-trainer.tokens_seen,0))
print('Initial validation:', trainer.evaluate())
''')
md('''
## Бенчмарк L4 і квота

Порівнюються microbatch 1/2/4 із незмінним effective batch (2048 tokens/update),
а за ввімкненого block checkpointing — також його вимкнення. Модель, Adam/RNG і
progress відновлюються після тестових updates. Benchmark витрачає час/квоту,
але не додає тестові tokens до навчання. Host RAM для backup ≈full checkpoint size.
На новому Adam спочатку виконується **один справжній** update. Тест автоматично
застосовує fastest layout. Якщо OOM, відновлює стан і зупиняється; зменшіть
MICROBATCH або ввімкніть checkpointing у **новій warm фазі**.

Введіть поточні L4 units/hour, щоб побачити tokens/unit. Старий G4 benchmark:
~6420 tokens/s при 8.9 units/hour ≈2.60M tokens/unit. L4 вигідніший, коли його
speed / units_per_hour > 6420 / 8.9 ≈721.3. Оцінка без validation/save overhead.
''')
code('''
if RUN_EFFICIENCY_BENCHMARK and trainer.cfg.steps-trainer.step >= 13:
    if not trainer.optimizer.state:
        trainer.run(max_updates=1)
    benchmark_results = benchmark_batch_layout(trainer, units_per_hour=COMPUTE_UNITS_PER_HOUR)
    speed = max(r['median_training_tokens_per_second'] for r in benchmark_results)
    hours = max(phase['target_tokens']-trainer.tokens_seen,0)/speed/3600
    print('Estimated remaining TRAINING hours (exclude eval/save):', round(hours,2))
    if COMPUTE_UNITS_PER_HOUR:
        print('Estimated units:', round(hours*COMPUTE_UNITS_PER_HOUR,2))
else:
    print('Benchmark disabled or goal too close. Current batch:', trainer.cfg.microbatch, trainer.cfg.accumulation)
''')
md('''
## Навчання до +20M

SESSION_UPDATES=None означає весь залишок бюджету. Зупинка кнопкою Stop намагається
зберегти останній завершений optimizer step. Повторне виконання цієї ячейки
продовжує поточний trainer, не пересоздає Adam і не змінює LR budget.
Для графіків посеред навчання можна навмисно поставити SESSION_UPDATES=1000.
''')
code('''
started = time.perf_counter()
final_metrics = trainer.run(max_updates=SESSION_UPDATES)
print('Minutes:', round((time.perf_counter()-started)/60,2))
print('Step:', trainer.step, 'tokens_seen:', trainer.tokens_seen, 'phase goal:', phase['target_tokens'])
print('Validation:', final_metrics)
weights_path = RUN_DIR/'weights.pt'
print('Drive weights:', weights_path)
_ = (RUN_DIR/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
''')
md('''
## Графіки loss, routing і швидкості

Стара Wikipedia validation — суцільні лінії; new-domain holdout — пунктир.
Metrics цього етапу окремі від старого run. Final evaluation теж показується,
навіть якщо останній update не кратний eval_every. Router heatmap — поточна фаза.
''')
code('''
import matplotlib.pyplot as plt
records = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
fig, axes = plt.subplots(1,3,figsize=(17,4))
axes[0].plot([r['tokens_seen']/1e6 for r in records],[r['train_ce'] for r in records],alpha=.45,label='train')
for lang in ('uk','en'):
    for prefix, style in [('val_ce_', '-'), ('val_new_ce_', '--')]:
        key=prefix+lang; subset=[r for r in records if key in r]
        xs=[r['tokens_seen']/1e6 for r in subset]; ys=[r[key] for r in subset]
        if 'final_metrics' in globals() and key in final_metrics:
            xs.append(trainer.tokens_seen/1e6); ys.append(final_metrics[key])
        axes[0].plot(xs,ys,style,label=key)
axes[0].set(xlabel='Processed tokens (M)',ylabel='Cross entropy');axes[0].legend(fontsize=8)
counts=trainer.routing_total.float().cpu()
im=axes[1].imshow((counts/counts.mean(-1,keepdim=True).clamp_min(1)).numpy(),aspect='auto')
axes[1].set(xlabel='Expert',ylabel='Layer',title='Assignments / mean, phase');fig.colorbar(im,ax=axes[1])
axes[2].plot([r['tokens_seen']/1e6 for r in records],[r['tokens_per_second_training_step'] for r in records],alpha=.6,label='training step')
subset=[r for r in records if 'tokens_per_second_including_interval_overhead' in r]
axes[2].plot([r['tokens_seen']/1e6 for r in subset],[r['tokens_per_second_including_interval_overhead'] for r in subset],label='interval incl. overhead')
axes[2].set(xlabel='Processed tokens (M)',ylabel='Tokens / second');axes[2].legend(fontsize=8)
plt.tight_layout();plt.show()
''')
md('## Перевірка генерації (base model, не чат)')
code('''
for prompt in ['Україна — це', 'The history of science']:
    torch.manual_seed(1234)
    ids=tokenizer.encode(prompt).ids
    with trainer.autocast():
        generated=trainer.model.generate(torch.tensor([ids],device=trainer.device),
            max_new_tokens=80, temperature=.8, top_k=40, context=SEQ_LEN,
            valid_vocab=tokenizer.get_vocab_size())
    print(tokenizer.decode(generated[0].tolist())); print()
''')
md('''
## Збереження та наступне відкриття

Drive: RUN_DIR/weights.pt + tokenizer.json + data-stage3/ + phase.json + configs,
source_hashes.json, metrics.jsonl, pip-freeze.txt. Local: LOCAL_DIR/last.pt із Adam.
Не дублюємо full checkpoint на Drive автоматично. frozen checkpoint 30M залишається
окремо; загальна квота Drive включає обидва run та інші файли. Run limit 4.5GiB
не перевіряє квоту всього Google Drive. При відсутності місця save видасть помилку.

Документація: https://github.com/FactorialD/next-moe/blob/work/docs/continuation-colab.md
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'next_moe_30m_to_50m_webpdf.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for index, cell in enumerate(cells): cell['id']=f'continuation-{index:02d}'
path=ROOT/'notebooks/next_moe_30m_to_50m_webpdf.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Built',path.name,len(cells),'cells')
