"""Build a standalone Colab notebook embedding the current reference source."""
from pathlib import Path
import hashlib
import json
import textwrap
ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n'})


def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': textwrap.dedent(source).strip() + '\n'})


markdown('''
# next-moe 0.2B + 0.1B Engram: следующий UK/EN эксперимент

Самостоятельный notebook, исходники встроены. Полный профиль: **293.69M параметров**, 12×512,
16 routed / top-2 + shared, GR×4, один Engram, GDN/full GQA 3:1 и pairwise shared KV.

**Runtime → Change runtime type → GPU.** На T4 по умолчанию FP16 веса/градиенты модели,
FP32 master weights и Adam moments, autocast + GradScaler. BF16 — для поддерживающих GPU.
Это безопасная mixed precision схема, а не FP16 Adam целиком. Она не обещает двукратной экономии
всей VRAM: master weights и transient FP32 optimizer gradients также занимают память.

План по умолчанию **10 млн processed tokens**, около 19 532 updates при batch 512 tokens,
по **1000 updates за сессию**. Это следующий эксперимент, не законченный pretraining или SFT.
По пользовательскому тесту 300 updates заняли 25.11 минуты: без ускорения 10M tokens ≈27.25 часа.
Сначала проверьте 20–50 updates нового FP16 режима, затем продолжайте до session limit.

Корпус — открытый [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
configs `20231101.uk/en`, pinned revision. До 10 000 документов и 30M Unicode characters на язык.
Для продолжения предыдущего run BPE и validation сохраняются. После перезапуска Colab нельзя
переобучать tokenizer существующей модели. Учитывайте лицензии корпуса при перераспространении.
''')
code('''
%pip -q install tokenizers==0.21.4 datasets==3.6.0 matplotlib
import torch, sys, json, gc, time, shutil, math
from pathlib import Path
print('Python:', sys.version.split()[0], 'Torch:', torch.__version__)
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6), 'Нужен PyTorch >=2.6.'
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(props.name, round(props.total_memory/2**30, 2), 'GiB; BF16:', torch.cuda.is_bf16_supported(including_emulation=False))
''')
markdown('''
## Бюджет, сессия и диск

В Drive сохраняется **один `weights.pt`**, около 0.55 GiB в FP16 (1.09 GiB при FP32).
При записи временного файла старый snapshot остаётся: пиковый объём двух FP16 snapshots ≈1.10 GiB.
Данные/metadata/source тоже входят в бюджет run **4.5 GiB**; это заданный лимит одного каталога,
а не проверка реальной свободной квоты Google Drive. Укажите лимит меньше доступного места.
Старые `last.pt` (3.28 GiB) и `model.pt` (1.09 GiB) предыдущего run сами не удаляются. Если они
уже заняли доступное место, освободите/скачайте старые файлы после проверки нового снимка весов.

При USE_DRIVE=False снимок весов тоже остаётся в /content и исчезает при потере runtime.
Полный optimizer checkpoint сохраняется **локально в /content**, не в Drive. После потери runtime
Drive snapshot требует **warm restart**: веса/step/tokens сохраняются, optimizer/master precision/RNG
начинаются заново, LR плавно нарастает 100 updates. Это не точное продолжение.
Пока local `last.pt` доступен, auto использует exact resume с optimizer/RNG. Для точного resume
после потери runtime понадобится отдельно хранить full checkpoint (FP16+master ≈3.83 GiB;
с временной копией ≈7.66 GiB). Вторая копия inference model по умолчанию не создаётся.
''')
code('''
PROFILE = 'full'               # tiny только для correctness tests
USE_DRIVE = True
RUN_NAME = 'next-moe-0p2b-engram0p1b-10m-v2'
TOTAL_TOKENS = 10_000_000      # включает tokens предыдущего checkpoint
SESSION_UPDATES = 1000        # сначала можно поставить 20 или 50
STEPS_OVERRIDE = None         # только debugging; обычно None
SEQ_LEN = 128
MICROBATCH = 1
GRAD_ACCUM = 4
PRECISION = 'auto_master'     # auto: fp16_master T4, bf16_master supported GPUs, fp32 CPU
OPTIMIZER = 'adamw'           # optional adamw8bit; не меняйте optimizer при exact resume
CHECKPOINT_BLOCKS = True
DOCUMENTS_PER_LANGUAGE = 10000
VALIDATION_DOCUMENTS = 128    # при продолжении сохраняем старые 32 вместо новых 128
MAX_CHARACTERS_PER_LANGUAGE = 30_000_000
SOURCE = 'wikipedia'
LOCAL_TEXT_DIR = '/content/my_texts'
SEED = 1234
PERSISTENT_BUDGET_GIB = 4.5   # лимит всего каталога run, включая temporary snapshot
RESUME_MODE = 'auto'         # auto / exact / warm / fresh
INIT_FROM_TEST_RUN = True
INIT_CHECKPOINT = ''         # можно явно указать старый last.pt или новый weights.pt
INIT_TOKENIZER = ''
INIT_METADATA = ''

assert PROFILE in ('full', 'tiny')
if PROFILE == 'full' and not torch.cuda.is_available():
    raise RuntimeError('Для полного профиля включите GPU; CPU предназначен для tiny tests.')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/next-moe')
else:
    STORAGE = Path('/content/next-moe-artifacts')
RUN_DIR = STORAGE / RUN_NAME
LOCAL_DIR = Path('/content/next-moe-local') / RUN_NAME
DATA_DIR = RUN_DIR / 'data-v2'
CACHE_DIR = LOCAL_DIR / 'data'
CODE_DIR = Path('/content/next-moe-code')
for directory in (RUN_DIR, LOCAL_DIR, CODE_DIR): directory.mkdir(parents=True, exist_ok=True)
OLD_RUN = STORAGE / 'next-moe-0p2b-engram0p1b-test-v1'
if RESUME_MODE != 'fresh' and INIT_FROM_TEST_RUN and not INIT_CHECKPOINT and (OLD_RUN / 'last.pt').exists():
    INIT_CHECKPOINT = str(OLD_RUN / 'last.pt')
    INIT_TOKENIZER = str(OLD_RUN / 'tokenizer.json')
    INIT_METADATA = str(OLD_RUN / 'data/metadata.json')
if RESUME_MODE == 'auto':
    if (LOCAL_DIR / 'last.pt').exists():
        resume_path, restore_mode = LOCAL_DIR / 'last.pt', 'exact'
    elif (RUN_DIR / 'weights.pt').exists():
        resume_path, restore_mode = RUN_DIR / 'weights.pt', 'warm'
    elif INIT_CHECKPOINT:
        resume_path, restore_mode = Path(INIT_CHECKPOINT), 'warm'
    else:
        resume_path, restore_mode = None, 'fresh'
elif RESUME_MODE in ('exact', 'warm'):
    resume_path = Path(INIT_CHECKPOINT) if INIT_CHECKPOINT else (
        LOCAL_DIR / 'last.pt' if RESUME_MODE == 'exact' else RUN_DIR / 'weights.pt')
    restore_mode = RESUME_MODE
elif RESUME_MODE == 'fresh':
    if (RUN_DIR / 'weights.pt').exists() or (LOCAL_DIR / 'last.pt').exists():
        raise RuntimeError('Для fresh запуска с новыми весами используйте новое RUN_NAME.')
    resume_path, restore_mode = None, 'fresh'
else:
    raise ValueError(RESUME_MODE)
if resume_path is not None and not resume_path.exists(): raise FileNotFoundError(resume_path)
if OPTIMIZER == 'adamw8bit':
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes==0.45.5'])
print('Restore mode:', restore_mode, 'checkpoint:', resume_path)
''')
markdown('''
## Встроенные исходники

Reference PyTorch kernels: triangular GDN FP32, no-drop MoE, GR, Engram, SDPA.
FlashAttention backend зависит от GPU/dtype. FP32 master optimizer использует FP32 gradients после
копирования scaled gradients модели; GradScaler не выполняет unscale_ на FP16 tensors.
Persistent KV/GDN decode API и production Triton/fused MoE ещё не реализованы.
''')
sources = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT/'next_moe').glob('*.py'))}
hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()}
code('SOURCES = ' + repr(sources) + '\nSOURCE_HASHES = ' + repr(hashes) + '''
for relative, content in SOURCES.items():
    target = CODE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
if str(CODE_DIR) not in sys.path: sys.path.insert(0, str(CODE_DIR))
from next_moe.model import ModelConfig, NextMoE
from next_moe.data import prepare_data
from next_moe.train import TrainConfig, Trainer
hash_path = RUN_DIR / 'source_hashes.json'
if restore_mode == 'exact' and hash_path.exists() and json.loads(hash_path.read_text()) != SOURCE_HASHES:
    raise RuntimeError('Для exact resume нужен тот же код; для новой фазы выберите warm.')
hash_path.write_text(json.dumps(SOURCE_HASHES, indent=2))
shutil.copytree(CODE_DIR/'next_moe', RUN_DIR/'source', dirs_exist_ok=True)
''')
markdown('''
## Короткая проверка графа

Tiny-модель проверяет causal prefix и backward. Новый FP16+master режим сначала проверяется короткой
сессией полноценной модели; прошлый пользовательский тест подтверждал FP32 weights + autocast.
''')
code('''
torch.manual_seed(SEED)
check_device = 'cuda' if torch.cuda.is_available() else 'cpu'
small = NextMoE(ModelConfig.tiny()).to(check_device)
probe = torch.randint(4, 512, (1, 9), device=check_device)
small.eval()
with torch.no_grad():
    full_logits, prefix_logits = small(probe), small(probe[:, :5])
torch.testing.assert_close(full_logits[:, :5], prefix_logits, atol=2e-4, rtol=2e-3)
small.train()
probe_loss = small(probe[:, :-1], probe[:, 1:])
(probe_loss['ce'] + .01*probe_loss['aux']).backward()
assert all(torch.isfinite(p.grad).all() for p in small.parameters() if p.grad is not None)
print('Graph smoke: OK')
del small, probe, probe_loss, full_logits, prefix_logits
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
''')
markdown('''
## Корпус и неизменный tokenizer

При наличии INIT_CHECKPOINT используются INIT_TOKENIZER/INIT_METADATA, чтобы расширить corpus,
сохранив token IDs и прежние validation documents. BPE не retrain. Для fresh run обучается новый BPE.
Документы deduplicate/split до обучения tokenizer. Остальная схема: ByteLevel, case/accents preserved,
max 20 000 symbols/article, EOS между документами, UK/EN sampling 50/50.

Token files копируются в local cache: random windows не читаются с Drive на каждом микробатче.
Фактический объём/ожидаемые проходы печатаются. Повторение маленького корпуса не заменяет новые данные.
Streaming/CDN errors останавливают подготовку, без скрытой замены toy corpus.
''')
code('''
model_cfg = ModelConfig() if PROFILE == 'full' else ModelConfig.tiny()
model_cfg.checkpoint_blocks = CHECKPOINT_BLOCKS
if not (DATA_DIR/'metadata.json').exists():
    if restore_mode == 'exact' or (RUN_DIR/'weights.pt').exists():
        raise RuntimeError('Нет исходных данных текущего run; восстановите их, не retrain tokenizer.')
    reuse_tokenizer = (INIT_TOKENIZER or None) if resume_path is not None else None
    reuse_validation = (INIT_METADATA or None) if resume_path is not None else None
    prepare_data(DATA_DIR, source=SOURCE, local_dir=LOCAL_TEXT_DIR,
                 documents_per_language=DOCUMENTS_PER_LANGUAGE, validation_documents=VALIDATION_DOCUMENTS,
                 vocab_size=model_cfg.vocab_size, seed=SEED, tokenizer_path=reuse_tokenizer,
                 fixed_validation_metadata=reuse_validation,
                 max_characters_per_language=MAX_CHARACTERS_PER_LANGUAGE)
metadata = json.loads((DATA_DIR/'metadata.json').read_text())
used = sum(p.stat().st_size for p in RUN_DIR.rglob('*') if p.is_file())
if used > PERSISTENT_BUDGET_GIB*2**30: raise OSError('Данные уже превысили budget run.')
shutil.copytree(DATA_DIR, CACHE_DIR, dirs_exist_ok=True)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(CACHE_DIR/'tokenizer.json'))
for text in ['Україна: і, ї, й, є, ґ; пам’ять.', 'Knowledge and language.']:
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
print(json.dumps({k:metadata[k] for k in ('source_revision','document_counts','actual_vocab','tokens')}, indent=2))
for lang in ['uk','en']:
    print(lang, 'expected sampled passes at 10M:', TOTAL_TOKENS/2/metadata['tokens'][f'train_{lang}'])
''')
markdown('''
## Создание модели, LR и ограничение сессии

LR schedule рассчитан на **весь token budget**, а не на SESSION_UPDATES. При прерывании сессии
кривая не доходит до минимального LR преждевременно. Warm restart дополнительно вводит плавный ramp.
Baseline validation теперь 16 фиксированных окон на язык с независимыми RNG; эти значения не надо
напрямую сравнивать с прежними 4 окнами. Начните с 20–50 updates FP16, проверьте loss/gradients/VRAM.
''')
code('''
precision = PRECISION
if precision == 'auto_master':
    precision = ('bf16_master' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
                 else 'fp16_master' if torch.cuda.is_available() else 'fp32')
initial_step, initial_tokens = 0, 0
if resume_path is not None:
    header = torch.load(resume_path, map_location='cpu', weights_only=True, mmap=True)
    if 'step' not in header:
        raise ValueError('Legacy inference model.pt lacks progress metadata; use old last.pt or new weights.pt.')
    initial_step, initial_tokens = header['step'], header['tokens_seen']
    del header
per_step = SEQ_LEN*MICROBATCH*GRAD_ACCUM
steps = initial_step + math.ceil(max(TOTAL_TOKENS-initial_tokens, 0)/per_step)
if STEPS_OVERRIDE is not None: steps = STEPS_OVERRIDE
steps = max(steps, 1)
train_cfg = TrainConfig(steps=steps, sequence_length=SEQ_LEN, microbatch=MICROBATCH,
                        accumulation=GRAD_ACCUM, precision=precision, optimizer=OPTIMIZER, seed=SEED,
                        warmup=min(200, max(steps//100, 1)), restart_warmup=100,
                        eval_every=500, eval_batches=16, save_every=500, log_every=10)
trainer = Trainer(model_cfg, train_cfg, CACHE_DIR, RUN_DIR, resume=resume_path,
                  resume_mode='exact' if restore_mode=='exact' else 'warm',
                  resume_tokenizer=INIT_TOKENIZER or None, checkpoint_dir=LOCAL_DIR,
                  persistent_weights=True, disk_budget_gib=PERSISTENT_BUDGET_GIB)
remaining_tokens = max(TOTAL_TOKENS-trainer.tokens_seen, 0)
print('Parameter dtype:', next(trainer.model.parameters()).dtype)
print('Total updates:', steps, 'session max:', SESSION_UPDATES, 'tokens/update:', per_step)
print('Remaining tokens:', remaining_tokens, 'rough hours at observed ~102 tokens/s:', remaining_tokens/102/3600)
print('Initial validation for current evaluator:', trainer.evaluate())
''')
markdown('''
## Обучение текущей сессии

Data, training step и interval throughput логируются отдельно, чтобы видеть влияние Drive saves.
Local full checkpoint и persistent weights сохраняются каждые 500 updates и в конце сессии,
без двойной записи на последнем update. Phase-wide load metrics дополняют last-update routing.
После завершения ячейки можно выполнить её ещё раз: модель/optimizer в памяти продолжают сессию.
После потери runtime auto загрузит weights из Drive с явным warm restart.
''')
code('''
start = time.perf_counter()
final_metrics = trainer.run(max_updates=SESSION_UPDATES)
print('Elapsed minutes:', round((time.perf_counter()-start)/60, 2), 'step:', trainer.step)
print('Validation:', final_metrics)
weights_path = RUN_DIR/'weights.pt'
print('Persistent weights:', weights_path)
import subprocess
(RUN_DIR/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
''')
markdown('''
## Графики и router load

Карта показывает накопленные assignments текущей optimizer phase, нормированные по среднему слоя.
Она информативнее единственного микробатча. Near-one auxiliary loss не доказывает равномерную нагрузку.
Исторические metrics предыдущих сессий сохранены; после warm restart phase counters начинаются заново.
''')
code('''
import matplotlib.pyplot as plt
records = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot([r['step'] for r in records],[r['train_ce'] for r in records],label='train',alpha=.6)
for lang in ['uk','en']:
    validation = [r for r in records if f'val_ce_{lang}' in r]
    ax[0].plot([r['step'] for r in validation],[r[f'val_ce_{lang}'] for r in validation],label=f'val {lang}')
ax[0].set(xlabel='Optimizer update',ylabel='Cross entropy');ax[0].legend()
counts = trainer.routing_total.float().cpu()
image=ax[1].imshow((counts/counts.mean(-1,keepdim=True).clamp_min(1)).numpy(),aspect='auto')
ax[1].set(xlabel='Expert',ylabel='Layer',title='Assignments / mean: current optimizer phase')
fig.colorbar(image,ax=ax[1]);plt.tight_layout();plt.show()
''')
markdown('''
## Продолжение текста

Сначала нужны языковые навыки; SFT/chat — отдельный следующий этап. Generation пересчитывает короткий
prefix, persistent decode cache пока нет. BOS не добавляем: текущий pretraining не обучал BOS в corpus.
Повторения не скрываются repetition penalty при оценке качества модели.
''')
code('''
for prompt in ['Україна — це','The history of science']:
    ids=tokenizer.encode(prompt).ids
    tokens=torch.tensor([ids],device=trainer.device)
    with trainer.autocast():
        generated=trainer.model.generate(tokens,max_new_tokens=48,temperature=.8,top_k=40,
                                         context=SEQ_LEN,valid_vocab=tokenizer.get_vocab_size())
    print(tokenizer.decode(generated[0].tolist()));print()
''')
markdown('''
## Экспорт без дополнительной копии

Для inference используйте уже существующий weights.pt и tokenizer.json. Он не содержит optimizer.
По умолчанию не создаём model.pt, не архивируем полный run с дублированием весов.
Для полноценного exact resume вне текущего runtime скачайте local last.pt отдельно, если нужно.
''')
code('''
RELOAD_FOR_INFERENCE = False
if RELOAD_FOR_INFERENCE:
    del trainer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    payload=torch.load(weights_path,map_location='cpu',weights_only=True,mmap=True)
    restored=NextMoE(ModelConfig(**payload['model_config']))
    dtype=next(iter(payload['model'].values())).dtype
    restored=restored.to(dtype=dtype)
    restored.load_state_dict(payload['model'])
    restored=restored.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    del payload
DOWNLOAD_WEIGHTS = False
if DOWNLOAD_WEIGHTS:
    from google.colab import files
    files.download(str(weights_path))
    files.download(str(RUN_DIR/'tokenizer.json'))
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'next_moe_0p2b_engram_0p1b.ipynb','provenance':[]},
    'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'}},'cells':cells}
for i,cell in enumerate(cells):cell['id']=f'next-moe-{i:03d}'
(ROOT/'notebooks/next_moe_0p2b_engram_0p1b.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print(f'Built notebook: {len(cells)} cells, {len(sources)} source files.')
