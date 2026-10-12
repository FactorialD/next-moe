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
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': textwrap.dedent(source).strip() + '\n'})


markdown('''
# next-moe: обучение 0.2B + 0.1B Engram в Google Colab

Самостоятельный ноутбук: исходный код встроен, клонировать или публиковать репозиторий не нужно.
Обучаем **с нуля** самый маленький профиль: 12 × 512, 16 routed experts / top-2 + shared,
GR × 4, один Engram (2/3-граммы, 8 таблиц), GDN/full GQA = 3:1, shared KV пары global слоёв.
Реализация содержит около **193.67M neural + 100.02M table = 293.69M** параметров.

**Runtime → Change runtime type → GPU**. Начните с последовательности 128 и microbatch 1.
T4 использует FP16 autocast + GradScaler, GPU с BF16 — BF16. Все resident weights остаются FP32.
Стандартный AdamW требует около **4.38 GiB только training states**; итоговый peak выше.
GPU Colab не гарантирован; доступность и скорость надо проверить на выделенном устройстве.

По умолчанию: UK/EN Wikipedia `20231101`, 1000 документов на язык, 300 optimizer steps.
Это проверка пайплайна, а не полноценный pretraining, instruction tuning или валидация 32K–256K.
После короткого запуска связные ответы не ожидаются. Wikipedia используется с условиями
[исходного датасета](https://huggingface.co/datasets/wikimedia/wikipedia) и лицензиями статей;
для распространения корпуса/модели сохраняйте происхождение и проверяйте требования лицензий.

Выполняйте ячейки сверху вниз. Режим `tiny` предназначен только для отладки; **по умолчанию full**.
''')
code('''
%pip -q install tokenizers==0.21.4 datasets==3.6.0 matplotlib

import torch, sys, json, gc, time, shutil
from pathlib import Path
print('Python:', sys.version.split()[0], 'Torch:', torch.__version__)
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6), 'Нужен PyTorch >= 2.6; обновите Colab runtime.'
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(props.name, round(props.total_memory / 2**30, 2), 'GiB')
    print('BF16:', torch.cuda.is_bf16_supported())
''')
markdown('''
## Параметры запуска и хранение

`USE_DRIVE=True` сохраняет данные, tokenizer и checkpoints в Google Drive. Потребуется разрешить
подключение своего Drive в Colab. Для полного профиля оставьте **не менее 10 GiB свободного места**:
AdamW checkpoint около 3.28 GiB, при atomic overwrite временно существуют старый и новый файлы,
отдельный export весов занимает около 1.09 GiB. `last.pt` перезаписывается, история не разрастается.

`USE_DRIVE=False` пишет в `/content`: при завершении runtime файлы исчезнут, скачайте их заранее.
Для нового эксперимента используйте новое `RUN_NAME`. Не меняйте tokenizer/data внутри старого run.
При восстановлении `STEPS` означает общее число updates, а не количество дополнительных.
''')
code('''
PROFILE = 'full'               # 'full' = 0.2B+0.1B; 'tiny' = только проверка кода
USE_DRIVE = True
RUN_NAME = 'next-moe-0p2b-engram0p1b-test-v1'
RESUME = False                # True после прерывания runtime
STEPS = 300
SEQ_LEN = 128
MICROBATCH = 1
GRAD_ACCUM = 4
OPTIMIZER = 'adamw'           # 'adamw8bit' — optional, требует bitsandbytes/CUDA
CHECKPOINT_BLOCKS = True
DOCUMENTS_PER_LANGUAGE = 1000
VALIDATION_DOCUMENTS = 32
SOURCE = 'wikipedia'          # либо 'local': отдельные UTF-8 .txt в uk/ и en/
LOCAL_TEXT_DIR = '/content/my_texts'
SEED = 1234

assert PROFILE in ('full', 'tiny')
if PROFILE == 'full' and not torch.cuda.is_available():
    raise RuntimeError('Для полного профиля включите GPU. CPU предназначен только для tiny-теста.')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/next-moe')
else:
    STORAGE = Path('/content/next-moe-artifacts')
RUN_DIR = STORAGE / RUN_NAME
DATA_DIR = RUN_DIR / 'data'
CODE_DIR = Path('/content/next-moe-code')
CODE_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
if not RESUME and (RUN_DIR / 'last.pt').exists():
    raise RuntimeError('Здесь уже есть checkpoint: задайте RESUME=True или новое RUN_NAME.')
if OPTIMIZER == 'adamw8bit':
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes==0.45.5'])
''')
markdown('''
## Встроенная реализация

Чистый PyTorch: no-drop MoE dispatch, chunkwise GDN через triangular solve с FP32 recurrent state,
causal Engram, GR и SDPA GQA. CUDA SDPA выбирает доступный backend автоматически: FlashAttention
не гарантирован на любой GPU/dtype. Triton GDN и fused MoE здесь не реализованы, поэтому throughput
может уступать production kernels. Checkpointing поддерживает gradient через shared K/V.

Генерация ниже пересчитывает короткий prefix; persistent KV/GDN decode API пока не реализован.
Это ограничение тестового runtime, архитектура по-прежнему допускает растущий KV.
''')
sources = {str(path.relative_to(ROOT)): path.read_text() for path in sorted((ROOT / 'next_moe').glob('*.py'))}
source_hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}
code('SOURCES = ' + repr(sources) + '\nSOURCE_HASHES = ' + repr(source_hashes) + '''
for relative, content in SOURCES.items():
    target = CODE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from next_moe.model import ModelConfig, NextMoE, delta_chunk
from next_moe.data import prepare_data
from next_moe.train import TrainConfig, Trainer
if RESUME and (RUN_DIR / 'source_hashes.json').exists():
    previous_hashes = json.loads((RUN_DIR / 'source_hashes.json').read_text())
    if previous_hashes != SOURCE_HASHES:
        raise RuntimeError('Исходники изменились: для strict resume нужен тот же notebook.')
(RUN_DIR / 'source_hashes.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
shutil.copytree(CODE_DIR / 'next_moe', RUN_DIR / 'source', dirs_exist_ok=True)
print('Исходники сохранены:', RUN_DIR / 'source')
''')
markdown('''
## Короткая проверка forward/backward

Используется отдельная tiny-модель, чтобы найти ошибки до выделения памяти полного профиля.
Проверяются causal prefix, finite loss/gradients и backward с checkpointing.
Эта модель удаляется перед настоящим обучением.
''')
code('''
torch.manual_seed(SEED)
check_device = 'cuda' if torch.cuda.is_available() else 'cpu'
small = NextMoE(ModelConfig.tiny()).to(check_device)
probe = torch.randint(4, 512, (1, 9), device=check_device)
small.eval()
with torch.no_grad():
    full_logits = small(probe)
    prefix_logits = small(probe[:, :5])
torch.testing.assert_close(full_logits[:, :5], prefix_logits, atol=2e-4, rtol=2e-3)
small.train()
probe_loss = small(probe[:, :-1], probe[:, 1:])
(probe_loss['ce'] + .01 * probe_loss['aux']).backward()
assert torch.isfinite(probe_loss['ce'])
assert all(torch.isfinite(p.grad).all() for p in small.parameters() if p.grad is not None)
print('Forward/backward и causal prefix: OK; test loss:', float(probe_loss['ce'].detach()))
del small, probe, probe_loss, full_logits, prefix_logits
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
''')
markdown('''
## UK/EN Wikipedia → train/validation → собственный tokenizer

Streaming читает ограниченное число документов из публичного snapshot, не загружает всю Википедию.
Полные документы дедуплицируются и разделяются до обучения tokenizer; validation не участвует
в BPE training. Сэмплирование при тренировке 50/50 по языкам. Byte-level BPE сохраняет і/ї/й/є/ґ,
регистр и пунктуацию. Число реально обученных BPE tokens может быть меньше 65 536; размер embedding
полного профиля всё равно 65 536, чтобы сохранять выбранный размер модели.

Статьи ограничены 20 000 символами; это намеренно маленький тестовый корпус. Статьи объединяются
через EOS, attention/GDN не сбрасываются на границе документа; Engram suffix hashing сбрасывается.
Начало sampled sequence не получает состояние предшествующего фрагмента.
Если источник недоступен, выберите `SOURCE='local'` и положите документы в `LOCAL_TEXT_DIR/uk` и `/en`.
Сетевые ошибки не заменяются скрытым toy corpus.
''')
code('''
model_cfg = ModelConfig() if PROFILE == 'full' else ModelConfig.tiny()
model_cfg.checkpoint_blocks = CHECKPOINT_BLOCKS
if not (DATA_DIR / 'metadata.json').exists():
    if RESUME:
        raise RuntimeError('Для resume нужны исходные tokenizer и token files; не пересоздавайте их.')
    prepare_data(DATA_DIR, source=SOURCE, local_dir=LOCAL_TEXT_DIR,
                 documents_per_language=DOCUMENTS_PER_LANGUAGE,
                 validation_documents=VALIDATION_DOCUMENTS,
                 vocab_size=model_cfg.vocab_size, seed=SEED)
else:
    print('Используем уже сохранённые данные; изменение настроек их не пересоздаёт.')
metadata = json.loads((DATA_DIR / 'metadata.json').read_text())
print(json.dumps({k: metadata[k] for k in ('source', 'source_revision', 'document_counts', 'actual_vocab', 'tokens')}, indent=2))
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(DATA_DIR / 'tokenizer.json'))
for text in ['Україна: і, ї, й, є, ґ; пам’ять.', 'Knowledge and language.']:
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
print('UK/EN tokenizer roundtrip: OK')
''')
markdown('''
## Создание модели и проверка памяти

Веса и градиенты FP32; autocast выполняет вычисления в FP16/BF16. Standard AdamW имеет
16 bytes/parameter постоянных состояний после первого update. AdamW8bit — ориентировочно
10 bytes/parameter при этих FP32 weights/grads, плюс quantization metadata и small FP32 tensors.
На T4 GradScaler уменьшает scale при overflow и повторяет те же микробатчи без учёта пропущенного
optimizer update. Устойчивые non-finite gradients останавливают запуск.

Не начинайте с 32K: это ещё не production GDN/MoE implementation. При OOM попробуйте `SEQ_LEN=64`
или 32, оставьте microbatch 1 и checkpointing; для изменения формы тренировочного run выберите
новое имя. Gradient accumulation увеличивает effective batch без хранения всех микробатчей одновременно.
''')
code('''
train_cfg = TrainConfig(steps=STEPS, sequence_length=SEQ_LEN, microbatch=MICROBATCH,
                        accumulation=GRAD_ACCUM, optimizer=OPTIMIZER, seed=SEED,
                        eval_every=50, save_every=100, log_every=10)
resume_path = RUN_DIR / 'last.pt' if RESUME else None
if RESUME and not resume_path.exists(): raise FileNotFoundError(resume_path)
trainer = Trainer(model_cfg, train_cfg, DATA_DIR, RUN_DIR, resume=resume_path)
parameter_count = sum(p.numel() for p in trainer.model.parameters())
print('Training state lower bound, GiB:', parameter_count * (16 if OPTIMIZER == 'adamw' else 10) / 2**30)
print('Tokens / optimizer step:', SEQ_LEN * MICROBATCH * GRAD_ACCUM)
print('Planned new tokens:', max(STEPS - trainer.step, 0) * SEQ_LEN * MICROBATCH * GRAD_ACCUM)
print('Initial validation:', trainer.evaluate())
''')
markdown('''
## Обучение

Метрики: training CE, UK/EN validation CE/perplexity, router load, gradient norm, tokens/s и peak VRAM.
Throughput включает overhead между логами, включая evaluation/checkpoint saves; это не isolated kernel benchmark.
В начале CUDA memory растёт из-за lazy allocation Adam states. Не путайте initial allocation с steady-state peak.

`last.pt` сохраняется каждые 100 updates и в конце; Ctrl+C пытается сохранить последний завершённый update.
При жёстком отключении runtime сохранится только последний успешно записанный checkpoint.
После такого отключения запустите ноутбук заново с тем же RUN_NAME и `RESUME=True`.
При увеличении STEPS меняется будущая cosine LR schedule; прежняя история обучения сохраняется.
''')
code('''
start = time.perf_counter()
final_metrics = trainer.run()
print('Elapsed minutes:', round((time.perf_counter() - start) / 60, 2))
print('Final validation:', final_metrics)
weights_path = trainer.export_weights()
print('Weights:', weights_path)
# Record exact runtime package versions for this run.
import subprocess
(RUN_DIR / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
''')
markdown('''
## Графики и загрузка экспертов

Validation curve важнее training loss для обнаружения переобучения. Баланс router не доказывает
качество экспертов. Метрика unselected относится к одному update, а не ко всему обучению.
''')
code('''
import matplotlib.pyplot as plt
records = [json.loads(line) for line in (RUN_DIR / 'metrics.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([r['step'] for r in records], [r['train_ce'] for r in records], label='train')
for lang in ['uk', 'en']:
    validation = [r for r in records if f'val_ce_{lang}' in r]
    ax[0].plot([r['step'] for r in validation], [r[f'val_ce_{lang}'] for r in validation], label=f'val {lang}')
ax[0].set(xlabel='Optimizer step', ylabel='Cross entropy'); ax[0].legend()
counts = trainer.last_counts
if counts is not None:
    image = ax[1].imshow(counts.float().cpu().numpy(), aspect='auto')
    ax[1].set(xlabel='Routed expert', ylabel='Layer', title='Assignments: last update')
    fig.colorbar(image, ax=ax[1])
plt.tight_layout(); plt.show()
''')
markdown('''
## Тестовая генерация

Это обычное продолжение текста, не чат-модель. Тестовый runtime пересчитывает prefix на каждом токене
и оставляет последние SEQ_LEN tokens; это не реализация long-context inference cache.
После 300 updates случайный или бессвязный результат нормален: главное — проверить корректность
обучения, finite gradients и изменение loss. Качество языка и знаний требует гораздо большего корпуса
и вычислений, а instruct поведение — отдельного этапа обучения.
''')
code('''
for prompt in ['Україна — це', 'The history of science']:
    ids = [tokenizer.token_to_id('<bos>')] + tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], device=trainer.device)
    with trainer.autocast():
        generated = trainer.model.generate(tokens, max_new_tokens=48, temperature=.8, top_k=40,
                                           context=SEQ_LEN, valid_vocab=tokenizer.get_vocab_size())
    print(tokenizer.decode(generated[0].tolist()))
    print()
''')
markdown('''
## Экспорт и восстановление для inference

В Drive уже лежат `model.pt`, `tokenizer.json`, `model_config.json`, `last.pt`, метрики и исходники.
`model.pt` нужен для inference; `last.pt` дополнительно содержит optimizer/scaler/RNG для продолжения.
На восстановление нужен тот же tokenizer; не retrain BPE для существующих Engram rows.
Загружайте только свои доверенные checkpoints. Пример отдельной загрузки ниже выключен по умолчанию,
чтобы не создавать вторую копию большой модели на GPU.
''')
code('''
RELOAD_FOR_INFERENCE = False
if RELOAD_FOR_INFERENCE:
    del trainer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    payload = torch.load(RUN_DIR / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    restored = NextMoE(ModelConfig(**payload['model_config']))
    restored.load_state_dict(payload['model'])
    restored = restored.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    del payload
    print('Restored model:', sum(p.numel() for p in restored.parameters()))

# Для USE_DRIVE=False можно скачать веса вручную; большой файл может занять время.
DOWNLOAD_WEIGHTS = False
if DOWNLOAD_WEIGHTS:
    from google.colab import files
    files.download(str(RUN_DIR / 'model.pt'))
    files.download(str(RUN_DIR / 'tokenizer.json'))
''')
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'colab': {'name': 'next_moe_0p2b_engram_0p1b.ipynb',
      'provenance': []}, 'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3',
      'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'cells': cells}
for i, cell in enumerate(nb['cells']):
    cell['id'] = f'next-moe-{i:03d}'
(ROOT / 'notebooks' / 'next_moe_0p2b_engram_0p1b.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f'Built standalone Colab notebook: {len(cells)} cells, {len(sources)} embedded source files.')
