# Ще10M після checkpoint≈55M

[Відкрити у Colab](https://colab.research.google.com/github/FactorialD/next-moe/blob/work/notebooks/next_moe_55m_plus10m_replay.ipynb).

Та сама схема, що попередні+5M:25%Wikipedia/75%web–PDF, мови50/50,
seq_len512, microbatch4, accumulation1, BF16 weights/gradients наL4,
FP32 master Adam, LRpeak5e-5, warm ramp100 та phase-token cosine до5e-6.
Архітектура/model_config із checkpoint. Tokenizer незмінний.

## Запуск

- BASE_RUN_NAME='' автоматично знаходить frozen checkpoint53–57M із
  weights.pt і data-replay/, інакше використовує next-moe-50m-plus5m-replay-v1.
  Можна явно вказати каталог вашого архіву. Base SHA фіксується: не змінювати
  попередні ваги після початку нового run.
- Новий RUN_NAME=next-moe-55m-plus10m-replay-v1. Старий run не перезаписується.
- ADDITIONAL_TOKENS=10_000_000; target=actual starting tokens+10M (приблизно65M).
- SESSION_UPDATES=None — весь бюджет. Для графіків посеред навчання поставте500:
  training cell → graph cell → знову training cell. Не пересоздавати trainer.
- Бенчмарк доступний, вимкнений за замовчуванням, щоб не повторювати замірL4.
  COMPUTE_UNITS_PER_HOUR=1.54 — попередній тариф користувача, перевірити актуальний.

## Дані, валідація та прогрес

Готовий filtered replay pool копіюється з base data-replay/ у новий run та local
cache. Немає повторного download FineWeb/FinePDFs/Wikipedia, потреби у початковому
data-v2/ чи повторної фільтрації. Metadata/source manifests/token counts і SHA
усіх файлів перевіряються. Fraction має збігатися з попереднім recipe.

Baseline55M фіксується перед навчанням. Графічна ячейка показує train/old/new
validation CE, phase expert heatmap, training/interval tokens/s і deltaCE
відносно baseline55M. Обидва validation token files залишаються незмінними.
Окрема генерація зі сталими prompts/seed. Progress цього run окремий від55M.

З Drive weights — warm restart із новим Adam; local last.pt — exact resume.
Phase metadata зберігає ціль, тому повторне відкриття не додає ще10M. Budget
може перевищитися максимум на tokens/update-1 через округлення останнього update.

## Місце та орієнтовна квота

Новий replay pool≈210MB, compact weights≈0.55GiB; при save потрібен додатковий
temporary snapshot≈0.55GiB. Загальна квота Drive включає всі старі архіви.
Local full optimizer checkpoint≈3.83GiB, автоматично наDrive не копіюється.
Зберігаються weights.pt, tokenizer.json, data-replay/, phase.json, baseline,
configs/source_hashes/metrics/pip-freeze. Старий checkpoint55M залишається.

При спостережуваних≈1960tokens/s10M займають≈85хвилин training або≈2.18units
за тарифу1.54/год; validation/save/preparation додають overhead. Це оцінка,
не гарантія Colab. Цей notebook — продовження pretraining, не SFT/чат.

## Перевірка

CPU offline tests виконують core notebook cells наtiny: loading55M-labelled
fixture checkpoint, exact corpus copy, replay sampler, обидві validation,
training, graphs/generation, exact reopen. Перевірено auto discovery і для
робочого run, і для frozen архіву. Full model/L4 speed перевіряється вColab.
