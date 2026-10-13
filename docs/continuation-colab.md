# Colab: checkpoint 30M → ще 20M нових текстів

[Відкрити notebook у Colab](https://colab.research.google.com/github/FactorialD/next-moe/blob/work/notebooks/next_moe_30m_to_50m_webpdf.ipynb).
[Файл notebook](../notebooks/next_moe_30m_to_50m_webpdf.ipynb).

## Запуск на L4

1. Runtime → Change runtime type → L4. Виконувати ячейки по порядку.
2. `BASE_RUN_NAME='checkpoint-30m-frozen'`, `BASE_DATA_NAME='data-v2'` — архів
   із weights.pt та старими data-v2/tokenizer.json, metadata.json і val_*.bin.
   Якщо архіву немає, вкажіть каталог робочого checkpoint 30M; після початку
   цього етапу не змінюйте його ваги. Базова SHA перевіряється при відкритті.
3. Новий RUN_NAME — `next-moe-30m-plus20m-webpdf-v1`; старий run не перезаписується.
4. У COMPUTE_UNITS_PER_HOUR введіть актуальний тариф L4 з панелі Colab.
   None означає speed-only benchmark без вигаданого тарифу.
5. SESSION_UPDATES=None — навчання до кінця додаткових 20M. Для навмисної
   паузи після 1000 updates поставте 1000 і повторюйте лише training cell.

L4 підтримує native BF16; auto_master обирає BF16 model weights/gradients,
FP32 master weights та Adam moments. Архітектура/Engram/model_config беруться
з checkpoint. Corpus tokenizer bytes та token IDs не змінюються.

## Джерела та кількість

| Мова | Джерело | Config | Revision |
|---|---|---|---|
| UK | HuggingFaceFW/fineweb-2 | ukr_Cyrl | af9c13333eb981300149d5ca60a8e9d659b276b9 |
| EN | HuggingFaceFW/fineweb-edu | CC-MAIN-2025-26 | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 |
| UK | HuggingFaceFW/finepdfs-edu | ukr_Cyrl | 9cfabe2127faca99b3d5c4dc6d1fcb397399ebde |
| EN | HuggingFaceFW/finepdfs-edu | eng_Latn | 9cfabe2127faca99b3d5c4dc6d1fcb397399ebde |

Картки та configs на цих revisions перевірені через HF 13 жовтня 2026.
[Порівняння джерел, dates та ліцензії](future-training-data.md).
Кожна мова: **20M train tokens** незмінним tokenizer, приблизно **16M web + 4M PDF**;
останній цілий документ може трохи перевищити quota. Обидві мови:
~40M training tokens pool, **~160MB int32 train files**, плюс manifests/holdouts.
Це обсяг підготовлених даних, не processed tokens: навчання додатково 20M.
Очікувано буде sampled ~10M tokens/lang, тобто ~половина пулу кожної мови.
Старі Wikipedia train texts у цю суміш не додаються; HPLT тут не завантажується.

Streaming + shuffle buffer256, seed1234; не терабайтний повний download.
80/20 регулюється при побудові token pool за counts, не за document counts.
Trainer далі вибирає випадкові windows 50/50 між мовами. Source proportions у
sampled training будуть близькі до pool proportions, не точно однакові що update.
Документи максимум20k characters; це bounded-cost recipe, не long-context training.

Фільтри: мінімум200 characters, >=35% alphabetic, repeated-line filter,
>=80% цільової мови серед відомих PDF page-language labels (якщо вони є),
глобальні exact hashes і approximate SimHash<=3. Класифікація мови джерел
успадковується від авторів. Near dedup є евристикою, не гарантією відсутності
всіх перетинів чи benchmark contamination. OCR і classifier errors можливі.
Не заявляємо незалежно перевірену якість усіх documents.

Data manifest `documents.jsonl`: language/split/source/dataset/config/revision,
URL/id, доступна crawl date, hashes, token count та використані characters.
`metadata.json`: exact counts для train/old validation/new validation, counts
за source, scanned/rejected, dedup settings, tokenizer signature і file hashes.
Packaging цих джерел ODC-BY1.0; зберігати provenance і враховувати terms джерел.

## Валідація, продовження, графіки

Старі val_uk.bin/val_en.bin копіюються **byte-identical**, старі тексти блокуються
за hashes і за decoded SimHash. New-domain holdout:128 documents/lang, приблизно
102 web+26 PDF. Обидві оцінки використовують16 windows/lang, validation microbatch1,
фіксовані seeds та seq_len512. Старі CE можна порівнювати зі старою validation,
new CE — лише в межах нового holdout. Не змішувати їх в одну метрику.

Графіки: train CE, old/new validation CE, phase router assignments/mean,
training-step tokens/s та interval throughput із overhead. Final validation
також додається до графіка, якщо останній step не кратний eval interval.
Generation — ті самі UK/EN prompts із fixed seed, без приховування повторень.

Нова фаза починається з weights-only snapshot, **Adam/RNG reset**, warm ramp100,
LR peak1e-4 (гіпотеза), phase-token cosine до1e-5. LR schedule відраховує нові
20M tokens, а не historical steps. phase.json фіксує base step/tokens, SHA та
phase target. Після відкриття target не збільшується ще на20M. Фактичний фінал
близько50M, перевищення до tokens/update-1 допустиме.

Auto resume: local last.pt — exact з Adam і попереднім training config;
інакше новий Drive weights.pt — warm на залишок того самого budget. У точному
resume microbatch/precision беруться з checkpoint; seq_len/code мають збігатися.
Для явно іншого recipe RESUME_MODE=warm. Не викликати extend_token_budget що сесію.

## Бенчмарк і сховище

Зберігає effective batch2048 tokens/update, порівнює microbatch1/2/4 і accumulation;
якщо block checkpointing увімкнений, тестує вимкнення. Default config успадковує
checkpoint 30M (зазвичай checkpoint_blocks=False). Тестові updates відкочуються
разом із model/masters/Adam/RNG/routing/progress. Fastest layout застосовується.
Новий Adam спершу отримує один реальний update. Benchmark потребує host RAM для
~3.83GiB backup та transient optimizer copy; при малому host RAM оберіть high-RAM
runtime або RUN_EFFICIENCY_BENCHMARK=False. Це вимірювання speed, не proof якості.

Порівняння G4:6420 tokens/s при8.9units/hour≈2.60M tokens/unit.
L4 виграє, якщо speed/price>721.3. Ціна L4 невідома до показань користувача.
Оцінки часу/units без eval/save overhead; final practical throughput може бути нижчим.

Drive: weights.pt≈0.55GiB, tokenizer.json, data-stage3/, phase.json, configs,
source_hashes.json, metrics.jsonl, pip-freeze.txt. Local full last.pt≈3.83GiB,
не копіюється на Drive автоматично. Архів30M залишається окремо. Run limit4.5GiB
не дорівнює quota всього Drive; перед запуском перевірити вільне місце.
Корпус спочатку збирається local, потім копіюється до Drive temporary directory
і атомарно перейменовується; неповний directory не використовується як готовий.
Збереження weights теж atomic temporary+replace. Неповні corpus copies після
переривання треба прибрати або використати інший run; готові cached data перевіряються.

## Перевірка в нашій cloud environment

CPU offline tests виконують справжні core notebook cells на tiny architecture:
checkpoint load, новий корпус із fixture records, незмінні BPE/old val, training,
обидва evaluations, benchmark helpers, plots/generation, exact reopen.
Fixtures — лише тестове джерело; user notebook не підміняє HF corpus fixture data.
Full model/L4 throughput і реальну parquet ingestion тут не перевірено: GPU
відсутній, а HF CDN downloads раніше блокувалися policy. Pinned cards/configs
перевірені онлайн; streaming paths відповідають інструкціям авторів.
