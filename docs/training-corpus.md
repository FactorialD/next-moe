# Обучающий корпус первого Colab-эксперимента

Дата: 12 октября 2026. Пользователь выбрал **украинскую и английскую Википедию**. Пайплайн использует готовый открытый датасет и запускается в [Colab notebook](../notebooks/next_moe_0p2b_engram_0p1b.ipynb); ручной scraping Wikipedia не выполняется.

## Точный источник

- Dataset repository: **[wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)**.
- Украинская конфигурация: **`20231101.uk`**.
- Английская конфигурация: **`20231101.en`**.
- Snapshot Wikipedia: **1 ноября 2023**, а не текущие версии статей.
- Dataset revision: **`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`**; [закреплённая карточка](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md). SHA и карточка проверены через Hugging Face API 12 октября 2026.
- Source split: **`train`** в обоих configs; собственный validation split строится из отдельных документов.
- Загрузка: `datasets.load_dataset(..., revision=..., split='train', streaming=True)`; читаются готовые parquet-данные с полями `id`, `url`, `title`, `text`.

Карточка источника указывает **CC BY-SA 3.0 и GFDL**. Происхождение статей сохраняется в `documents.json`; при перераспространении проверять применимые требования лицензий датасета и исходных статей. Сохранённая ссылка/хеш фиксирует происхождение, но сама по себе не заменяет выполнение условий лицензии.

## Количество первого теста (notebook v1)

| Язык | Отобранные документы | Train | Validation |
| --- | ---: | ---: | ---: |
| Украинский | 1000 | 968 | 32 |
| Английский | 1000 | 968 | 32 |
| Всего | **2000** | **1936** | **64** |

Это настройки первого теста v1; пользователь сообщил успешный Colab run, но его corpus metadata не приложена, поэтому фактические counts надо читать из metadata того run. Отбираются первые 1000 подходящих уникальных документов каждого config в порядке streaming; не равномерная случайная выборка всей Википедии. Число просмотренных исходных документов может быть больше 1000 из-за фильтрации. После отбора документы переставляются с seed **1234** отдельно для каждого языка; первые 32 уходят в validation, остальные — в train. Train/validation не имеют одинаковых SHA256 используемого текста, в том числе между языками.

Если источник исчерпан раньше, код допускает меньше выбранных документов при наличии минимум 33; фактические counts записываются и имеют приоритет перед этой таблицей. При недостатке документов для обоих splits запуск останавливается. Количество можно менять через `DOCUMENTS_PER_LANGUAGE` и `VALIDATION_DOCUMENTS` в notebook.

## Обработка и токенизация

1. Читать поле `text`, удалять только внешние пробельные символы (`strip`).
2. Пропускать документы короче **200 символов** после strip.
3. Брать максимум **20 000 Unicode-символов** каждого документа, не bytes. Усечение может оборвать предложение.
4. Удалять дубликаты по SHA256 именно используемого, уже ограниченного текста; сохранять также хеш полного stripped текста.
5. Делить на train/validation по целым документам до обучения tokenizer.
6. Обучать собственный byte-level BPE **только на train UK/EN**, целевой vocabulary **65 536**, `min_frequency=2`, специальные tokens `<pad>/<bos>/<eos>/<unk>` с IDs 0/1/2/3. Для `tiny` целевой vocabulary 512.
7. Сохранять регистр, украинские і/ї/й/є/ґ и пунктуацию; дополнительной Unicode normalization, strip accents или case folding нет. Число реально обученных tokens может быть ниже целевого.
8. Токенизировать отдельно документы и добавлять EOS; сохранять четыре int32 token files: `train_uk.bin`, `train_en.bin`, `val_uk.bin`, `val_en.bin`.

Верхняя граница текста после усечения — **40 млн символов суммарно**, из них 38.72 млн train и 1.28 млн validation при полном отборе. Это предел, не заявленный фактический размер: большинство статей короче. Точное число tokens зависит от выбранных статей и выученного BPE; оно вычисляется в Colab и не подменяется прогнозом.

## Что сохраняет каждый запуск

В `RUN_DIR/data/metadata.json` записываются dataset/config/revision, requested/actual document counts, число просмотренных документов, пороги длины, seed, train/validation hashes, requested/actual tokenizer vocabulary, special IDs и **точное число tokens в каждом из четырёх файлов**. `documents.json` содержит `id`, `url`, `title`, хеш используемого и полного stripped текста, фактические использованные символы для каждого документа с привязкой к языку/split. Tokenizer сохраняется в `tokenizer.json`.

Trainer дополнительно сохраняет SHA256 corpus/tokenizer signature в checkpoint и проверяет его при resume. Не переобучать tokenizer и не пересоздавать таблицы/корпус внутри существующего run: Engram адреса зависят от неизменных token IDs.

## Что реально увидит модель

Сэмплирование: **50/50 по языкам в ожидании** на каждую последовательность, независимо от размеров token files. Для каждого микробатча выбирается случайное contiguous окно языка; это не последовательный epoch-pass по всему корпусу. Между статьями стоит EOS, но GDN/attention не сбрасываются на document boundary. Engram hashing suffix сбрасывается на EOS; в начале окна предшествующая история не передаётся.

Стартовые настройки: 300 optimizer updates, sequence length 128, microbatch 1, gradient accumulation 4. Обработано **153 600 training tokens** за такой запуск, примерно 76 800 на каждый язык в ожидании. Это tokens с возможными повторениями, **не 153 600 уникальных tokens и не весь собранный корпус**. Число actual processed tokens и completed updates пишется в checkpoint/метрики.

Небольшой корпус и короткая тренировка предназначены для проверки численной устойчивости, routing, loss и сохранения. Они не достаточны для качественного pretraining модели на 294M параметров или оценки знаний/контекста 256K.

## Текущий notebook v2

После пользовательского run на T4 следующий эксперимент планируется до **10M processed tokens**, сессиями по1000 updates. Corpus caps: **10000 документов и30M Unicode characters на язык**, max20000/article. Fresh split имеет128 validation documents/lang (при полном отборе9872 train/lang); при продолжении старого checkpoint сохраняются старые validation hashes и tokenizer, так что обычно остаётся32 val/lang. Caps не означают фактический объём: character limit или исчерпание source могут уменьшить document counts.

BPE при продолжении **не retrain**. Правила corpus expansion сохраняют old validation порядок/hash, не допускают эти документы в train и копируют точные tokenizer bytes. Token files записываются документами, затем кешируются локально в Colab. Измерять степень повторения как ожидаемые sampled tokens/lang, делённые на actual train token count; чтение random windows не является полным epoch-pass. [Workflow v2](colab-training.md), [выбор следующих шагов](colab-results-and-next-steps.md).

## Наступний етап після 30M

[Окремий notebook і точні джерела/кількість](continuation-colab.md): ~20M tokens/lang
із FineWeb2 UK, FineWeb-Edu EN2025 та FinePDFs-Edu UK/EN; 80/20 web/PDF.
BPE та старий validation незмінні. HPLT не входить до цього recipe.
