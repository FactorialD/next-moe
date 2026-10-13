# Checkpoint50M: короткий етап із Wikipedia replay

[Відкрити у Colab](https://colab.research.google.com/github/FactorialD/next-moe/blob/work/notebooks/next_moe_50m_wikipedia_replay.ipynb).

За замовчуванням: checkpoint **50,001,920 tokens → ще5M**, приблизно55M загалом.
Це короткий перевірочний етап; відновлення Wikipedia quality не гарантоване.
Learning rate5e-5,100-update warm ramp, cosine за tokens нової фази до5e-6.
Після weights-only load optimizer новий. Розмір/Engram/tokenizer незмінні.

## Що запускати

- L4 runtime; BASE_RUN_NAME=`next-moe-30m-plus20m-webpdf-v1`, data-stage3/.
- Новий RUN_NAME=`next-moe-50m-plus5m-replay-v1`: base50M не перезаписується.
- Wikipedia train files беруться з data-v2/ у checkpoint-30m-frozen або
  next-moe-0p2b-engram0p1b-10m-v2. Для іншого шляху задайте WIKI_DATA_PATH.
- Датасети повторно з мережі не завантажуються. Потрібні наявні original token
  files, metadata, tokenizer; жоден tokenizer не перевчається.
- RUN_EFFICIENCY_BENCHMARK=False: L4 уже виміряний. True запускає попередній
  reversible benchmark. Ціна1.54units/hour — останнє показання користувача,
  не гарантований тариф; оновлюйте за актуальною панеллю Colab.
- SESSION_UPDATES=None — весь новий бюджет. Для перегляду графіків посеред
  навчання SESSION_UPDATES=500. Після кожних500 updates запускайте графіки,
  потім знову training cell; не повторюйте створення trainer.

## Як змішуються тексти

Sampler для кожної train sequence спочатку обирає UK/EN50/50, потім Wikipedia
з probability25%, інакше попередній web/PDF pool. Sequence length512 однакова:
очікувані частки tokens теж25/75. Оскільки попередній pool≈80/20web/PDF,
загалом маємо≈25%Wikipedia,60%web,15%PDF. Частки стохастичні; sampling із
повтореннями. На додаткових5M очікувано1.25M Wikipedia і3.75M web/PDF tokens.
WIKIPEDIA_FRACTION=.25 змінюється лише для нового RUN_NAME.

Wikipedia replay очищається від exact/approximate збігів із обома holdouts:
перевіряються SHA та SimHash decoded документів. Кількість вилучених documents
і остаточні Wikipedia tokens записуються у metadata. Near dedup евристичний,
не абсолютна гарантія відсутності пов'язаних текстів. EOS boundaries збережені.
Оригінальні дані не змінюються; filtered replay pool записується окремо.

Стара й нова validation копіюються byte-identical з50M run. Перед навчанням
baseline-validation.json фіксує CE/PPL50M на обох holdout. Графічна ячейка:
train loss, old/new validation, router heatmap, training/interval tokens/s;
також таблиця deltaCE проти baseline. Від'ємна delta означає поліпшення.
Мета — old UK CE нижче4.399 без значного погіршення new UK/EN; окремо перевіряємо
генерацію зі сталими prompts/seed. Якщо обидва holdout погіршуються, зупиняємо
довге продовження й досліджуємо дані/recipe.

## Збереження і відновлення

data-replay/ містить original web/PDF train та обидва holdout, filtered
train_wikipedia_uk.bin/train_wikipedia_en.bin, BPE, source manifests, original
Wikipedia metadata, recipe/fingerprints. Exact resume signature включає hashes
replay pools через metadata. Усі source draws використовують checkpointed RNG.
Sampler обов'язково встановлюється після load; notebook робить це автоматично.
Local last.pt — точне відновлення Adam/RNG, Drive weights.pt — warm restart.
phase.json зберігає starting tokens, fixed target, fraction і baseSHA; повторне
відкриття не додає ще5M. Base checkpoint має залишатися незмінним.

Новий corpus cache приблизно210MB до overlap filtering, weights≈0.55GiB,
плюс temporary weights≈0.55GiB при save. Перевірте вільну загальну квоту Drive:
вона включає старі run. Автоматичної копії full3.83GiB optimizer checkpoint
на Drive немає. Дані й weights копіюються через temporary paths; неповний cache
не приймається за готовий. Source notebook автономний, вихідний код вбудований.

## Перевірка

Offline CPU tests: створення replay corpus, unchanged tokenizer/holdouts,
recipe-cache validation, checkpointed sampling RNG, реальне training наtiny,
виконання notebook cells включно із graphs/generation, exact reopen.
L4 throughput і якість повної моделі на цій фазі потребують запуску користувача.
