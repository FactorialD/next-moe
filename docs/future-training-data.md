# Наступні корпуси UK/EN

Перевірено 12 жовтня 2026 року за офіційними Hugging Face dataset cards/API.
Версії та посилання на закріплені картки: [corpus-candidates.json](../configs/corpus-candidates.json).
Дата зміни репозиторію не означає дату написання документа або свіжість фактів.
Повні корпуси не завантажувалися; якість на нашому прототипі ще не вимірювалася.

## Вихідна точка

Поточний корпус користувача: train UK 6,852,035; train EN 6,330,987;
validation UK 152,327; validation EN 384,544 tokens. Разом train 13,183,022,
validation 536,871. Snapshot Вікіпедії — 2023-11-01.
30M processed tokens при 50/50 відповідають ~2.19 UK / ~2.37 EN обсягів корпусу;
100M — ~7.30 / ~7.90. Random windows не гарантують повного проходу кожного тексту.

## Рекомендовані джерела pretraining

| Джерело | Мови / subset | Новизна джерел | Відбір та використання |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | EN; для свіжих текстів `CC-MAIN-2025-26` та інші snapshots 2024–2025 | Перевірена версія додає January–June 2025 | Освітній classifier score≥3; основа англійської суміші. Це відібрані вебтексти, не синтетично написаний корпус. Оцінки classifier отримані з LLM-анотацій. |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | UK: `ukr_Cyrl` | Common Crawl 2013–April 2024 | Language-specific filtering, глобальна дедуплікація на мову. ~25.59B UK tokens за tokenizer авторів; достатньо для обмеженої вибірки. Не гарантує освітню якість кожного тексту. |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | UK `ukr_Cyrl`, EN `eng_Latn` | Common Crawl `2013-20`–`2025-08` (February 2025) | 69 мов, 350B+ tokens сумарно; top 10% за educational score для кожної мови. Частка для навчальної прози й довших документів. Перевіряти OCR та змішування мов. |
| [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0) | UK `ukr_Cyrl`, EN `eng_Latn` | Internet Archive + Common Crawl 2012–2024; release July 2025 | WDS quality bins, provenance, document/segment language labels. Додаткове різноманіття до FineWeb2. UK ~81.22B tokens за Gemma 3 tokenizer авторів, ~80M documents. |

Числа авторів не дорівнюють кількості токенів нашого BPE. Після відбору рахуємо
точні токени незмінним tokenizer.json. Дата crawl також не гарантує, що сам текст
написано того року. Для свіжості фільтруємо доступні crawl metadata, а не лише
lastModified репозиторію; архівні матеріали залишаємо для мови та сталих знань.

FineWeb/FinePDFs packaging — ODC-BY 1.0 плюс Common Crawl terms. HPLT CC0
стосується упаковки; автори прямо зазначають, що не володіють правами на всі
вихідні тексти. Зберігаємо URL/джерело/ліцензійні відомості, не прирівнюємо
відкритий доступ до єдиної ліцензії всіх документів.

HPLT 3.0 завантажується через language `.map` і JSONL.zst shards за інструкцією
картки; це не звичайний `load_dataset(repo, name=...)` із parquet configs.
Початкова проба: WDS bins 8–10, з аудитом тематичного покриття; цей поріг —
наша гіпотеза, не доведений оптимум для української. Не завантажувати всі shards.

## Рекомендована перша суміш (гіпотеза для експерименту)

Залишити 50/50 UK/EN **за sampled training tokens**. Всередині кожної мови:

| Мова | Джерело | Частка токенів |
|---|---|---:|
| UK | FineWeb2 | 40% |
| UK | HPLT 3.0, відібрані bins | 20% |
| UK | FinePDFs-Edu | 20% |
| UK | Вікіпедія | 20% |
| EN | FineWeb-Edu, переважно snapshots 2024–2025 | 60% |
| EN | FinePDFs-Edu | 20% |
| EN | Вікіпедія | 20% |

Спочатку підготувати 50–100M tokens **на кожну мову**, а не качати терабайти.
Для 200M int32 tokens потрібно ~0.8GB token files; raw text, manifests і
тимчасові файли потребують додаткового місця. Це розмір підготовленого пулу,
не обов'язковий бюджет навчання. Частки відносяться до токенів, не документів.
Наявний sampler уміє лише 50/50 мов: source-aware sampling і нові readers ще
потрібно реалізувати. Цей документ не змінює вже запущене навчання.

Спочатку аудит ~200 випадкових документів на мову/джерело: зв'язність, мова,
boilerplate, повторення, OCR, домени та довжини. Потім exact + near dedup між
усіма джерелами; видалити збіги зі старим validation і окремими benchmark sets.
Контролювати частку доменів, news/SEO, календарних списків та шаблонних сторінок.
Потокове читання + reservoir/shuffle buffer: перші N документів не є
репрезентативною вибіркою. Для довгих документів зберігати повний текст або
ділити по межах абзаців, а не обривати перші 20k символів усіх документів.
Зберігати revision, URL, crawl timestamp, filters, hashes та per-source counts.

Tokenizer і старий validation залишити незмінними; потрібен новий corpus run
із явним переходом даних. Старий validation показує Wikipedia quality,
тому додати **окремий** domain-diverse holdout, не замінюючи старі метрики.
A/B з одного checkpoint і однаковим token budget: Wikipedia baseline проти
нової суміші; порівняти CE на обох holdouts і генерацію зі сталими prompts/seeds.

## Пізніше: навчання чату (SFT)

- [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset): ~204k
  human-annotated prompt/completion pairs загалом, є українська `language_code=ukr`.
  Apache 2.0. Фільтрувати лише train; точну кількість UK прикладів ще треба порахувати.
  Це не Aya Collection з сотнями мільйонів шаблонних/перекладених прикладів.
- [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2): новіша
  англійська SFT-суміш; вибирати короткі non-thinking conversation/instruction
  компоненти для малого прототипу. Нові компоненти Apache 2.0, існуючі мають
  власні ліцензії. Не використовувати всю суміш reasoning/tool traces одразу.
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk): простіший
  старт для EN; ліцензії також перевіряти за компонентами.
- Додати перевірені носіями українські діалоги. Синтетичні або перекладені
  відповіді маркувати окремо; автоматичний переклад сам по собі не гарантує
  природну українську чи правильність фактів.

Для SFT потрібні chat formatting та loss лише на assistant responses;
pretraining на вебтекстах саме по собі не навчає модель бути асистентом.

## Резерви, а не перший вибір

[Malyuk](https://huggingface.co/datasets/lang-uk/malyuk): ~38.94M documents,
суміш UberText 2.0, старого OSCAR та Ukrainian News. Автор називає це неофіційною
компіляцією; відсутня єдина declared license картки, свіжість кожного джерела
не підтверджена. Корисно дослідити відібрані news/fiction компоненти, але не
вважати автоматично кращим за сучасні очищені корпуси.

FineWeb2-HQ у перевіреній картці має 20 мов, **без української**: не підходить
як прямий UK source. Ukrainian-CulturalHeritage-Books містить відібрані видання
до 1884 року: культурний додаток після OCR-аудиту, не сучасні знання.
Rukopys — датасет розпізнавання рукописів, а не основний текстовий pretraining.
Tulu-3-SFT має компоненти з non-commercial умовами; не брати всю суміш без
аналізу компонентів.

## Межі перевірки

Картки/API доступні. Прямий доступ із цієї cloud environment до lang.org.ua,
hplt-project.org і dumps.wikimedia.org заблокований мережевою політикою (403).
Тому не заявляємо перевірений свіжий Wikipedia dump 2026 року або дату збору
UberText за цими сайтами. Для дійсно актуальних фактів 2026 потрібна додаткова
перевірка офіційних dumps/ліцензованих джерел; жоден відібраний корпус не є
підтвердженою повною базою подій 2026 року.
