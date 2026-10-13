# next-moe

Исследовательский проект LLM: sparse MoE + hybrid attention + Engram-память + Gated Residual.

Подготовлен проект архитектуры и обучаемый PyTorch-прототип самого маленького профиля. Полноценное обучение и оценка качества ещё не проведены.

- [Самостоятельный Colab notebook: обучение 0.2B + 0.1B Engram](notebooks/next_moe_0p2b_engram_0p1b.ipynb).
- [Новий Colab: checkpoint 30M → ще 20M на FineWeb/FinePDFs, benchmark L4](docs/continuation-colab.md).
- [Инструкция запуска, сохранения и восстановления](docs/colab-training.md).
- [Результаты 300 шагов полного профиля в облачной среде на CPU](docs/cloud-validation.md).
- [Пользовательский Colab run на T4: выводы и следующий бюджет](docs/colab-results-and-next-steps.md).
- [Обучающий корпус: точный источник и количество](docs/training-corpus.md).
- [Наступні якісні корпуси UK/EN: джерела й рекомендована суміш](docs/future-training-data.md).
- [Этапы превращения base-модели в чат](docs/chat-training.md).
- [Память и скорость обучения](docs/training-memory.md).
- [Экономия квоты Colab: бенчмарк текущего trainer](docs/colab-efficiency.md).

- [Архитектура и оценки памяти](docs/architecture.md).
- [DeepSeek V4.1: исследование и изменения архитектуры v0.3](docs/deepseek-v41-review.md).
- [Десять размеров: от 0.2B + 0.1B до 1T + 200B](docs/model-family.md).
- [Современные модели разных размеров: источники и сравнение](docs/model-size-references.md).
- [Gated Residual: схема, источники и стоимость](docs/residual-design.md).
- [Проверенные источники](docs/sources.md).
- [Сравнение n-gram моделей и решение по второму модулю](docs/ngram-placement-review.md).
- [План исследований и интерактивного конструктора](docs/roadmap.md).
- [Десять конфигураций и аналитические оценки](configs/model-family.json).
- [Версии и конфигурации моделей-референсов](configs/reference-models.json).
- [Прежние два иллюстративных примера с двумя модулями памяти](configs/proposed-profiles.json).

Основные требования: украинский/английский, контекст до 256K, GDN + редкие full-attention слои, четырёхветвевой GR, прямое редактирование Engram-векторов. Размерная сетка предложена для обсуждения; конечный размер и оборудование на этом этапе не выбираются. Neural, active и n-gram параметры считаются отдельно.

Текущий Colab workflow v2 планирует следующий эксперимент на 10M processed tokens сессиями по 1000 updates. На T4 используются FP16 параметры/градиенты модели и FP32 master optimizer. Drive хранит один компактный weights snapshot; полный optimizer checkpoint остаётся локально. После потери runtime — явный warm restart. Перед длинной сессией проверить новый режим на 20–50 updates; user T4 run проверял прежний FP32+autocast recipe.

Базовое размещение: один ранний Engram-модуль. В v0.3 пара full GQA слоёв читает общий KV source; GDN/full остаётся 3:1. Кеш хранит полную causal историю и растёт с контекстом, фиксированный общий размер не требуется. Экономия памяти аналитическая; качество cross-layer sharing предстоит проверить обучением.
