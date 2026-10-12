# План next-moe

## Первый запуск обучения: Colab

После пользовательского [T4 run на 300 updates](colab-results-and-next-steps.md) подготовлен workflow v2: stage до10M tokens, сессии, FP16 model/gradients с FP32 masters, compact Drive snapshots с явным warm restart, local exact checkpoints и cache token files. Corpus expansion сохраняет tokenizer/validation. Следующая проверка — 20–50 CUDA updates нового recipe, затем progress около1M tokens; GPU режим v2 пока не измерен. Дальнейшее полноценное pretraining зависит от качества, данных и kernel throughput.

По запросу пользователя 12 октября подготовлены [самостоятельный notebook](../notebooks/next_moe_0p2b_engram_0p1b.ipynb) и [инструкция](colab-training.md) для полного малого профиля 0.2B + 0.1B Engram. [Открытый UK/EN Wikipedia corpus](training-corpus.md) закреплён по snapshot/config/revision; исходный test budget v1 — 2000 документов и 153 600 processed training tokens. Начать с короткой последовательности 128 и проверки loss, routing, VRAM, сохранения и resume. Это технический эксперимент, не подтверждение качества языковой модели. Дальнейшие этапы pretraining → SFT → оценка chat → optional DPO записаны в [chat-training.md](chat-training.md); SFT пока не реализован.

## Сейчас: определить и проверить архитектуру

1. Зафиксированы требования: украинский/английский, цель 256K с этапами 32K/128K, GDN + редкие full-attention слои. Кеш должен быть меньше all-full варианта, но может линейно расти с контекстом; фиксированный размер не требуется. По новому запросу предложена [сетка десяти размеров](model-family.md), без выбора конечной модели или оборудования; размеры остаются параметрами будущего конструктора. Уточнить задачи/данные и обучение с нуля либо модификацию готового backbone, когда это станет нужно для экспериментов.
2. Уточнены референсы: Gemma 4 26B-A4B-it, GPT-OSS-20B, Qwen3.8-Flash-Next (preview архитектуры Qwen4) и [DeepSeek-V4.1-Flash](deepseek-v41-review.md). Последний добавлен по запросу пользователя 11 октября; card/config/reference code проверены, PDF report остаётся недоступным через сеть среды. Выбрано прямое управление Engram-векторами. По просьбе пользователя детали edit API, адресации, aliases и получения новых векторов отложены; не подменять Engram внешней базой фактов.
3. Зафиксировать конфигурацию прототипа, tokenizer, causal block order, routing policy, numerical precision, память и ablation-план. Текущий проект v0.3: [architecture.md](architecture.md). Включён [GR](residual-design.md): четыре residual-ветви, low-rank read gates и scalar write gates, отдельные operators для attention/MoE; финальное read-only сведение. Пары full GQA слоёв читают один KV source; его receiver не имеет своих K/V projections.
4. Реализовать небольшой CPU-correct reference: MoE, GDN/GQA с explicit cache ownership, GR и causal Engram с branch-wise fusion. Проверить equivalence prefill/decode, отсутствие future-token leakage, gradient receiver→KV source, единственное append/cache allocation на source, chunked prefill, branch normalization, final collapse и lookup. Сверять GR/GDN с точным Flash-Next report/implementation; не добавлять второй pre-norm после GR read. Не заявлять CED skip-prefill при несформированных decoder GDN states.
5. Выполнить небольшое контролируемое pretraining исследование; обучить baselines на тех же данных. Новые компоненты тестировать при одинаковых параметрах и отдельно при одинаковом compute budget. Установить seeds и бюджет токенов; не использовать крошечную модель как доказательство качества большой.

   Для n-gram placement стартовать с одного раннего модуля; отдельно сравнить два с разделением того же table budget, shared-table repeated read и matched-capacity варианты. [Источники и протокол](ngram-placement-review.md). Второй модуль не включать по умолчанию только на основании малого улучшения loss в другом backbone.

   Для KV-sharing сравнить group=1/2 при одинаковых данных и отдельно matched parameter/compute budgets; измерить UK/EN качество и recall на 32K/128K/256K. По V4.1 дополнительно сравнить memory orders 2/3 против 2/3/4 и conv on/off; tokenizer compression не должен объединять ї/і или й/и. Routing selection-only bias и score function исследовать как разные изменения.
6. После стабилизации backbone вернуться к деталям редактирования таблиц и его проверке: изменение ассоциаций, парафразы UK/EN, locality, коллизии и откат. На текущем этапе эта задача отложена, не реализуется.
7. Измерить VRAM/RAM, prefill/decode, latency и throughput на выбранном для эксперимента оборудовании. Сравнить таблицы на GPU с RAM+prefetch, проверить отсутствие неучтённой репликации shared KV и измерить GR/live-state traffic. Сопоставить GR с plain pre-norm, GatedNorm и static widened residual; Single-Pass mHC V4.1 и Block AttnRes — отдельные альтернативы. После reference оценивать fused/FP8 GR kernels, expert offload, FP4 packed KV, CSA2/QSA indexers и DSpark/MTP. Основной attention путь остаётся GDN+full с pairwise KV-sharing, residual — GR. CED вынести в отдельный проектный variant с полным accounting encoder boundary/replay/recurrent states.
8. Выбрать окончательный размер и путь обучения большой модели. Сохранить ограничения и неудачные эксперименты вместе с результатами.

## Следующий этап: интерактивный конструктор

Разработка записана в план; интерфейс пока не реализуется. Начать после стабилизации формул и схемы конфигурации, чтобы пользователь менял реальные параметры модели.

Начальные presets: десять профилей из [model-family.json](../configs/model-family.json), от 0.2B neural + 0.1B n-gram до 1T + 200B. Импортировать их полную параметризацию, включая разные row counts, attention schedule и GR. Показывать номинальный размер отдельно от пересчитанного; neural/active/table budgets должны регулироваться независимо. Сравнение с референсами использовать только при явно совпадающем accounting boundary, без автоматического прогноза качества.

Входы:

- Layers, width, vocabulary, tied/untied LM head; dense/MoE слой и общий эксперт, routed experts, top-k, intermediate widths.
- Attention schedule по слоям: GDN/full/sliding; Q/KV heads, head dims, GDN key/value heads, recurrent dtype, conv kernel, local window. KV-sharing group size и явные source→reader maps; отдельные настройки source projections, RoPE compatibility и cache replication. Baseline group=2, comparison group=1.
- Residual type, число ветвей и gate rank; separate attention/MoE operators, read-only output collapse, residual dtype и live prefill chunk size. Plain residual и n=1 GR должны быть разными настройками.
- Engram orders, heads, row counts отдельно по таблицам, row dim, placement layers, shared/independent tables, GPU/RAM split и GPU-cache budget.
- Dtypes/quantization по типам тензоров; group size, scales, padding; microbatch, cached tokens, simultaneous sequences, beams и speculative state.
- Training mode: full, LoRA/QLoRA, adapter/table-only, edit-only; optimizer/master/gradient dtypes, checkpointing, FSDP/ZeRO/TP/EP, число GPU и CPU offload.
- Размеры изменяемой n-gram памяти, ключей и метаданных после определения будущего edit-контракта.

Выходы:

- Общие neural/table параметры, активные routed параметры, shared compute, vocab-head compute и фактически извлекаемые scalars отдельно.
- Checkpoint bytes с precision/scales/padding, VRAM weights, RAM tables, KV cache по уникальным owners, GDN/Engram recurrent/conv state, GR weights и live residual activations, memory caches и метаданные изменяемой памяти. Число GR-ветвей не умножает KV. Показывать unshared comparison и различать logical sharing/physical copies.
- Inference peak estimate отдельно для prefill/decode, concurrency и ограничения конкретного backend.
- Training state breakdown и peak range с явными предположениями; full training не оценивать по active parameters. Выделять данные, которым нужен реальный benchmark.
- Теоретические FLOPs по компонентам, приблизительный traffic и host transfer; throughput показывать только при калибровке на выбранном hardware/backend.
- Сравнение профилей на графиках context/concurrency → memory и experts/top-k → capacity/compute, диаграмма состава памяти и список несовместимых настроек.
- Экспорт/import JSON/YAML и воспроизводимый отчёт с единицами GB/GiB, версией формул и подтверждёнными/аналитическими числами.

Валидация конструктора: размерности голов и групп, `top-k <= experts`, совместимость kernel/dtype, логика weight sharing и causal schedule. KV source должен быть тем же или более ранним full слоем, без циклов; head dims/RoPE/dtype должны совпадать у его readers. У reader нет отдельных K/V projections, cache allocation и append. Сверять формулы параметров с `numel()` reference-моделей, storage с реальными сериализованными весами, а VRAM — с измерениями CUDA peak. Аналитические оценки не подписывать как measured.

## Что уже сделано

- Прочитаны официальные материалы Gemma, GPT-OSS, Qwen и Engram; сохранены ссылки и revision IDs.
- Добавлены LongCat-Flash-Lite, BLT, N-grammer и ограниченный обзор SCONE/OverEncoding; принято решение по базовому числу модулей памяти.
- Residual-схема изучена по точному Flash-Next report/code и сопоставлена с mHC/AttnRes. GR включён в архитектуру и аналитические профили.
- Предложены десять актуальных профилей с одним Engram-модулем; рассчитаны neural/active/table параметры, KV, веса и состояния полного обучения. Прежние два примера с J=2 оставлены для сопоставления формул.
- Изучены современные модели от LFM2.5-230M до Kimi K3/Qwen3.8 на 2.8T/2.4T; после добавления V4.1 сохранены 27 reference snapshots с конкретными revision IDs и count conventions.
- По V4.1 введено pairwise KV-sharing в v0.3, пересчитаны все десять профилей и явно отделены CED/FP4/Single-Pass mHC/DSpark production claims от возможностей простого reference runtime.
- Зафиксировано требование изменяемой Engram-памяти; дальнейший дизайн редактирования отложен по просьбе пользователя.
- План интерактивного конструктора сохранён.
- Выполнен [облачный CPU diagnostic run](cloud-validation.md): 300 updates полного малого профиля на искусственном UK/EN corpus, full checkpoint restore и 301-й update. Это проверка устойчивости реализации, не качество Wikipedia/chat и не GPU benchmark.
- Реализован обучаемый PyTorch reference малого профиля: GR, causal Engram, chunkwise GDN, GQA с pairwise shared K/V и no-drop MoE. Подготовлены tokenizer/data pipeline, trainer, save/resume, генерация и Colab notebook. Девять CPU-проверок, включая core notebook cells, причинность/градиенты/overfit/resume, проходят; полный профиль выполнил один forward/backward/AdamW update. Получение Wikipedia parquet в этой среде блокирует CDN policy, CUDA/Colab ещё не проверены.

Не выполнены: production kernels и persistent KV/GDN decode API, интерактивный конструктор, полноценное pretraining/SFT, измерения GPU и quality/knowledge-edit benchmarks. Основные требования пользователя внесены в проект; малый профиль выбран для первого технического теста, окончательный размер модели не выбран.
