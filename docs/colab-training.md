# Обучение 0.2B + 0.1B Engram в Colab: workflow v2

После [пользовательского T4 run на 300 updates](colab-results-and-next-steps.md) notebook обновлён для ограниченного Drive, FP16 model parameters и более длинного token budget. Состав модели v0.3 не менялся.

## Открытие

[Открыть в Colab](https://colab.research.google.com/github/FactorialD/next-moe/blob/work/notebooks/next_moe_0p2b_engram_0p1b.ipynb), включить **Runtime → Change runtime type → GPU**, выполнять ячейки сверху вниз. Notebook содержит пять Python-модулей; клонировать repo не требуется. Источники и SHA256 сохраняются в run. Для пересборки после изменения Python-модулей: `python scripts/build_colab.py`.

## Первый запуск обновлённого режима

1. Оставить `PROFILE='full'`, context128, microbatch1, accumulation4, optimizer AdamW. `PRECISION='auto_master'` выбирает **FP16 модель + FP32 masters на T4**, BF16 модель на поддерживающих GPU и FP32 в CPU tiny tests.
2. Сначала поставить **`SESSION_UPDATES=20` или 50**. Новый CUDA FP16-master режим ещё не измерен на T4, даже если старый FP32+autocast notebook работал. Проверить конечность loss/gradients, GradScaler retries, memory и save/load.
3. Затем оставить `TOTAL_TOKENS=10_000_000`, увеличить session limit до 1000, повторить training cell. Внутри живого runtime сохраняется optimizer; не требуется пересоздавать модель при каждом session limit.
4. Около 1M cumulative tokens проверить validation, generation, накопленный routing и corpus sampling passes. 10M — исследовательская цель, не обещание chat quality/полноценного pretraining. По прежнему времени это ~27.25 h с нуля без ускорения.

Total tokens включают прошлый checkpoint. Steps вычисляются из remaining tokens и текущего effective batch, LR рассчитан на весь план. Например, продолжение с step300/tokens153600 при batch512 до10M заканчивается около step19532. Session boundary не охлаждает LR до минимума.

## Продолжить предыдущую тренировку

По умолчанию новый run называется `next-moe-0p2b-engram0p1b-10m-v2`. Если в `MyDrive/next-moe/next-moe-0p2b-engram0p1b-test-v1` есть старый `last.pt`, notebook выбирает **warm start** из него и использует старые tokenizer/metadata при расширении корпуса. Веса и progress переносятся, optimizer/RNG создаются заново; есть 100-update LR ramp. Это намеренная новая фаза с иной precision/данными/schedule, не exact продолжение первого run.

Для другого расположения указать `INIT_CHECKPOINT`, `INIT_TOKENIZER`, `INIT_METADATA`. Старый inference-only `model.pt` не содержит step/tokens; автоматический импорт его не угадывает. Использовать старый full `last.pt` либо новый `weights.pt`. Старые файлы не удаляются.

`RESUME_MODE='auto'` для текущего v2 run:

- Если local `last.pt` доступен, exact resume с optimizer/masters/scaler/RNG, неизменным recipe/data/source.
- После потери Colab runtime — warm restart из Drive `weights.pt`, с сохранением весов/step/tokens и сбросом optimizer/RNG/master low bits. Начинается новая optimizer phase.
- Если snapshot отсутствует, импорт INIT checkpoint либо fresh run.

Для нового tokenizer/весов выбрать новый RUN_NAME и `RESUME_MODE='fresh'`. Не retrain BPE существующей модели: Engram hash addresses зависят от неизменных token IDs. Exact resume требует совпадения кода и данных; изменение precision/recipe делать явно через warm phase.

## Корпус и кеш

Источник [wikimedia/wikipedia](training-corpus.md), pinned snapshot/revision. Новый предел — 10 000 документов и 30M Unicode characters на язык, max20K/article; fresh validation128/lang. При продолжении сохраняются прежние validation hashes/order и точные BPE bytes; исходные val token files должны совпасть. Реальные document/token counts и ожидаемая степень повторения корпуса печатаются, не предполагаются по nominal caps.

В Drive лежит `data-v2`; token files копируются в local cache, чтобы train sampler не делал random reads через Drive. Corpus collection ограничен по characters; token files пишутся документами, без огромного общего Python списка всех token IDs. EOS между статьями; attention/GDN не сбрасываются на границе, Engram suffix hashes сбрасываются. Сеть/датасет не заменяются скрытым toy corpus.

## Precision и память

Модель содержит **293 688 272 parameters**, neural193672208/table100016064. FP16 параметры и gradients модели обновляются через **FP32 master weights и moments**. Scaled model gradients копируются в FP32 masters, после чего half gradients освобождаются; public GradScaler unscale_/clip выполняются на masters. Master update копируется обратно в модель. FP32 GDN recurrence/router сохраняются, scalar decay вычисляется FP32.

FP16 модель не означает FP16 Adam и не гарантирует меньше total VRAM: master copy + transient FP32 gradients тоже занимают место. Старый user run имел peak allocated4.811/reserved5.334 GiB с FP32 parameters/autocast; эти числа не переносятся автоматически на новый recipe. Начать с тех же коротких context/batch и измерить новый peak. Optional AdamW8bit уменьшает moments, но требует CUDA/стабильности проверки; он не включён по умолчанию.

GQA использует SDPA с автоматически выбранным backend; FlashAttention зависит от GPU/dtype. GDN — exact triangular PyTorch reference, MoE — no-drop Python dispatch; production kernels не реализованы. Не начинать с 32K на этой реализации.

## Drive до 5 GiB

В Drive сохраняется один `weights.pt` без optimizer, около **0.55 GiB FP16**. Пока пишется временная копия, нужен объём старого + нового snapshot, примерно **1.10 GiB**, плюс corpus/metadata/source/metrics. `PERSISTENT_BUDGET_GIB=4.5` проверяет весь каталог одного run с запасом на temporary save; это не чтение реальной Drive quota и не учёт остальных пользовательских файлов.

Full optimizer checkpoint только локально: `LOCAL_DIR/last.pt`, примерно **3.83 GiB** после полного lazy allocation moments. Local atomic overwrite требует около7.66 GiB; Colab runtime disk должен иметь место. Local файл исчезает при завершении runtime. Не создаётся отдельный duplicate `model.pt` и ZIP каталога.

При `USE_DRIVE=False` все snapshots находятся в `/content` и тоже исчезнут при потере runtime.

Если old last.pt/model.pt уже занимают доступное место в Drive, скачать/освободить ненужные old files самостоятельно после проверки нового weights snapshot. Workflow не удаляет их и не обещает, что пользовательская квота равна заданному budget run. При недостатке места запись останавливается, предыдущий complete snapshot сохраняется.

Цена такой экономии: weights-only snapshot позволяет **warm restart**, но не exact optimizer/RNG восстановление после потери runtime. Если нужен exact resume, нужен full checkpoint на постоянном диске либо отдельно проверенный compact optimizer recipe; память не исчезает от малого active parameter count.

## Метрики и generation

Validation16 fixed windows/lang, independent RNG; новый initial baseline записывается отдельно, values отличаются от старого evaluator. Метрики: CE/ppl, GradScaler/gradient norm, peak VRAM, time data loading, training step throughput без eval/save и interval throughput с overhead. Phase-wide routing counters дополняют last-update assignments; heatmap нормирована по mean load каждого слоя.

Live metrics пишутся локально и копируются в Drive при checkpoint save (каждые500 updates и в конце session). Последний checkpoint не записывается дважды. Ctrl+C пытается сохранить последний завершённый update; жёсткий обрыв сохраняет лишь последний complete snapshot. Частично накопленные gradients не сохраняются.

Generation остаётся коротким prefix recomputation; persistent KV/GDN decode API пока нет. BOS не добавляется, поскольку corpus не обучал его на starts. Нельзя оценивать качество чата по этому продолжению текста: [SFT/chat план](chat-training.md) следует после base training.

## Что проверено

- 13 automated CPU tests: causality, GDN recurrence/gradients, shared K/V gradient, chunk/checkpoint invariance, full profile size, overfit, notebook core cells, master optimizer exact/warm resume, quota failure и сохранение tokenizer/validation при расширении.
- Новый full FP16-master профиль выполнил два synthetic CPU updates, finite gradients и сохранение weights0.5473/full3.8099 GiB. [Результаты и ограничения](colab-results-and-next-steps.md).
- Старый полный CPU run: [300 updates + resume](cloud-validation.md). User T4 v1 run: [300 updates на выбранном корпусе](colab-results-and-next-steps.md).
- Новый FP16-master CUDA/T4 recipe, расширенный Wikipedia download и длительное качество пока не проверены в этой среде. Здесь нет GPU, CDN Wikipedia блокируется network policy.
