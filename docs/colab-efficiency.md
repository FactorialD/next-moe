# Экономия квоты Colab: живой бенчмарк

По пользовательскому run: 9.216M processed tokens, seq_len=512, microbatch=1,
accumulation=4, G4 с ~96 GB VRAM; шаг обучения ~1155 tokens/s, полный интервал
~1100–1150 tokens/s. Модель занимает ~6.4 GB по панели Colab. Это показатель памяти,
а не загрузки вычислительных блоков GPU. CUDA-профилирование ещё не проводилось.

При остатке 57.54 units и расходе 8.9 units/hour остаётся ~6.46 GPU-hours.
При прежней скорости это ~25.6–26.8M дополнительных tokens, до расходов на
запуск, validation, saves и benchmark. До 10M остаётся 784k tokens (~12 минут,
~1.7–1.8 units). Общее достижение 30M выглядит реалистично с запасом; до 100M
при прежней скорости потребуется ~22–23 часа и ~195–204 units.
Все числа — оценки по конкретному сеансу, не тарифы или гарантии Colab.

## Что измеряем

`scripts/colab_efficiency.py` сравнивает для текущего effective batch 4 sequences:

| Microbatch | Accumulation | Block checkpointing |
|---:|---:|---|
| 1 | 4 | включён |
| 2 | 2 | включён |
| 4 | 1 | включён |
| 4 | 1 | выключен |

При seq_len=512 во всех вариантах 2048 tokens/update. Более крупный microbatch
уменьшает число Python-вызовов GDN/MoE и увеличивает матричные операции.
Отключение checkpointing убирает повторные вычисления блоков ценой активаций.
Checkpointing выходного loss остаётся включён. Ускорение на G4 заранее неизвестно.
Router auxiliary loss зависит от microbatch, поэтому одинаковое число tokens
не означает полностью идентичный градиент между вариантами.

Каждый вариант: 2 прогревочных + 10 измеряемых optimizer updates. На CPU создаётся
копия весов, FP32 masters, Adam, scaler, routing и RNG (~размер full checkpoint
в RAM). Перед каждым вариантом и после теста всё восстанавливается. Тестовые
токены не прибавляются к обучению, но GPU-время расходует квоту.
Drive snapshots и validation на время теста отключены. В конце сохраняется
маленький `efficiency-benchmark.json`; fastest layout применяется по умолчанию.

Проверено на CPU: точное восстановление модели, Adam moments/step, FP32 masters,
scaler, RNG, прогресса и исходного metrics.jsonl. GPU speed надо измерить в Colab.
При ошибке выполняется восстановление и ошибка передаётся пользователю; не
продолжайте обучение после прерывания самого восстановления. Benchmark требует
хотя бы один завершённый optimizer update и 12 updates до конца текущего бюджета.

## Уже открытый Colab: не пересоздавать trainer

В новую ячейку после создания текущего `trainer`:

```python
import urllib.request
url = "https://raw.githubusercontent.com/FactorialD/next-moe/work/scripts/colab_efficiency.py"
exec(urllib.request.urlopen(url).read().decode("utf-8"))
results = benchmark_batch_layout(trainer, units_per_hour=8.9)
```

Это исполняет код проекта из ветки work; для воспроизводимости замените work
на конкретный commit. Не запускайте повторно ячейку создания модели: она заменит
живой trainer. Benchmark сохраняет optimizer в памяти. Проверка validation
использует прежние eval_batches и microbatch=1 даже при новом train microbatch.
Это сохраняет набор validation windows при прежних data/tokenizer/seq_len/seed.

Продолжение в том же runtime, например до 30M **общих** tokens:

```python
extend_token_budget(trainer, 30_000_000, ramp_updates=100)
trainer.run(max_updates=1000)
```

Helper сохраняет Adam и включает LR ramp на 100 updates вместо скачка LR при
увеличении step-based cosine budget. Это изменение расписания, а не exact
продолжение прежнего 10M recipe. Для будущего reload запишите новые TOTAL_TOKENS,
MICROBATCH, GRAD_ACCUM и CHECKPOINT_BLOCKS в параметры ноутбука. Локальный
last.pt сохраняет полный optimizer; Drive weights.pt даёт warm restart после
потери runtime. Прямое увеличение trainer.cfg.steps без ramp не рекомендуется.

В обновлённом notebook эти функции встроены: RUN_EFFICIENCY_BENCHMARK=False
по умолчанию; True запускает тест при создании trainer. На свежей модели сначала
выполняется один настоящий update для выделения Adam states.

## Как выбирать GPU

Сравнивайте tokens per compute unit = tokens/s × 3600 / units/hour,
а для итогового бюджета используйте скорость с validation/save overhead.
При 1150 tokens/s и 8.9 units/hour G4 даёт ~465k training tokens/unit.
T4 при прежних ~145 tokens/s выиграет только при расходе менее ~1.12 units/hour.
Текущие тарифы T4/A100/L4 неизвестны: нужны их реальные показания и тот же
benchmark. Частичная занятость VRAM сама по себе не повод менять GPU.

Несколько независимых Colab sessions не складываются в одну обученную модель.
DDP требует координации и связи, а оплачивается время всех ускорителей. Поэтому
сначала используем один G4 эффективнее, затем сравниваем tokens/unit с другими
GPU. После сохранения weights завершайте неиспользуемый GPU runtime: ожидание
тоже может расходовать квоту. При смене runtime локальный optimizer теряется.
