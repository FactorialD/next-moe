# Пользовательский Colab run: выводы и следующий эксперимент

12 октября 2026 пользователь предоставил лог 300 updates, графики и generation output; GPU **T4** подтверждён отдельно. [Сводка исходных чисел](../reports/colab-user-300/report.json). Это пользовательский результат, не собственный rerun GPU. Dataset metadata и версия CUDA/PyTorch не предоставлены; точный размер corpus/tokenizer не угадываем по одиночному числу после лога.

## Выводы

- Training CE 11.100 → 8.266 для отдельных sampled batches. Старт близок к `ln(65536) ≈11.09`, то есть почти равномерному предсказанию по vocab.
- Validation UK 9.294 → 8.712, EN 8.845 → 8.255 между updates 50 и 300. Оба продолжают улучшаться; очевидного разворота validation вверх в этих точках нет. Это лишь четыре окна на язык, не comprehensive language benchmark.
- Последние train values колеблются около 8; plateau не означает, что модель обучена. Схема быстро обучает частоты/простые связи, но 153 600 tokens недостаточно для хорошего языка. LR уменьшается с 2e-4 до 2e-5 уже за 300 updates; дальнейший эксперимент должен иметь schedule на весь бюджет. Спад LR не обязательно единственная причина plateau.
- Repetitive «the», пунктуация и смешение языков в generation согласуются с очень ранним обучением. Это не подтверждение chat skills или работоспособности Engram edits. Repetition penalty мог бы косметически скрыть повторы, но не исправил бы знания/язык.
- Peak CUDA allocated **4.811 GiB**, reserved **5.334 GiB**. Это реальное измерение данного v1 recipe и короткого context, не прогноз для нового FP16-master режима или 256K.
- Worst-layer load/mean доходил до 7.97, в конце 4.78. При E=16,k=2 максимум этой метрики — 8: ранний hot expert почти всегда выбирался, позже распределение улучшилось. Последний update не оставил невыбранных experts. Auxiliary loss около 1.04 сам по себе не доказывает баланс; cumulative phase counters полезнее одной heatmap.

Полный elapsed 25.11 min даёт **101.95 tokens/s**, ниже обычных 140–158 tokens/s в тренировочных интервалах. Провалы после checkpoint saves согласуются с IO overhead; поздние замедления могут иметь и другие причины. Не объявляем Drive единственной причиной без отдельного измерения.

## Выбранный следующий шаг

1. Проверить **20–50 updates** нового FP16-master режима на T4: finite loss/gradients, GradScaler retries, VRAM и save/reload. Предыдущий GPU run проверял FP32 parameters + autocast.
2. Продолжить прежние weights с **неизменным tokenizer**; расширить Wikipedia до 10 000 документов на язык, max 30M characters на язык, сохранив прежние validation documents. Фактические counts и sampled corpus passes печатаются.
3. Сначала оценить прогресс около **1M processed tokens**; план notebook — **10M**, сессиями по 1000 updates. При batch 512 это около 19 532 updates суммарно. Это исследовательский этап, не завершённый pretraining.
4. По старому полному времени 1M ≈2.72 h, 10M ≈27.25 h; по steady-state 150 tokens/s около 1.85/18.52 h. Новые IO/precision могут изменить скорость, поэтому это только экстраполяции.
5. Оценивать UK/EN validation по большему фиксированному набору окон, cumulative routing и generation. При ухудшении validation расширять/очищать data, а не повторять тот же небольшой корпус без ограничения.
6. Не менять E/top-k/GR/Engram одновременно с recipe. После более длинного baseline сделать matched-data/budget ablations, например без Engram, чтобы отделить пользу архитектуры. SFT/chat — после приобретения base language skills.

Нельзя назвать универсальное «полноценное число шагов»: оно зависит от tokens/update, данных и цели. Для качественного pretraining потребуются гораздо больший корпус и, вероятно, миллиарды tokens. При нынешнем pure-PyTorch throughput это дорого; production GDN/MoE kernels стоит исследовать до большого бюджета.

## Исправления workflow

- Persistent Drive snapshot содержит **только weights + config/progress**, без Adam/master tensors: FP16 около **0.55 GiB**. Существующий snapshot плюс временная новая копия около **1.10 GiB**, далее data/metadata. Budget run 4.5 GiB проверяется перед сохранением; остальные каталоги Drive не входят в него.
- Full resume checkpoint хранится локально в Colab. Новый exact FP16-master payload примерно **3.83 GiB** после полного создания Adam states; atomic overwrite требует места на две копии. После потери runtime доступен **warm restart**, а не exact optimizer/RNG resume. Это явный компромисс limited disk, не незаметная потеря states.
- Старые файлы пользователя не удаляются. Если old `last.pt`/`model.pt` уже заняли Drive, сначала сохранить/проверить новый weights snapshot и освободить ненужные старые файлы самостоятельно. Не создавать автоматически второй model.pt или ZIP большого run.
- T4: model parameters/gradients **FP16**, FP32 master weights/moments, GradScaler; gradients копируются в FP32 для clipping/update, затем освобождаются. BF16 только на поддерживающих GPU. FP16 Adam moments и masterless FP16 Adam не включаем.
- Это не обещает меньшего total VRAM: master weights занимают место, а FP32 optimizer gradients временно создаются. Рекомендуемый старт остаётся context128/microbatch1, дальше benchmark context256 отдельно. Optional 8-bit Adam нужен для снижения moment storage, но требует CUDA проверки и отдельного сравнения устойчивости.
- LR ориентирован на **общий token budget**, session limit его не обрывает. Warm restarts имеют 100-update ramp; momentum/RNG и extra master precision из weights snapshot не восстанавливаются.
- Token files и live metrics кешируются локально; Drive получает snapshot/metrics при save. Измеряется training-step throughput отдельно от interval overhead и data loading.
- Validation использует независимые RNG на язык и 16 окон. Новый evaluator отличается от прежнего, поэтому его initial baseline записывается заново; не сравнивать сырые values с v1 как будто окна одинаковы.
- Убрано добавление BOS при generation: текущий corpus не обучал BOS на document starts. Это исправление format mismatch, не решение проблемы повторов.

## Проверка изменений

13 automated CPU tests проходят, включая master-weight update/exact resume/warm restart, quota rejection без повреждения предыдущего snapshot и expansion с byte-identical tokenizer/validation token files. Полный 293.69M FP16-master profile выполнил два updates на CPU с synthetic data, finite loss/gradients и сохранением: **weights 0.5473 GiB**, full checkpoint **3.8099 GiB** (часть Adam states ещё могла не создаться). [Отчёт](../reports/fp16-master-smoke/report.json).

Это подтверждает CPU implementation mechanics, но **новый FP16-master CUDA режим на T4 ещё не измерен**. Его speed/peak VRAM/long-run quality проверяются первым коротким session run пользователя.
