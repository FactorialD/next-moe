# Тестовое обучение 0.2B + 0.1B Engram в Colab

## Открытие и запуск

1. Скачайте [next_moe_0p2b_engram_0p1b.ipynb](../notebooks/next_moe_0p2b_engram_0p1b.ipynb).
2. Откройте [Google Colab](https://colab.research.google.com/), выберите **File → Upload notebook** и загрузите файл.
3. Выберите **Runtime → Change runtime type → GPU** и выполняйте ячейки сверху вниз.
4. Оставьте `PROFILE='full'`, `SEQ_LEN=128`, `MICROBATCH=1`, `GRAD_ACCUM=4`, `STEPS=300` для первого эксперимента.
5. По умолчанию подключается ваш Google Drive: код сохраняет результаты в `MyDrive/next-moe/<RUN_NAME>`. Для временного хранения задайте `USE_DRIVE=False`; файлы `/content` исчезнут при отключении runtime.

Ноутбук самостоятельный: четыре Python-модуля встроены в ячейку и записываются в `/content/next-moe-code`. Он не зависит от доступности текущей ветки GitHub. Исходники и SHA256 каждого файла сохраняются с результатами. При изменении реализации пересоберите notebook командой `python scripts/build_colab.py`. Публикация или commit не выполнялись.

## Что обучается

Это реализация текущей архитектуры v0.3 для самого маленького профиля: 12 слоёв шириной 512, 16 routed experts / top-2 плюс один shared в каждом слое, четыре GR-потока, один Engram на втором decoder block. GDN/full GQA = 3:1; global слои 4/8 используют K/V source 4, слой 12 — собственный source. Reader имеет свои Q/gate/output и получает градиент через source K/V.

Фактический размер реализации: **193 672 208 neural + 100 016 064 table = 293 688 272 параметра**. Разница с аналитическим профилем — 320 небольших Q/K RMSNorm gains, исключённых из старой оценки. Сохраняются 65 536 embedding/head rows, даже если tokenizer на небольшом корпусе выучил меньше BPE tokens. Embedding и LM head связаны.

Resident weights и gradients FP32; autocast выполняет часть вычислений в FP16 на T4 или BF16 на поддерживаемых GPU. GDN recurrence и router scores считаются FP32. Standard AdamW хранит около 16 bytes/parameter после lazy allocation states: приблизительно 4.38 GiB без активаций и workspace. В отличие от альтернативного BF16 + FP32 master accounting, здесь 4 bytes weights + 4 gradients + 8 moments дают ту же сумму. Optional AdamW8bit — приблизительно 10 bytes/parameter плюс metadata и small FP32 tensors; CUDA/пакет надо проверить на конкретном Colab runtime.

В обоих режимах GQA использует PyTorch SDPA; backend автоматически выбирается для GPU/dtype, наличие именно FlashAttention не гарантируется. Block checkpointing и chunked/checkpointed LM loss включены. GDN реализован через точное chunkwise triangular solve, MoE — простой no-drop dispatch. Это проверяемый PyTorch reference, без production Triton GDN или fused MoE. Не предполагается конкретная скорость T4 и не гарантируется peak VRAM до измерения.

При CUDA OOM уменьшите последовательность до 64/32 и создайте новый run; microbatch оставьте 1, checkpointing включённым. Менять форму запуска при strict resume нельзя. Объём effective batch: `microbatch × sequence_length × accumulation` tokens на optimizer update.

## Корпус

Используется готовый открытый датасет, а не ручной scraping статей. Точный источник, версия, объёмы, split и правила обработки описаны в [training-corpus.md](training-corpus.md). По умолчанию 2000 отобранных документов UK/EN, 1936 для train и 64 для validation. BPE обучается только на train; фактические token counts записываются в metadata.

Первый запуск читает данные через Hugging Face streaming. Если сеть/датасет недоступны, ошибка видна пользователю; автоматической подмены toy data нет. Альтернативный `SOURCE='local'` читает отдельные UTF-8 `.txt` документы из `uk/` и `en/`. Для основной тренировки выбран пользователем Wikipedia.

## Метрики, сохранение и resume

Лог содержит training CE, auxiliary routing loss, UK/EN validation CE/perplexity, gradient norm, загрузку экспертов, tokens/s и peak CUDA allocated/reserved memory. Validation использует фиксированные окна и не сдвигает training sampler. Значения throughput включают overhead интервала логирования и не заменяют отдельный benchmark kernels. После запуска строятся loss curves и карта expert assignments.

`last.pt` содержит model, optimizer, GradScaler, completed step, tokens_seen и RNG. Сохраняется каждые 100 updates, в конце и по возможности при KeyboardInterrupt. При жёстком обрыве Colab восстанавливается только последний сохранённый checkpoint. При прерывании внутри accumulation частичные gradients не сохраняются; sampler/RNG может уже продвинуться. FP16 overflow вызывает уменьшение GradScaler scale и повтор тех же микробатчей без засчитывания update. OOM, non-finite loss и устойчивые non-finite gradients останавливают запуск с понятным сообщением.

Восстановление: сохраните тот же `RUN_NAME`, поставьте `RESUME=True`, выполняйте notebook заново. Требуются неизменные tokenizer, token files, model shape, batch shape и optimizer recipe; содержимое data files проверяется SHA256. `STEPS` — общее число updates; его увеличение меняет будущую cosine LR schedule. Исходные token files и tokenizer берутся из Drive, не создаются заново.

Экспорт `model.pt` содержит только модель/config, `tokenizer.json` хранится рядом. Код отдельной загрузки для inference дан в notebook и выключен по умолчанию, чтобы не создавать вторую копию на GPU. Источники и конфигурация достаточны для воспроизведения graph; training resume требует также data files и `last.pt`.

Full AdamW checkpoint примерно 3.28 GiB; для atomic replace одновременно нужны старый и временный новый файл. Отдельный model export примерно 1.09 GiB. Рекомендуется минимум 10 GiB свободного диска. Drive mount выполняет пользователь в своём Colab; этот workflow не загружает результаты на Hugging Face или в чужие сервисы.

## Границы результата

300 updates × 128 tokens × microbatch 1 × accumulation 4 = **153 600 processed training tokens**, с возможными повторами окон. Это существенно меньше бюджета языковой модели этого размера; связные ответы, знание мира или instruction following не обещаются. Режим `tiny` — исключительно debugging profile и явно не модель 0.2B + 0.1B.

Генерация пересчитывает последние `SEQ_LEN` tokens и не имеет persistent KV/GDN decode API. Ограничение относится к тестовому runtime, а не к архитектурной цели растущего кеша. В этой версии не проверены 32K/128K/256K, incremental decode equivalence, edits Engram, производительность CUDA kernels и качество long-context recall.

Проверки реализации запускаются `python -m pytest tests -q`: GDN chunk outputs и gradients против независимой recurrence, causal prefix/future-token isolation, chunk-size invariance, checkpoint/no-checkpoint gradients, receiver→source KV gradients, Engram reset, параметризация full profile, overfit tiny batch, tokenizer roundtrip и leak-free document split, сохранение/resume optimizer/RNG и генерация. CPU-проверки не заменяют реальный запуск на Colab GPU.

## Проверено в этой среде 12 октября 2026

- По следующему запросу пользователя выполнены также **300 updates полного профиля** на CPU с sequence length 32 и искусственным UK/EN diagnostic corpus, затем full checkpoint восстановлен и выполнен 301-й update. [Результаты и границы проверки](cloud-validation.md); это не Wikipedia run или тест default Colab 128/1/4.
- Девять автоматических проверок прошли, включая исполнение встроенных core cells notebook в режиме tiny/local/CPU, обучение, export и генерацию. Структура notebook валидна; встроенные исходники совпадают с файлами проекта.
- Полный профиль на **293 688 272 параметра** выполнил forward/backward и один AdamW update на CPU с синтетической последовательностью из 16 токенов. Loss и gradient norm конечные; это проверка исполнимости, не обучение на Wikipedia и не benchmark Colab.
- Revision и карточка Wikipedia проверены через API. Получение parquet здесь блокирует сетевой policy для `us.aws.cdn.hf.co` (HTTP CONNECT 403). Фактический streaming download и полный запуск на Colab GPU **не проверены**; данные должны загружаться в пользовательском Colab, а source/volume manifest будет создан там.
