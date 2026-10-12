# DeepSeek V4.1 и изменение next-moe v0.3

Дата: 11 октября 2026. Проверенный референс — **DeepSeek-V4.1-Flash**, а не неуточнённое имя V4.1. Прочитаны официальные model card, checkpoint config и reference inference code; новые веса не скачивались и не запускались. Revision: `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`.

## Что подтверждено

| Компонент | DeepSeek-V4.1-Flash | Значение для проекта |
| --- | --- | --- |
| Размер | Заявлены 552B backbone + 196B Engram | Разделять neural и table counts; это около 748B до отдельных vision/draft overhead |
| MoE | 40 слоёв, d=5120; 384 routed/top-6 + 1 shared; f=2304 | Больше экспертов не требует пропорционального увеличения top-k; размеры не оптимальны автоматически для нашей сетки |
| Causal Encoder-Decoder | По карточке: causal encoder 20 слоёв + decoder 20; 8B active prefill / 16B decode | Может экономить prefill compute; требует архитектурно готового decoder state |
| CSA2 | Global KV sources `[2,8,14,20]`, index sources `[2,8,14,20,24,28,32,36]`, индексы code с 0 | Разные слои могут читать одну глобальную память, сохраняя собственные queries |
| Global compression | Encoder ratio=2, decoder ratio=1; static Full/Reindex/Reuse modes по карточке | Полный cache length всё ещё растёт с контекстом; sparse selection не равно удалению всей истории |
| Hierarchical indexer | Candidate source=20, 2048 blocks × 8; финальный top-512 | Последующие indexers могут работать по меньшему candidate pool; начальный поиск по истории остаётся |
| Quantized cache | Main KV: FP4 E2M1 + E4M3 scale на 16 каналов; index K: FP4 + scale на 32 | Scales обязательно включать в storage; заявленные 890 bytes/token относятся к shared global KV + index K |
| Residual | Single-Pass mHC, четыре потока, Sinkhorn=20 | Новый содержательный сравнительный residual; Single-Pass не означает отсутствие Sinkhorn |
| Engram | Два модуля: decoder blocks 2 и 15 (1-based); 2/3/4-граммы, по 8 heads, D=256 | Поддерживает масштабирование таблиц; само наличие второго модуля не доказывает его преимущество при фиксированном бюджете |
| Memory fusion | Общий value, отдельные keys/gates для потоков; conv в данном reference отсутствует | Наш fusion близок, causal conv стоит проверить отдельно on/off |
| MoE routing | `sqrt(softplus(logit))`; selection bias влияет на выбор, weights берутся без bias | Возможный auxiliary-loss-free routing variant; inference code не раскрывает полный training update recipe |
| DSpark | Три draft layers, block size=5; confidence-scheduled verification заявлена авторами | Инференс-ускорение для будущего этапа, не уменьшение основной памяти модели |

Config даёт точнее `196,613,849,600` scalars таблиц: `(384,006,168 + 384,016,682) × 256`. Это отдельно от округлённого авторского 196B. API checkpoint element counts для packed/quantized weights не используются как логические параметры.

## Что стало базовым изменением

**В next-moe v0.3 пары последовательных full GQA слоёв используют один KV source.** Сохраняем GDN/full 3:1, GR, MoE, один ранний Engram и весь causal KV prefix. От V4.1 переносим принцип явного владения глобальной памятью и её чтения несколькими слоями; наша версия использует несжатые BF16 K/V и dense full attention, без CSA2 indexer.

Для global слоёв `[4,8,12,16,20,24,...]` группы имеют вид `{4→4,8→4}`, `{12→12,16→12}`, `{20→20,24→20}`, …, индексы с 1. Первый слой пары строит `K=Wk*x_source`, `V=Wv*x_source`, где `x_source` — его собственный GR attention read. Второй не имеет отдельных K/V projections и читает тот же cache. Q, output gate и output projection у каждого full слоя собственные. Непарный последний full слой имеет собственный KV source.

Это **изменение вычислительного графа**, а не упаковка двух разных кешей в одну allocation и не простое tying весов: одинаковые Wk/Wv, применённые к разным hidden states, дали бы разные K/V и не сократили число кешей. Каждая пара читает одни и те же построенные source tensors; сохранённая память содержит представление каждого токена без eviction/pooling. Качество по сравнению с независимыми KV остаётся экспериментальным вопросом.

Почему включаем: число уникальных cache owners уменьшается примерно вдвое без дополнительных таблиц, quantization или sparse retrieval. Это прямое улучшение аналитической памяти под цель меньшего растущего кеша. Дополнительный референс [Cross-Layer Attention, 2024](https://huggingface.co/papers/2405.12981) сообщает 2× снижение KV при близком качестве в обученных 1B/3B MQA моделях; здесь прочитан авторский abstract. Он не подтверждает автоматически наш interleaved GDN/GQA вариант: у наших двух global слоёв между ними есть GDN-блоки.

Нужны общие KV head dims, position IDs, RoPE settings и dtype внутри группы. Causal mask применяется по позиции query, не по номеру source layer. При decode владелец дописывает токен ровно один раз, reader только читает. Prefill cache должен включать уже обработанные chunks; нельзя detach source K/V при обучении, если требуется gradient от reader. Pipeline placement может потребовать перенос или репликацию кеша; логическая экономия не гарантирует такую же физическую VRAM без поддержки backend.

При `L_full=L/4`, `L_kv=ceil(L_full/2)`:

`P_full_GQA = L_full*3*d*Hq*dh + L_kv*2*d*Hkv*dh`.

`M_KV = 2*B*T*L_kv*Hkv*dh*bytes_per_scalar`.

Четыре GR-ветви не умножают эти значения. GDN state и количество MoE/Engram параметров не изменяются. Core/active/weight/optimizer estimates десяти профилей пересчитаны, поскольку удалены receiver K/V projections.

| Профиль, neural + table | Full слоёв / KV owners | BF16 KV 256K до / после, GiB |
| --- | ---: | ---: |
| 0.2B + 0.1B | 3 / 2 | 0.375 / 0.250 |
| 0.5B + 0.2B | 4 / 2 | 0.500 / 0.250 |
| 1B + 0.4B | 5 / 3 | 1.250 / 0.750 |
| 3B + 1B | 6 / 3 | 1.500 / 0.750 |
| 7B + 2B | 7 / 4 | 1.750 / 1.000 |
| 20B + 5B | 8 / 4 | 2.000 / 1.000 |
| 70B + 20B | 10 / 5 | 2.500 / 1.250 |
| 125B + 50B | 12 / 6 | 3.000 / 1.500 |
| 400B + 100B | 15 / 8 | 7.500 / 4.000 |
| 1T + 200B | 18 / 9 | 9.000 / 4.500 |

Одна последовательность, 262144 cached tokens. Это KV capacity, не peak VRAM. Сравнение с all-full без sharing даёт примерно 6–8× меньше KV в зависимости от нечётного числа global слоёв, вместо прежних 4×. Full-attention вычисления не становятся sparse и не перестают быть квадратичными при prefill.

## Что исследуем, но не включаем по умолчанию

**CED и SWA Bounded Replay.** Для GDN состояние зависит от всего причинного префикса; replay последних 128 токенов в общем случае не восстановит его точно. Нельзя одновременно оставить прежние decoder GDN и заявить пропуск их prefill. Отдельный CED-вариант должен либо сохранять/строить необходимые recurrent states, либо менять decoder mixers на подходящий stateless/global-memory + SWA путь. Нужны budget для encoder boundary representations, replay state, KV/index K и дополнительные latency measurements.

**Single-Pass mHC.** Reference computes коэффициенты следующего sublayer заранее и передаёт pre-mix от предыдущего. GR использует другой channel-wise read и не имеет H_res. Данных, что новая mHC лучше GR при одинаковом нашем backbone, не найдено; сравнение Flash-Next поддерживает сохранение GR. Добавляем этот mHC как отдельный variant с корректным coefficient handoff через Engram insertion, final collapse и стоимостью Sinkhorn/mixing, а не заменяем GR по одному benchmark всей модели.

У mHC есть конкретное преимущество по размеру coefficient projections: `2*L*n²*(n+2)*d`, против наших GR matrices с low-rank channel gates. При L=72, d=5120, n=4 это ≈70.8M только mHC projections, тогда как полный текущий GR accounting даёт ≈3.82B. К mHC ещё нужны scales/bases/norms и Sinkhorn/stream mixing; эти разные parameter boundaries нельзя объявлять измеренным speedup. Это основание приоритетно сравнить residual variants на крупных размерах, но не доказательство лучшего качества или latency.

**Два Engram-модуля и 4-граммы.** Добавлены абляции one/two modules при одинаковом table budget, 2/3 против 2/3/4 orders, conv on/off. У V4.1 другой CED/mHC backbone и огромный training budget; наличие двух модулей не отменяет предыдущие matched-budget результаты Flash-Next. Единственный ранний модуль остаётся стартовым вариантом.

**Token-ID compression.** V4.1 normalizer применяет NFD + StripAccents + Lowercase. Для украинского это объединит, например, `ї→і` и `й→и`. Этот normalizer не переносим в baseline. Любой compressed map должен сохранять значимые различия UK/EN и иметь стабильную версию. Подробности edit API по-прежнему отложены.

**Aux-free routing, hierarchical indexing, FP4 KV и DSpark.** Routing selection-only bias можно проверить при тех же E/k/f, отдельно от смены score function. Hierarchical indexing применим только после разработки sparse-attention варианта и accounting первого candidate search. FP4 KV требует quality tests и настоящего packed-storage backend; он не включён в пересчитанные BF16 estimates. DSpark требует отдельного draft training/verification contract.

## Как понимать заявленные 890 bytes/token

По config есть три encoder KV sources с ratio=2 и один decoder source с ratio=1. Main KV dim=512; index K dim=128. Для packed main FP4 + scale byte на 16 channels: `0.5+1/16=0.5625 bytes/scalar`. Для packed index FP4 + scale byte на 32: `0.5+1/32=0.53125`.

`M_main/token = (3/2 + 1)*512*0.5625 = 720 bytes`.

`M_index/token = (3/2 + 1)*128*0.53125 = 170 bytes`.

Итого 890 bytes/token, совпадающие с карточкой. В число не входят SWA buffers, replay/workspace, transient index scores, encoder boundary storage и backend padding. При 256K это примерно 0.217 GiB **глобальной packed памяти этой модели**; переносить число на наши heads/layers нельзя.

Reference code выделяет BF16 cache tensors и делает quantize/dequantize in-place. Поэтому он симулирует FP4 precision, но не реализует физические 890 bytes/token. Его `Transformer.forward` также проходит все backbone layers; CED skip-prefill и SWA replay не продемонстрированы этим простым runtime. Заявленные 8B/16B active и compressed-cache performance следует считать результатами авторского production design.

В reference `Indexer.forward` сначала вычисляет scores по всей доступной истории, затем маскирует позиции вне candidate pool. Это проверяет смысл hierarchical selection, но не доказывает production cost, независимый от context length, для глубоких indexers. Такой выигрыш требует kernels, которые физически читают/оценивают только кандидатов.

## Источники и границы

- [Официальная карточка](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/README.md).
- [Checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/config.json).
- [Attention, routing, Engram fusion и Single-Pass mHC](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py).
- [Hash layout и token normalizer](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/engram.py).
- [Quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/kernel.py) и [границы reference runtime](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/README.md).
- [Technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/DeepSeek_V41_Tech_Report.pdf): найден, но PDF redirect на `us.aws.cdn.hf.co` блокируется сетью среды. Полный paper и его ablations не заявляются прочитанными.

Эти источники подтверждают механизм sharing и позволяют проверить storage arithmetic. Они не доказывают quality gain именно next-moe v0.3. До выбора финального графа сравнить group sizes 1/2 при одинаковых данных и отдельно при одинаковом parameter/compute budget; измерить UK/EN quality, long-context recall и prefill/decode equivalence. Для GQA с sharing нужна проверка градиентов receiver → source и отсутствие future-token leakage.
