# Residual-схема next-moe v0.3

Дата: 9 октября 2026. Статус на 12 октября: включено в проект архитектуры и [обучаемый PyTorch-прототип](colab-training.md); полноценное обучение ещё не выполнено. Основной референс — Qwen3.8-Flash-Next; используются его technical report и соответствующий код `Qwen4ExpTextGatedResidual`. Источник attention/GDN — также Flash-Next.

Обновление 11 октября: после [DeepSeek-V4.1-Flash review](deepseek-v41-review.md) GR сохранён; shared KV меняет attention/cache ownership, а не residual. Добавлен Single-Pass mHC как сравнительный вариант. Его inference code прочитан; полный V4.1 report недоступен через сеть среды, поэтому matched-backbone превосходство над GR не заявляется.

## Выбор

**Gated Residual (GR), четыре ветви, отдельные read/write operators для attention и MoE каждого слоя.** N-gram память на втором decoder block включается в widened stream до attention, с общим value и отдельными keys/gates для ветвей. Одна ранняя точка памяти остаётся baseline; четыре residual-ветви не означают четыре n-gram модуля.

Почему GR подходит: он управляет тем, какую информацию читать из нескольких накопителей по глубине и насколько сильно записывать новое преобразование в каждый из них. Mixer и FFN при этом работают с шириной d, а widened stream — с n*d. Так получается расширенный residual path без увеличения ширины всех attention/expert matrices.

Начальные параметры: число ветвей `n=4`, gate bottleneck rank `rho=d/8`; оба должны быть настраиваемыми в конструкторе. BF16 residual storage для первого reference. FP8 storage и fused kernels — следующий этап, не предварительно обещанная экономия.

## Сравнение подходов

| Подход | Как передаётся информация по глубине | Стоимость и причина выбора |
| --- | --- | --- |
| Обычный pre-norm residual | `h_next = h + F(Norm(h))` | Дешёвый и нужный baseline; один накопитель |
| GatedNorm в одном потоке | Gate после RMSNorm меняет чтение | Проверить отдельно пользу gate, без расширения residual stream |
| HC / mHC | Несколько ветвей, read/write и матрица mixing `H_res`; mHC ограничивает mixing матрицу | Содержательный baseline. Для начальной версии не добавляем mixing/Sinkhorn machinery |
| Single-Pass mHC V4.1 | Sublayer заранее вычисляет coefficients следующего; pre-mix передаётся от предыдущего; сохраняются constrained H_res и Sinkhorn | Отдельный вариант для оценки traffic/latency; качественное преимущество над GR в нашем графе не подтверждено |
| GR | Несколько накопителей; динамический gate по branch+channel при чтении, scalar write per branch; без `H_res` | Выбранный путь Flash-Next; bounded read/write, компактное число одновременно нужных ветвей |
| Full / Block Attention Residuals | Softmax attention по выходам прежних sublayers или их блоковым суммам | Прямой выбор по глубине, но нужно хранить/передавать историю этих представлений; оставить сравнительным вариантом |

Это механизмы передачи по **глубине**, не механизмы attention между токенами. GR не заменяет GDN/GQA и не решает задачу KV compression.

## Первичные результаты

Flash-Next §2.2, Table 5: сравнение 25B-A3B MoE после 560B training tokens, widened variants с четырьмя ветвями:

| Residual | Loss | Average downstream score |
| --- | ---: | ---: |
| Pre-norm | 1.617 | 50.91 |
| mHC static | 1.596 | 52.49 |
| mHC dynamic | 1.594 | 54.47 |
| GR | 1.590 | 54.66 |

В этом эксперименте GR улучшает baseline и близок к dynamic mHC по downstream. Это не доказательство универсального превосходства GR. Report сообщает, что channel-wise read полезнее усложнения scalar write, чтение всех ветвей лучше pooled/last-branch reads, а `H_res` не даёт значимого дополнительного выигрыша в их ablation.

Flash-Next Table 6: Full AttnRes + GatedNorm достигает loss 1.758 против GR 1.762 в отдельном 28-layer эксперименте. Поэтому GR выбран за согласованность с целевым hybrid backbone и fixed branch count, а не как лучший residual во всех метриках.

Официальный AttnRes paper §3.1–3.2: Full AttnRes хранит O(depth*d) предыдущих представлений на live token; Block AttnRes хранит O(number_of_depth_blocks*d). При обычном training часть памяти совпадает с activations, уже нужными backward; при recomputation и pipeline parallelism возникают дополнительные retention/communication requirements. Нельзя объявлять весь этот объём дополнительным без учёта training strategy.

Сведения о manifold constraint mHC проверены по авторскому abstract и обсуждению в Flash-Next; полный исходный mHC paper напрямую здесь не прочитан. Для выбранного GR доступны и report, и конкретный implementation.

## Контракт GR

Состояние одного live token: `R ∈ R^(n × d)`, в батче `[B,T,n,d]`. На входе `R_i = token_embedding` для каждой ветви; это отдельные накопители, не четыре модели. Можно хранить тензор как `[B,T,n*d]`; normalization всегда по d внутри каждой ветви.

Для каждого attention/MoE sublayer свои параметры:

`Rhat_i = RMSNorm(R_i; gamma_i)`.

`z = flatten(Rhat)`.

`G = reshape(sigmoid(W_up * SiLU((W_down * z)/n)), [n,d])`.

`x = mean_i(G_i ⊙ Rhat_i)`.

`s = 2 * sigmoid((W_write * z)/n)`.

`y = F(x)`.

`R'_i = R_i + s_i * y`.

Shapes: `W_down:[rho,n*d]`, `W_up:[n*d,rho]`, `W_write:[n,n*d]`. Read gate G в (0,1), write coefficient s в (0,2); s не является softmax и не обязан суммироваться в единицу. Гейты token-dependent. GR сохраняет прямое аддитивное skip-слагаемое R; gate определяет новое чтение/запись. Матрица перемешивания ветвей отсутствует.

GR read уже содержит normalization: **не ставить прежний дополнительный pre-RMSNorm между ним и F**. Внутренние Q/K norms и GDN output norms остаются, поскольку выполняют другую функцию. Рецепт zero-centered RMSNorm можно реализовать с gain `1 + weight`; количество gain parameters всё равно n*d на GR operator.

На выходе сети final read-only GR (`W_write` отсутствует) сводит ветви к одному вектору d, затем применяется tied LM head. В базовом графе отдельного прежнего final RMSNorm нет: это соответствует проверенному implementation Flash-Next. Final normalization/gate fusion можно исследовать отдельно.

## Engram внутри widened stream

Lookup остаётся один на выбранный слой и зависит только от token prefix. Из конкатенированных n-gram vectors `e ∈ R^m` получаем shared value `v=Wv*e ∈ R^d` и branch-specific keys `K=Wk*e ∈ R^(n×d)`.

Для ветви i: `a_i = dot(RMSNorm(R_i), RMSNorm(K_i))/sqrt(d)`, `phi(a)=sign(a)*sqrt(max(abs(a),epsilon))`, `g_i=sigmoid(phi(a_i))`, `u_i=g_i*v`.

`delta_R = U + SiLU(CausalDepthwiseConv(GroupRMSNorm(U)))`, `R = R + delta_R`.

Для начального варианта conv kernel c=4, dilation a=3 (максимальный n-gram order). Group norms — query, key и convolution input. Это branch-wise адаптация по пути PLE/Engram в Flash-Next; это не точная копия всех деталей исходного DeepSeek Engram. Таблицы не дублируются по residual-ветвям, но keys, gates и conv channels учитывают n.

Для одного decoder block:

1. На выбранных слоях: `R ← R + Engram(tokens,R)`.
2. `R ← GR_attention(R, GDN_or_GQA)`.
3. `R ← GR_moe(R, shared_plus_routed_experts)`.

Размещение памяти **до attention второго блока** означает, что её gates уже видят результат первого блока и позволяет overlap lookup prefetch с первым блоком. Это уточняет прежний вариант с памятью после attention текущего слоя.

## Параметры и память

Без bias, с branch norm gains:

- `P_GR_read = 2*n*d*rho + n*d`.
- `P_GR_read_write = P_GR_read + n²*d`.
- Для L decoder layers: `P_GR = 2*L*P_GR_read_write + P_GR_read` (последний член — final read-only).
- Для J memory layers: `P_Engram_adapter = J*((n+1)*m*d + c*n*d + 3*n*d)`; последний член — три branch group norms.

Число table parameters не меняется от n: `P_table=J*N_orders*H_hash*Rows*D_row`. GR weights и adapter weights считаются активными на каждом токене. GR используется один раз вокруг общего результата MoE, не отдельно на каждого routed expert.

Live residual state: `M_residual = B*T_live*n*d*bytes`. BF16 при n=4 хранит в четыре раза больше **этого компонента**, чем single-stream вариант. Это не множитель всей VRAM, weights или KV. Для incremental decode T_live — число обрабатываемых новых токенов, а не весь cached prefix; для prefill — размер обрабатываемого chunk. Training может сохранять многие live states для backward; checkpoint policy учитывать отдельно.

Full GQA KV и GDN recurrent state прежних head dimensions не умножаются на n: GR подаёт mixer один вектор d. В v0.3 source строит KV из этого read, а второй full слой группы строит только Q/gate/output и читает source KV. Полный KV-кеш по-прежнему линейно растёт с контекстом, как требуется пользователю.

Memory conv history добавляет `B*J*n*d*(c-1)*a*bytes`, плюс короткую token history порядка `n_max-1`. Этот conv state сохраняется между decode steps, поэтому его нужно учитывать отдельно от transient residual activations. Детальная backend аллокация может отличаться.

FP8 residual storage в report уменьшает traffic относительно BF16 при сохранении качества в их recipe. Мы не используем FP8 в начальной аналитической оценке: требуется backend support, scaling и проверка ошибки.

## Проверки и сравнительные варианты

Проверить branch shapes, индивидуальную нормализацию, bounded gates, final collapse, отсутствие двойного pre-norm, gradient flow ко всем branches/gates и prefill/decode equivalence. Не применять in-place updates, портящие tensors, нужные backward; не допускать общей writable storage у ветвей при embedding expansion.

Сравнить plain pre-norm, single-stream GatedNorm, static widened residual и dynamic GR при том же tokenizer/data/training budget. mHC и Block AttnRes — отдельные дополнительные variants, не одновременно включённые с GR. Число потоков 1/2/4 и rank должны быть отдельными экспериментами; **n=1 GR не равен plain residual**, поскольку сохраняются learned gates.

Для Single-Pass mHC отдельно определить coefficient handoff при Engram insertion: carry coefficients от прошлого sublayer и заново вычисленные по обновлённому stream коэффициенты имеют разные semantics. Считать pre/post/H_res projections, Sinkhorn, live stream traffic, отдельные sublayer RMSNorm и final collapse. Название Mega-mHC kernel не доказывает преимущество в другом backend.

Новый multi-branch backbone меняет контекст fusion Engram, поэтому результаты one-vs-two memory modules проверить повторно при одинаковом GR, а не сравнивать single-stream one-module с GR two-module. Базовое решение — один memory module — пока сохраняется.

## Источники

- [Flash-Next technical report §2.2, Tables 5–6](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf).
- [Transformers: Qwen4ExpTextGatedResidual, PLELayer и DecoderLayer](https://github.com/huggingface/transformers/blob/927af2f73bd0d53d339f8fc0d4bd8b053f3836bb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py). Код прочитан; не запускался.
- [Flash-Next pinned config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json): `hc_count=4`, `hc_lowrank=320`, `hidden_size=2560`, rank=d/8. Эти размеры подтверждают механизм; размер нашего backbone ими не фиксируется.
- [Attention Residuals: официальный paper](https://github.com/moonshotai/Attention-Residuals/blob/85e22310fe5ee860b4a023de312d791de8a5a5e6/Attention_Residuals.pdf), прочитаны архитектурные разделы и описание memory costs.
- [mHC: авторский abstract](https://huggingface.co/papers/2512.24880); в дополнение к сравнению в Flash-Next report. Полный paper напрямую не проверен.
