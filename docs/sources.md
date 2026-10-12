# Источники и подтверждённые выводы

Проверено 9 октября 2026. Прочитаны официальные GitHub README, implementation/configuration files и относящиеся к архитектуре разделы двух technical reports. Во время первого этапа Hugging Face возвращал HTTPS tunnel 403; в последующем исследовании текстовые HF-файлы стали доступны. Прочитаны model card/config/code LongCat-Flash-Lite и config Flash-Next. Полнота проверки у каждого источника указана отдельно.

## Референсы пользователя

- [Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b).

## Gemma 4

[Официальная конфигурация Gemma4_26B_A4B, Google DeepMind](https://github.com/google-deepmind/gemma/blob/0f7e7420e84c6789d96ba465608d2a008ecb443d/gemma/gm/nn/gemma4/_gemma4.py).

Revision: `0f7e7420e84c6789d96ba465608d2a008ecb443d`.

Проверено по конкретному классу, а не defaults всего семейства: 30 слоёв, d=2816; 128 routed experts на слой, top-8; expert_dim=704, shared dense MLP intermediate=2112. Attention pattern: пять local sliding слоёв и один global, local window=1024. Local KV heads=8 при head_dim=256; global KV heads=2, global key size=512 и `k_eq_v_global=True`. Без per-layer input embedding у этого класса (`per_layer_input_dim=0`).

Важно: PLE и KV-sharing из других конфигураций семейства нельзя автоматически приписывать 26B-A4B. Экономия хранения K=V требует соответствующего engine/storage, а не только равенства проекций.

[Реализация MoE](https://github.com/google-deepmind/gemma/blob/0f7e7420e84c6789d96ba465608d2a008ecb443d/gemma/gm/nn/gemma4/_moe.py) подтверждает top-k routing, перенормировку весов и gated expert MLP. Здесь GELU-gated MLP; выбранный нами SwiGLU — собственное решение, а не копия Gemma.

## GPT-OSS

[Официальный README OpenAI](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/README.md).

Revision: `7b583341fe16729127f6d5b94a7b09ccae97e1a1`.

README указывает для 20B около 21B общих и 3.6B активных параметров, MXFP4-квантование MoE-весов и вариант запуска в пределах 16 GB. Это результат конкретной модели/backend, не обещание для новой архитектуры или произвольного контекста/батча. Публичные веса требуют правильного Harmony format.

[Официальная PyTorch reference implementation](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py) показывает чередование sliding/full layers, GQA, attention sinks, router и SwiGLU experts. Defaults `ModelConfig` относятся к большому варианту; не используем их как подтверждение точного числа слоёв/экспертов checkpoint 20B. README отдельно предупреждает о большом расходе памяти неоптимизированной reference implementation, разворачивающей веса в BF16.

## Qwen3.8-Flash-Next — точный референс

[Официальный README](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/README.md).

[Technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf).

Revision: `69885871a64393807d988b27b1b5e380e8f28526`.

Проверено: модель выпущена 26 августа 2026 как preview архитектуры Qwen4. Заявлены 125B backbone parameters, дополнительные 51B n-gram embeddings и 6B activated parameters/token. Для общего capacity учитывать обе части: ≈176B, а не только 125B; точное сравнение active-count требует сверки соглашений.

- Report §2.1.1, pp. 3–4: GDN + periodic full attention 3:1; фиксированная recurrent matrix, decay/write gates, bounded sigmoid output gate, RoPE в full layers. GDN не даёт точного хранения всех прежних токенов.
- §2.1.2, pp. 5–10: full layers заменяются на QSA во время continued pretraining на 256K; compressed MQA indexer выбирает micro-blocks, обучение проходит через dense warm-up и sparse adaptation. Индексирование prefill `O(T²/r)`, а не `O(T)` при фиксированном r. Снижение attention computations не означает устранения полной истории KV.
- §2.2: Gated Residual расширяет residual stream до четырёх ветвей. Дополнительная ёмкость/traffic и kernel engineering должны учитываться отдельно. В v0.2 GR включён в основной граф; рассмотрение и параметризация описаны в [residual-design.md](residual-design.md).
- §2.3.1, pp. 14–15: один n-gram embedding layer на Layer 2; размещение даёт overlap prefetch с первым блоком. Разделение фиксированного бюджета на несколько слоёв не приносит устойчивого улучшения downstream.
- §2.3.2, p. 15: уменьшение MoE experts ради n-gram table при фиксированном общем бюджете не показывает ясного downstream выигрыша. При добавлении памяти без уменьшения MoE loss улучшается, но отдельные benchmarks насыщаются/флуктуируют.
- §3: разделение Muon и Adam/AdamW по типам параметров; таблицы n-gram не следует автоматически оптимизировать как плотные линейные матрицы.

Эти результаты не доказывают преимущество любого собственного MoE+Engram варианта. Требуются отдельные ablations и checks редактирования.

## Engram

[Официальный репозиторий DeepSeek](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/README.md).

[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/Engram_paper.pdf).

[Demonstration implementation](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py).

Revision: `fb7f84a21f91223715394a33a1dc24bbfb7f788e`.

Проверены архитектурные разделы и experiments: suffix n-grams, multi-head hashing для снижения коллизий, context-aware gating, causal short convolution и deterministic host prefetch. В Engram-27B используются 2/3-граммы и два memory layers. Обученные hash vectors не являются явными фактами, а разрешение коллизий не становится точным только за счёт нескольких heads.

Результат paper §6.4: offload 100B-parameter table с малой потерей throughput получен на H800, dense backbones, 512 sequences и коротких последовательностях Uniform(100,1024). Его нельзя переносить на любой consumer GPU, batch=1, большой prefill или CPU-training таблиц. Demo явно имитирует стандартные Attention/MoE/mHC components и не является готовой end-to-end LLM.

В прочитанных источниках не найдено доказательство универсальной корректности прямого CRUD над n-gram vectors как способа замены/удаления мировых знаний. Вариант exact-key mutable overlay был рассмотрен как возможное собственное расширение; по просьбе пользователя подробный дизайн редактирования отложен и не является утверждённой частью архитектуры.

## Residual-схемы

[DeepSeek-V4.1-Flash review](deepseek-v41-review.md), добавлено 11 октября: pinned card/config/reference code подтверждают Single-Pass mHC, explicit global KV ownership, Engram и routing. В next-moe v0.3 переносится ownership/sharing для пар full GQA; GR остаётся baseline. Полный V4.1 PDF недоступен после CDN redirect; его ablations не заявляются прочитанными. Revision: `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`.

[Подробный выбор GR](residual-design.md) опирается на Flash-Next §2.2 и [implementation Qwen4ExpTextGatedResidual/PLELayer/DecoderLayer](https://github.com/huggingface/transformers/blob/927af2f73bd0d53d339f8fc0d4bd8b053f3836bb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py). Проверены shape contracts, branch-wise normalization, sigmoid read/write, placement памяти до attention и final read-only collapse. Код статически прочитан, не запускался.

[Flash-Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) подтверждает n=4 и gate rank=d/8. [Официальный AttnRes paper](https://github.com/moonshotai/Attention-Residuals/blob/85e22310fe5ee860b4a023de312d791de8a5a5e6/Attention_Residuals.pdf) прочитан в архитектурных разделах для сравнения механизмов и memory costs. mHC рассмотрен по авторскому abstract и сравнению в Flash-Next; полный mHC paper непосредственно не проверен.

## Дополнительные источники для исследований

- [Современные модели разных размеров](model-size-references.md): 27 официальных model-card/config snapshots от LFM2.5-230M до Kimi K3; подтверждены дополнительные размеры Qwen3.5/3.6/3.8, Nemotron 3/3.5, GLM-5.3-Flash, DeepSeek V4/V4.1 и Kimi K2.7-Code. Revision IDs сохранены в [reference-models.json](../configs/reference-models.json). Сопоставление используется для [десяти предлагаемых профилей](model-family.md); качество и VRAM не выводятся из чужих benchmark scores.
- [Обзор размещения n-gram модулей](ngram-placement-review.md): добавлены первичные код/карточка LongCat-Flash-Lite, код BLT, Google N-grammer/Praxis и авторские abstracts SCONE/OverEncoding. Количественное решение по одному/двум модулям опирается прежде всего на прямые matched-budget ablations Engram §6.2 и Flash-Next §2.3.1; наличие многих таблиц не считается множеством модулей по глубине.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/blob/855e502613d47d8cc139153f0d87ecae5831ac3c/README.md): kernel/reference ecosystem; README содержит ссылку на [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464). Прочитан README; paper по этой ссылке напрямую здесь не проверялся.
- [ROME](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/README.md) и [MEMIT](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/README.md): прочитаны README как references для будущих efficacy/locality/paraphrase benchmarks. Эти методы редактируют нейросетевые веса и не являются реализацией выбранного direct Engram CRUD.

## Текущий доступ к источникам

Ранее добавление `huggingface.co` было сохранено в сетевой черновик. В последнем исследовании реальные GET к HF API и текстовым model files успешны; это наблюдение доступа, а не доказательство публикации конфигурации. Прямые arXiv/OpenReview/PMLR запросы и redirect скачивания LongCat PDF возвращают tunnel 403. LongCat layer-ablation из полного paper не заявляется проверенной; SCONE и OverEncoding рассмотрены на уровне авторских abstracts. Архитектурные сведения BLT/N-grammer проверены по code, а количественные one-vs-two результаты — по доступным официальным Engram/Flash-Next PDF.

11 октября V4.1 card/config и reference model/engram/kernel files доступны; PDF redirect на `us.aws.cdn.hf.co` получает tunnel 403. Вместо предположений о содержимом paper проверены опубликованный code и packed-cache arithmetic. [Cross-Layer Attention](https://huggingface.co/papers/2405.12981) дополнительно проверен только по авторскому abstract; его 1B/3B results не считаются validation нашего GDN/GQA варианта.
