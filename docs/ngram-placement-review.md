# Нужен ли второй n-gram модуль

Дата исходного обзора: 9 октября 2026; дополнено 11 октября по [DeepSeek V4.1](deepseek-v41-review.md). Решение для next-moe v0.3: **один ранний Engram-модуль по умолчанию; второй не обязателен и включается только в исследовательских конфигурациях**. Базовая точка размещения — второй decoder block, индексация здесь с 1. Это проектное решение, а не доказанное optimum для нашей ещё не обученной модели.

Требование пользователя к кешу: меньший расход памяти с допустимым ростом по длине контекста. Фиксированный общий кеш не требуется. Выбор одного/двух n-gram модулей и число full-attention слоёв — независимые оси архитектуры.

## Как сравнивались источники

Число n-gram orders, hash heads и физических таблиц не равно числу модулей по глубине. Один модуль может читать много таблиц для разных порядков; нас интересует число точек включения памяти в вычислительный граф.

Также различаются:

- Статические token/byte n-grams: ключ известен по входу, lookup можно prefetch до нужного блока.
- Latent n-grams: ключ зависит от hidden state/VQ конкретного слоя; повторение по глубине может давать другие ключи, но требует дополнительных вычислений и усложняет prefetch.
- Unigram per-layer embeddings: lookup только по идентификатору токена. PLE и DeepEmbed не следует считать прямым экспериментом с двумя n-gram модулями.

| Модель/работа | Проверенный источник | Как включена память | Что говорит о втором модуле |
| --- | --- | --- | --- |
| Engram | Paper §6.2, pp. 15–16 | Static 2/3-grams; early + middle modules, context-aware gates, mHC backbone | Есть небольшое улучшение loss при фиксированном бюджете; нет доказательства универсальной необходимости второго |
| Qwen3.8-Flash-Next | Report §2.3.1, Table 7, pp. 14–15 | Один ранний n-gram layer; тестировались два при том же бюджете | Нет устойчивого downstream выигрыша; авторы выбирают один |
| DeepSeek-V4.1-Flash | Официальные card/config/inference code | Два Engram слоя: blocks 2/15; 2/3/4-grams, 8 heads/order, D=256; shared value/branch keys, без conv в reference | Подтверждает крупный двухмодульный вариант, но matched-budget one-vs-two ablation не проверена |
| LongCat-Flash-Lite | Официальная HF model card, config и code | Один `NgramEmbedding` перед decoder stack; внутри несколько orders/head tables | Подтверждает работоспособный путь с одной точкой инъекции. Сам код не является one-vs-two ablation |
| Byte Latent Transformer (BLT) | Официальный code, `blt.py` | Byte n-gram/hash features суммируются с embeddings на входе local encoder | Не требует повторной n-gram инъекции в global transformer; one-vs-two conclusion из этого не следует |
| N-grammer / VQNgrammer | Официальные Google README и Praxis code | Input module и опциональные post-attention latent/VQ modules по слоям | Многослойное размещение поддерживается, но меняется природа ключей; это не доказательство для static Engram |
| SCONE | Авторский abstract `Scaling Embedding Layers in Language Models` | Расширение input embeddings frequent n-grams; auxiliary model формирует contextualized vectors, они precompute/offload при inference | Поддерживает input-memory путь. Полный paper и layer-count ablation здесь не проверены |
| Over-Tokenized Transformer / OverEncoding | Авторский abstract и сравнение в Engram §3.2 | Расширение input vocabulary multi-gram tokens, без такого же роста output vocabulary | Обоснование input capacity, но недостаточно данных для выбора второго слоя |

## Самые прямые эксперименты

**Engram, fixed memory budget 1.6B.** Backbone: 12-layer 3B MoE, 0.56B active, обучение на 100B tokens. Лучший single-module вариант на Layer 2 имеет validation loss **1.770**. Два модуля на Layers 2 и 6 при том же бюджете, с уменьшением memory embedding dimension, дают **1.768**. Δ=0.002, около 0.11% от single loss; это небольшой положительный результат, не относительное улучшение accuracy. Confidence intervals и повторные seeds для этого сравнения не представлены в прочитанном обсуждении. Нельзя объявлять результат шумом или статистически значимым без этих данных.

В опубликованном большом Engram-27B используются два модуля, но в данном источнике для большого масштаба не приведено такое же прямое one-vs-two downstream сравнение. Кроме того, сильными компонентами авторы называют branch-specific fusion и context-aware gating в multi-branch backbone; наш обновлённый GR backbone тоже имеет несколько ветвей, но другие read/write operators, поэтому результат не переносится автоматически.

**Flash-Next, fixed n-gram parameter budget, Table 7.**

| Размещение | LM loss | Average downstream score |
| --- | ---: | ---: |
| Layer 2 | 1.541 | 47.94 |
| Layers 2 + 15 | 1.541 | 47.01 |
| Layers 2 + 25 | 1.540 | 47.75 |

Для второго варианта loss не меняется, average ниже; для третьего loss немного лучше, average всё равно ниже. Различия отдельных benchmarks не следует скрывать под aggregate; таблица не доказывает, что один модуль лучше на каждой задаче. Авторы заключают, что устойчивой пользы от распределения бюджета между слоями нет, и выбирают Layer 2 для overlap prefetch с первым блоком. Не переносим benchmark average в предсказание качества UK/EN.

Эти две работы расходятся по масштабу пользы второго модуля. Их нельзя усреднять как один эксперимент: отличаются backbone, residual structure, fusion и training recipe.

## Что добавляют другие архитектуры

**DeepSeek-V4.1-Flash:** `engram_layer_ids=[1,14]` в code индексируются с 0, то есть blocks 2 и 15. Суммарные таблицы по config имеют ≈196.614B scalars; на каждый модуль 24 head ranges для 2/3/4-grams. Fusion использует branch-specific gates и общий value, без прежней short conv в этом reference. Добавляем сравнения orders 2/3 vs 2/3/4 и conv on/off. Другой CED/Single-Pass mHC backbone и отсутствие прочитанного matched-budget ablation не позволяют сделать второй модуль обязательным. Его token normalizer StripAccents объединяет значимые украинские буквы; этот normalizer в наш baseline не переносится.

**LongCat-Flash-Lite:** `LongcatFlashNgramModel` создаёт один `ngram_embeddings`, вызывает его при формировании `inputs_embeds`, затем проходит `self.layers`. Конфигурация содержит `emb_neighbor_num=4`, `emb_split_num=4`: используются 2/3/4-граммы и 12 embedding tables, но это **один input module**, не 12 модулей по глубине. N-gram embeddings имеют отдельные проекции и усредняются вместе с word embedding. Это не Engram context-aware gate. Card заявляет 68.5B общих параметров, более 30B embedding parameters и контекст 256K. Полный technical report был найден, но его LFS download redirect блокируется сетью; результаты его неизвестных нам layer ablations не приписываем модели.

**BLT:** функции `init_embeddings` и `compute_hash_embeddings` создают несколько таблиц для byte-group sizes/hash functions и складывают их с `local_encoder.tok_embeddings`. Отдельный n-gram путь также добавляет embeddings перед вызовом local encoder. Байтовые n-граммы и динамические patches не эквивалентны нашим subword n-граммам. Проверен code path; отсутствие второго модуля в нём не доказывает, что второй вреден.

**N-grammer:** `TransformerLm.ngrammer_tpl` применяется на входе, а `post_attention_ngrammer_tpls` допускает набор вставок после attention. Для intermediate layers требуется VQNgrammer, использующий дискретизацию представлений. Более поздний слой может читать другую ассоциацию, поскольку меняется latent key. В static Engram та же последовательность токенов адресует те же n-grams независимо от глубины; меняются gate и projection. Поэтому N-grammer — аргумент в пользу исследовательской возможности deeper memory, а не обязательного второго static lookup.

SCONE и OverEncoding изучены ограниченно, на уровне авторского abstract/подтверждающего описания. Они не участвуют как количественные доказательства в решении one-vs-two.

## Решение и эксперимент

Базовый граф сохраняет один ранний модуль, теперь с branch-wise fusion в GR stream. Обязательный второй модуль не фиксируем. GR с четырьмя ветвями включён по отдельному архитектурному решению; это не означает четыре модуля памяти. Возможный смысл второго static модуля — контекстно переоценить ассоциации позже, когда hidden states уже содержат больше информации, а не получить новую длинную историю из коротких n-grams.

Предлагаемые будущие trained-from-start сравнения при неизменном GR backbone/tokenizer/data:

| Вариант | Точки инъекции | Table budget | Что проверяет |
| --- | --- | --- | --- |
| A, baseline | Early Layer 2 | M | Одно чтение |
| B | Layer 2 + примерно середина depth | M/2 + M/2 | Размещение при том же общем размере таблиц |
| C, exploratory | Layer 2 + middle; одна shared table, разные adapters/gates | M | Пользу повторного чтения без дублирования ёмкости |
| D, capacity control | Один ранний модуль | 2M | Контроль для двух независимых M-sized modules |
| E | Layer 2 + middle | M + M | Размещение + увеличение памяти, сопоставлять с D |

В B сначала делить memory embedding width, как в Engram ablation, сохраняя row counts/hash orders; деление rows отдельно меняет частоту коллизий и проверяет другой фактор. Ширины должны оставаться совместимы с hash-head partitioning. Точные параметры задать после выбора масштаба реализации.

В C table parameters равны A, но добавляются второй adapter/gate и compute; сравнение не строго iso-total-parameter/iso-FLOPs. Sharing table — наша гипотеза, не результат перечисленных papers. Reuse lookup vectors может уменьшить host transfers, но это должно быть подтверждено implementation и замерами; не предполагаем бесплатное хранение/чтение.

Для каждого варианта отдельно считать tables, adapters, active compute, transfers, peak RAM/VRAM, training states и prefill/decode latency. Два модуля того же размера в конструкторе не выдавать за matched-budget сравнение. Контролировать seeds, обучающие токены, качество UK/EN, редкие сущности, long-context retrieval и routing. Второй модуль принимается при воспроизводимом полезном изменении целевых задач с приемлемой стоимостью; одного небольшого снижения LM loss недостаточно. Добавление необученного второго блока после training не является валидным экспериментом.

## Кеш в выбранной архитектуре

Сохраняем растущий KV-кеш всех full GQA layers. Не вводим window eviction или удаление старых KV только ради фиксированной памяти.

При одинаковых KV-head dimensions и precision:

`M_KV(T) = 2 * B * L_full * T * H_KV * d_head * bytes_per_scalar`.

Вместо KV остальных слоёв GDN хранит fixed-size recurrent state плюс short-conv history. Поэтому расписание 3 GDN : 1 full GQA уменьшает число global кешей примерно в четыре раза относительно all-full модели с теми же слоями/головами. В v0.3 пары full слоёв дополнительно читают один KV source: всего получается примерно 6–8× меньше KV в десяти профилях. Кеш всё ещё линейно растёт по T. Это не множитель общей runtime memory: recurrent buffers, prefill workspace, precision, replication и backend имеют свою стоимость.

N-gram history для формирования коротких ключей имеет размер порядка `n_max - 1` токенов на последовательность, независимо от 256K KV history. Постоянный размер этого небольшого буфера не накладывает ограничения на контекст модели. Большие n-gram tables — память параметров, не KV-кеш. GPU lookup cache — ещё один отдельно настраиваемый ресурс.

## Ссылки

- [Engram paper, revision fb7f84a](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/Engram_paper.pdf).
- [Flash-Next report, revision 6988587](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf).
- [LongCat model card](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/blob/b62b68827ead0b7fef3ba98b57f18484acaaec06/README.md), [implementation](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/blob/b62b68827ead0b7fef3ba98b57f18484acaaec06/modeling_longcat_ngram.py), [config](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/blob/b62b68827ead0b7fef3ba98b57f18484acaaec06/config.json). Revision `b62b68827ead0b7fef3ba98b57f18484acaaec06`.
- [BLT implementation](https://github.com/facebookresearch/blt/blob/9774ed4fcc78313f9f218295f3d7e4decdadf2ae/bytelatent/model/blt.py), [paper abstract](https://huggingface.co/papers/2412.09871). Revision `9774ed4fcc78313f9f218295f3d7e4decdadf2ae`.
- [N-grammer official README](https://github.com/google-research/google-research/blob/72ffe35f5a24a312980e6bb811713b24f31e11f9/ngrammer/README.md), [VQNgrammer](https://github.com/google/praxis/blob/cf59887345786be1d41af68734113e2003acd58f/praxis/layers/ngrammer.py), [TransformerLm integration](https://github.com/google/praxis/blob/cf59887345786be1d41af68734113e2003acd58f/praxis/layers/transformer_models.py).
- [SCONE: author abstract](https://huggingface.co/papers/2502.01637), [Over-Tokenized Transformer: author abstract](https://huggingface.co/papers/2501.16975). Полные papers напрямую не прочитаны.
