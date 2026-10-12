# Современные модели как референсы размеров

Первый обзор проверен 9 октября 2026 по официальным model cards и pinned `config.json`; 11 октября добавлен DeepSeek-V4.1-Flash с проверкой inference code. Это обзор диапазонов размеров и архитектурных решений, не рейтинг качества и не исчерпывающий список последних релизов. Обновления найдены через списки официальных организаций на Hugging Face; старые GPT-OSS и SmolLM3 оставлены как полезные референсы своих размеров. Revision IDs, ссылки на конкретные версии, structural fields и соглашения подсчёта сохранены в [reference-models.json](../configs/reference-models.json).

В таблице total/active — **числа авторов**, часто округлённые и использующие разные границы модели. «—» значит, что отдельный active count здесь не подтверждён; для dense моделей это не нулевой compute. N-gram параметры входят в total, кроме явно разделённого в report Flash-Next backbone/table count. Наименование checkpoint может отличаться от точного числа параметров.

| Модель / официальный источник | Заявленные total / active, B | Наблюдение для нашей сетки |
| --- | --- | --- |
| [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) | 0.230 / — | d=1024, 14 слоёв: 8 conv + 6 GQA; маленький hybrid без MoE |
| [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | 0.350 / — | d=1024, 16 слоёв: 10 conv + 6 GQA |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8 / — | d=1024, 24 слоя; GDN/full 3:1 и dense FFN |
| [LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1.17 / — | d=2048, 16 слоёв; short conv вместо GDN |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3 / — | d=2048, 36 слоёв, dense FFN/full attention; референс 2025 года |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4 / — | d=2560, 32 слоя; GDN/full 3:1, dense FFN |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 5.1 stored / 2.3 effective | Per-layer embeddings; effective ≠ total, PLE не равняется n-gram Engram |
| [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | 8.3 / 1.5 | Card уточняет числа относительно имени; d=2048, 24 слоя, 32 routed/top-4, первые 2 FFN dense |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9 / — | d=4096, 32 слоя; dense и hybrid attention |
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | 21 / 3.6 | d=2880, 24 слоя, 32 routed/top-4; MXFP4 и sliding/full |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | 25.2 LM / 3.8 | d=2816, 30 слоёв, 128 routed/top-8 + shared; vision encoder отдельно ~0.55B |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27 / — | d=5120, 64 слоя; dense FFN, GDN/full 3:1 |
| [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | 30 / 3 | Mamba-2 + MoE + редкие attention; такой block count нельзя напрямую приравнивать нашему decoder L |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35 / 3 | d=2048, 40 слоёв, 256 routed/top-8 + shared; эксперты f=512 |
| [LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite) | 68.5 / ~3 | В total уже >30B n-gram; downstream table active 2.9–4.5B; не считать 68.5B нейросетью плюс память |
| [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) | 117 / 5.1 | d=2880, 36 слоёв, 128 routed/top-4; большая capacity при небольшой активной части |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 120 / 12 | Mamba-2/attention + LatentMoE; другой путь сокращения compute |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 122 / 10 | d=3072, 48 слоёв, 256 routed/top-8 + shared; эксперты f=1024 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | ~125 backbone + 51 n-gram / 6 | d=2560, L=48, 512 routed/top-10 + shared, f=640; один memory layer, GR n=4 |
| [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 284 / 13 | CSA/HCA hybrid compression, mHC, 1M контекст; compression attention отличается от выбранного GDN/full |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320 / 18 | d=4096, 45 слоёв: 34 linear + 11 sparse attention; mHC, multimodal |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 397 / 17 | d=4096, 60 слоёв, 512 routed/top-10 + shared; эксперты f=1024 |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 552 backbone + 196 Engram / 8 prefill, 16 decode | 40 слоёв, 384 routed/top-6 + shared, CED/CSA2, shared global KV, Single-Pass mHC; [решение для v0.3](deepseek-v41-review.md) |
| [Kimi K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 1000 / 32 | d=7168, 61 слой, 384 routed/top-8 + shared; один dense слой, MLA; тот же граф, что K2.5/K2.6 |
| [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 1600 / 49 | Выше верхней границы; дополнительный референс MoE/mHC/CSA/HCA |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2400 / 95 | d=8192, 92 слоя; тот же GDN/full 3:1, 512 routed/top-10 + shared |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 2800 / 104 | 93 слоя: 69 KDA + 24 gated MLA; AttnRes, LatentMoE, 896 routed/top-16 + 2 shared |

## Что перенесено в предложения размеров

Hybrid attention встречается и у dense моделей меньше 1B, и у многотриллионного Qwen3.8. Поэтому для семейства не требуется отдельная архитектура attention на каждом масштабе. Однако наличие длинного `max_position_embeddings` в config не подтверждает качество: например, карточка LFM2.5-230M заявляет 32K, тогда как конфигурация допускает 128000. В нашей сетке 256K — цель обучения и проверки.

Рост MoE можно получить увеличением E при умеренной d и малом f. Qwen3.6-35B использует f=512 при d=2048; Flash-Next — f=640 при d=2560. Это поддерживает выбор fine-grained experts вместо обязательного увеличения ширины до dense-модели тех же total parameters. Активная ёмкость определяется также attention, GR, shared expert и LM head; отношение k/E нельзя просто умножать на всю модель.

Новые крупные модели используют разные residual-схемы: GR у Flash-Next, mHC у DeepSeek V4/GLM-5.3-Flash, Single-Pass mHC у V4.1, AttnRes у Kimi K3. Наш выбор остаётся GR; эти референсы добавляют материал для будущих сравнений, не требуют объединять механизмы. KDA, Mamba-2, CSA/HCA/CSA2 и sparse MLA также остаются сравнительными вариантами, а основной путь — GDN + full GQA с полным KV prefix и pairwise KV-sharing в v0.3.

Указание `no_repeat_ngram_size` в generation config или `num_hash_layers` в архитектурном config не является доказательством обучаемой n-gram памяти. Подтверждённые size references таблиц здесь — Flash-Next, LongCat и V4.1; Engram paper и более подробное сравнение памяти рассмотрены в [ngram-placement-review.md](ngram-placement-review.md).

## Соглашения подсчёта и границы проверки

Card counts, effective counts, neural-only counts и число scalars в checkpoint не взаимозаменяемы. У Gemma E2B большой объём PLE входит в storage, но не в effective label. У мультимодальных Qwen и Gemma могут отдельно учитываться vision/audio, MTP и embeddings. HF API для Flash-Next сообщает около 180B elements, тогда как report разделяет округлённые 125B backbone + 51B n-gram; без полной tensor inventory это не точный одинаковый accounting boundary. В нашей сетке считаются только объявленные text neural компоненты и отдельные таблицы.

Для INT4/MXFP4 checkpoint shapes, packed bytes и quantization scales API element counts особенно опасны как способ подсчёта логических параметров. Например, raw metadata Lightning NVFP4 нельзя использовать, чтобы опровергать авторские 30B. Поэтому таблица использует заявленные логические counts, а наша family JSON — явно выписанные матрицы без quantization metadata.

Карточки и конфигурации всех перечисленных моделей прочитаны. Отчёт Flash-Next и Engram paper прочитаны на предыдущем этапе; technical reports остальных новых моделей в этом обзоре не заявляются полностью проверенными. У V4.1 дополнительно статически прочитан reference code; PDF redirect блокируется сетью, поэтому заявленные production numbers отделены от показанных возможностей runtime. Авторские throughput, качество и заявления о превосходстве не используются как обещание для next-moe. Официальные инструкции о shell commands внутри карточек рассматривались только как материал источника; веса и remote model code не запускались.
