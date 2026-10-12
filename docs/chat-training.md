# Как превратить base-модель в чат

Записано 12 октября 2026. Текущий [Colab workflow](colab-training.md) делает только короткое **обучение продолжению текста с нуля** на UK/EN Wikipedia. Это тест пайплайна, а не chat training. SFT/DPO и dialog template ещё не реализованы.

## Этапы

1. **Pretraining base-модели.** Обучение next-token prediction на хорошем UK/EN корпусе, с полноценным бюджетом, валидацией и проверкой языковых навыков. Wikipedia подходит для первого технического теста, но 153 600 processed tokens и небольшой корпус не создают качественную языковую модель. Перед SFT требуется увеличить разнообразие и объём данных; зафиксировать tokenizer/Engram hashing и базовый checkpoint.
2. **Supervised fine-tuning (SFT).** Продолжить обучение того же backbone на качественных UK/EN диалогах и instruction/answer примерах, включая multi-turn. Loss считать по содержимому ответов ассистента и его end-of-turn marker; system/user spans маскировать через `ignore_index=-100`. Вход по-прежнему causal: user text участвует в вычислениях и формировании hidden states, но не становится target для assistant-only objective. Начать с полного fine-tuning малого профиля с меньшим LR; freezing/table-only/LoRA исследовать отдельно, не предполагать автоматически.
3. **Оценка chat behavior.** Проверить выполнение инструкций, роли, завершение ответа, follow-up сообщения, качество UK/EN, factuality и сохранение базовых языковых навыков. Разделить train/validation/test по разговорам/источникам; не считать только уменьшающийся SFT loss доказательством полезного чата. Сравнивать с base checkpoint и сохранять data provenance.
4. **Предпочтения — при необходимости.** DPO или другой preference training на парах предпочтительных/непредпочтительных ответов для одинакового prompt. Это необязательный последующий этап после работающего SFT, требующий отдельного набора и контроля качества. Учитывать память reference policy и не переносить на DPO оценки VRAM обычного single-model pretraining без пересчёта.

## Формат диалога и tokenizer

Выбрать один сериализованный template с ролями `system`, `user`, `assistant` и явным end-of-turn. И training examples, и интерфейс inference используют **точно тот же** template. На inference prompt заканчивается началом assistant message; generation прекращается на его end-of-turn marker. Историю диалога явно передавать в prompt: память прошлой Colab ячейки сама по себе не является chat history модели.

В текущем tokenizer зарезервированы `<pad>/<bos>/<eos>/<unk>`, а chat-specific IDs ещё не выбраны. Перед полноценным pretraining желательно зарезервировать role/end-of-turn tokens из существующих 65 536 rows и сохранить их IDs. Для уже обученного checkpoint добавление role tokens допустимо только без перенумерации прежних token IDs, с проверкой tokenizer compatibility и свободных embedding rows; нужен отдельный warm-up/SFT. **Не retrain BPE заново:** это изменит Engram hash addresses и соответствие весов token IDs. Текущее strict resume проверяет неизменность tokenizer и не заменяет специальный переход base→SFT.

## Корпус диалогов

Конкретный SFT dataset ещё не выбран и не скачивается текущим notebook. При выборе проверить источник, license, качество, долю украинского/английского, наличие multi-turn и дедупликацию. Не считать machine-translated Ukrainian автоматически качественным; проверять естественность и фактическую точность. Отдельно включить полезные запросы и хорошие примеры ответов для задач пользователя; объём определить по эксперименту, не обещать качество из одного числа диалогов.

Сначала стабилизировать текущий test run, затем подготовить самостоятельный SFT workflow с template, assistant-only mask, chat inference и сохранением base/SFT checkpoints. Chat template сам по себе не обучает следовать инструкциям; короткий SFT не заменяет достаточное pretraining.
