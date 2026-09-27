"""Пример 3 урока 4: настраиваемая модель.
"""

import os

from langchain.chat_models import init_chat_model

from course_model import gateway_kwargs

MODEL_NAME = os.environ["MODEL_NAME"]
QUESTION = "Ответьте одним словом: столица Франции?"

# 1. Модель без имени. Имя и провайдер становятся настраиваемыми по умолчанию,
#    а остальные параметры остаются такими, какими заданы здесь.
configurable = init_chat_model(temperature=0, **gateway_kwargs())

print("ВЫЗОВ С ИМЕНЕМ МОДЕЛИ В КОНФИГУРАЦИИ")
response = configurable.invoke(QUESTION, config={"configurable": {"model": MODEL_NAME}})
print(f"  ответ: {response.text.strip()!r}")
print(f"  модель в ответе: {response.response_metadata.get('model_name')}")
print()

# 2. Тот же объект без конфигурации. Имени модели взять неоткуда.
print("ТОТ ЖЕ ОБЪЕКТ БЕЗ КОНФИГУРАЦИИ")
try:
    configurable.invoke(QUESTION)
except Exception as error:
    print(f"  {type(error).__name__}: {str(error)[:200]}")
else:
    print("  вызов прошёл: у вашей сборки имя модели откуда-то взялось")
print()

# 3. Настраиваемые поля перечислены явно, ключи конфигурации получили префикс.
#    Модель по умолчанию задана, поэтому вызов без конфигурации работает.
answers = init_chat_model(
    model=MODEL_NAME,
    configurable_fields=("model", "max_tokens"),
    config_prefix="answer",
    temperature=0,
    max_tokens=32,
    **gateway_kwargs(),
)

LONG_QUESTION = "Опишите очередь задач в трёх предложениях."

default_run = answers.invoke(LONG_QUESTION)
print("ПО УМОЛЧАНИЮ, MAX_TOKENS=32")
print(f"  выход: {(default_run.usage_metadata or {}).get('output_tokens')} токенов")
print()

with_config = answers.invoke(
    LONG_QUESTION,
    config={"configurable": {"answer_max_tokens": 300}},
)
print("ПЕРЕОПРЕДЕЛЕНИЕ НА ВЫЗОВ, ANSWER_MAX_TOKENS=300")
print(f"  выход: {(with_config.usage_metadata or {}).get('output_tokens')} токенов")
print()

# 4. Ключ без префикса до модели не доходит, и молча: ошибки не будет.
without_prefix = answers.invoke(
    LONG_QUESTION,
    config={"configurable": {"max_tokens": 300}},
)
print("КЛЮЧ БЕЗ ПРЕФИКСА, MAX_TOKENS=300")
print(f"  выход: {(without_prefix.usage_metadata or {}).get('output_tokens')} токенов")
print("  столько же, сколько по умолчанию: ключ не подошёл под префикс")
