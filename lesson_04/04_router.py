"""Пример 4 урока 4: одна настраиваемая модель на три разные задачи.

MODEL_NAME_STRONG, это необязательная переменная курса. Есть у вашего
провайдера вторая модель, положите её имя рядом с MODEL_NAME. Нет, пример
отработает на одной модели и скажет об этом.
"""

import os

from langchain.chat_models import init_chat_model

from course_model import gateway_kwargs

FAST_MODEL = os.environ["MODEL_NAME"]
STRONG_MODEL = os.getenv("MODEL_NAME_STRONG") or FAST_MODEL

TASKS = [
    (
        "классификация",
        "Отнесите обращение к одной категории: оплата, доставка, возврат. "
        "Обращение: деньги списали дважды. Ответьте одним словом.",
        {"model": FAST_MODEL, "max_tokens": 16},
    ),
    (
        "короткая справка",
        "Что такое очередь задач? Одно предложение.",
        {"model": FAST_MODEL, "max_tokens": 120},
    ),
    (
        "разбор с доводами",
        "Сравните очередь задач и стек по трём признакам, с доводами.",
        {"model": STRONG_MODEL, "max_tokens": 400},
    ),
]

router = init_chat_model(
    configurable_fields=("model", "max_tokens"),
    temperature=0,
    **gateway_kwargs(),
)

if STRONG_MODEL == FAST_MODEL:
    print("MODEL_NAME_STRONG не задана: обе роли идут на одну модель.")
    print("Маршрутизация от этого не ломается, но разницу видно только в параметрах.")
    print()

for title, prompt, settings in TASKS:
    response = router.invoke(prompt, config={"configurable": settings})
    usage = response.usage_metadata or {}
    print(f"{title.upper()}")
    print(f"  настройки: {settings}")
    print(f"  модель в ответе: {response.response_metadata.get('model_name')}")
    print(f"  вход {usage.get('input_tokens')}, выход {usage.get('output_tokens')}")
    print(f"  ответ: {response.text.strip()[:100]!r}")
    print()
