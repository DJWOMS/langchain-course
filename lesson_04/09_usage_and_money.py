"""Пример 9 урока 4: расход и деньги, когда моделей в приложении больше одной.
"""

import os

from langchain_core.callbacks import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)

from course_model import build_model

# ПОДСТАВЬТЕ СВОИ СТАВКИ: доллары за миллион токенов, строка на модель.
# Имя модели пишется так, как оно приходит в ответе провайдера.
RATES_BY_MODEL = {
    # "имя-модели-как-в-ответе": {"input": 0.0, "input_cached": 0.0, "output": 0.0},
}

# Запасные ставки: модель курса на 22.09.2026, ночной тариф, как в уроке 1.
# Ими считается модель, которой нет в RATES_BY_MODEL, её строка в счёте помечается.
DEFAULT_RATES = {"input": 0.15, "input_cached": 0.003, "output": 0.60}

FAST_NAME = os.environ["MODEL_NAME"]
STRONG_NAME = os.getenv("MODEL_NAME_STRONG") or FAST_NAME

fast = build_model(FAST_NAME, temperature=0, max_tokens=64)
strong = build_model(STRONG_NAME, temperature=0, max_tokens=256)

TASKS = [
    (
        "классификация",
        fast,
        "Отнесите обращение к одной категории: оплата, доставка, возврат. "
        "Обращение: деньги списали дважды. Ответьте одним словом.",
    ),
    (
        "короткая справка",
        fast,
        "Что такое очередь задач? Одно предложение.",
    ),
    (
        "разбор с доводами",
        strong,
        "Сравните очередь задач и стек по трём признакам, с доводами.",
    ),
]

callback = UsageMetadataCallbackHandler()

print("ПРОГОН ТРЁХ ЗАДАЧ")

for title, model, prompt in TASKS:
    response = model.invoke(
        prompt,
        config={
            "callbacks": [callback],
            "run_name": title,
            "tags": ["lesson-04", "router"],
            "metadata": {"task": title},
        },
    )
    usage = response.usage_metadata or {}
    print(f"  {title:<20} выход {usage.get('output_tokens')} токенов")

print()
print("РАСХОД ПО МОДЕЛЯМ")

total_money = 0.0

for model_name, usage in callback.usage_metadata.items():
    rates = RATES_BY_MODEL.get(model_name)
    guessed = rates is None
    rates = rates or DEFAULT_RATES
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    fresh = usage["input_tokens"] - cached
    money = (
        fresh * rates["input"]
        + cached * rates["input_cached"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000
    total_money += money

    print(f"  модель: {model_name}")
    print(f"    вход:  {usage['input_tokens']} токенов, из них из кеша {cached}")
    print(f"    выход: {usage['output_tokens']} токенов")
    print(f"    всего: {usage['total_tokens']} токенов")
    mark = "  ставок для этой модели нет, считано запасными" if guessed else ""
    print(f"    деньги: ${money:.6f}{mark}")

print(f"  итого по всем моделям: ${total_money:.6f}")
print()

print("ТО ЖЕ САМОЕ МЕНЕДЖЕРОМ КОНТЕКСТА")

with get_usage_metadata_callback() as usage_callback:
    fast.invoke("Ответьте одним словом: да или нет?")
    strong.invoke("Ответьте одним словом: да или нет?")

    for model_name, usage in usage_callback.usage_metadata.items():
        print(
            f"  {model_name}: вход {usage['input_tokens']}, "
            f"выход {usage['output_tokens']}"
        )

names = len(callback.usage_metadata)

print()
print("СКОЛЬКО ИМЁН В СЧЁТЕ:", names)

if names == 1:
    print("Одно имя означает, что обе роли ушли на одну модель.")
else:
    print("Имён столько, сколько разных моделей вы позвали, счёт разложен по ним.")
