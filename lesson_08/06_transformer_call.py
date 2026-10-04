"""Пример 6 урока 8: свой преобразователь, зарегистрированный на вызове.

Класс преобразователя передаётся в stream_events аргументом transformers, и его
проекции появляются в stream.extensions под теми же ключами, которые вернул
init(). Подписка на tool_totals оформляется до прогона: значение туда кладётся в
finalize(), и без подписчика оно потерялось бы.
"""

from collections import Counter

from langchain.agents import create_agent

from course_model import build_model
from tool_activity import ToolActivityTransformer


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


def get_population(city: str) -> str:
    """Возвращает число жителей указанного города."""
    numbers = {"Сан-Франциско": "808 437", "Бостон": "653 833"}
    return numbers.get(city, "нет данных")


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather, get_population],
    system_prompt="Отвечайте по-русски и коротко.",
)

stream = agent.stream_events(
    {
        "messages": [
            {
                "role": "user",
                "content": "Какая погода и сколько жителей в Сан-Франциско?",
            }
        ]
    },
    version="v3",
    transformers=[ToolActivityTransformer],
)

print("КЛЮЧИ EXTENSIONS:", sorted(stream.extensions))

# Подписка без обхода: значение из finalize() иначе некуда будет положить.
totals = iter(stream.extensions["tool_totals"])

print("ЗАПИСИ ПРОЕКЦИИ tool_activity")
seen = Counter()
for record in stream.extensions["tool_activity"]:
    seen[record["event"]] += 1
    print(f"  {record['event']:<18} {record['tool']}")

print()
print("СЧЁТ НА СТОРОНЕ ЧИТАТЕЛЯ:", dict(seen))
print("СЧЁТ ИЗ finalize():", next(totals, None))
