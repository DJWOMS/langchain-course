"""Пример 3 урока 8: сырые события протокола и счётчик по каналам.

Объект прогона можно обходить сам по себе, и тогда приходят события
протокола. У короткого запуска агента их десятки, поэтому скрипт печатает
сводку: первые восемь событий и счёт по каналам.
"""

from collections import Counter

from langchain.agents import create_agent

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски и коротко.",
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]},
    version="v3",
)

counts = Counter()
first_rows = []
for event in stream:
    counts[event["method"]] += 1
    if len(first_rows) < 8:
        params = event["params"]
        first_rows.append((event.get("seq"), event["method"], params["namespace"]))

print("ПЕРВЫЕ ВОСЕМЬ СОБЫТИЙ")
for seq, method, namespace in first_rows:
    print(f"  seq={seq} method={method!r} namespace={namespace}")

print()
print("ВСЕГО СОБЫТИЙ:", sum(counts.values()))
print("ПО КАНАЛАМ:")
for method, number in counts.most_common():
    print(f"  {method:<12} {number}")

final = stream.output
print()
print("КЛЮЧИ ИТОГОВОГО СОСТОЯНИЯ:", list(final))
print("СООБЩЕНИЙ В СОСТОЯНИИ:", len(final["messages"]))
