"""Пример 4 урока 8: жизненный цикл вызова инструмента через stream.tool_calls.

Проекция stream.tool_calls отдаёт по объекту на каждый запущенный инструмент.
В объекте лежат имя, аргументы, поток частичного вывода, итоговый результат и
строка ошибки. Подписка здесь одна, на эту проекцию, и прогон вперёд двигает
её обход.
"""

from langchain.agents import create_agent

from course_model import build_model


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
)

number = 0
for call in stream.tool_calls:
    number += 1
    deltas = list(call.output_deltas)
    print(f"вызов {number}: {call.tool_name}")
    print(f"  вход: {call.input}")
    print(f"  дельт вывода: {len(deltas)}")
    print(f"  результат: {call.output!r}")
    print(f"  ошибка: {call.error!r}")
    print(f"  завершён: {call.completed}")

print()
print("ВЫЗОВОВ ИНСТРУМЕНТОВ ВСЕГО:", number)
print("ОТВЕТ:", stream.output["messages"][-1].text)
