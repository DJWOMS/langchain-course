"""Пример 5 урока 2: готовый агент одним вызовом. Разбор идёт в уроке 11.
"""

from langchain.agents import create_agent

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0),
    tools=[get_weather],
    system_prompt="Вы помощник. Отвечайте коротко.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Какая погода в Краснодаре?"}]}
)

print("ЧТО ЛЕЖИТ В СОСТОЯНИИ:", sorted(result))
print()

for message in result["messages"]:
    print(f"{type(message).__name__}: {message.content!r}")

print()
print("ОТВЕТ:", result["messages"][-1].text)
