"""Пример 5 урока 8: две проекции подряд против interleave.

Проекция накапливает значения только с того момента, как у неё появился
подписчик. Поэтому в синхронном коде два обхода подряд и один обход через
interleave дают разные числа. Скрипт прогоняет одного и того же агента дважды
и печатает обе сводки.
"""

from langchain.agents import create_agent

from course_model import build_model

QUESTION = {
    "messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]
}


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски и коротко.",
)

print("ДВА ОБХОДА ПОДРЯД")
stream = agent.stream_events(QUESTION, version="v3")
messages = [message.node for message in stream.messages]
calls = [call.tool_name for call in stream.tool_calls]
print("  сообщений:", len(messages), messages)
print("  вызовов инструментов:", len(calls), calls)

print()
print("ОДИН ОБХОД ЧЕРЕЗ INTERLEAVE")
stream = agent.stream_events(QUESTION, version="v3")
order = []
for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        order.append(f"messages(узел {item.node})")
    elif name == "tool_calls":
        order.append(f"tool_calls({item.tool_name})")
    else:
        order.append(f"values(сообщений {len(item['messages'])})")

for step, row in enumerate(order, start=1):
    print(f"  {step}. {row}")

print("  всего записей:", len(order))
