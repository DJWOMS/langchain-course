"""Пример 2 урока 8: проекция stream.messages у агента и имя узла.

У агента вызовов модели за один запуск несколько, и stream.messages отдаёт по
одному объекту на каждый вызов. У каждого объекта есть .node, имя узла графа,
из которого пришло сообщение. В текущей версии узел модели называется "model",
а не "agent", и строки "ОТБОР ПО ИМЕНИ" в выводе показывают разницу числом.
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

rows = []
for message in stream.messages:
    # Дельты считаются обходом .text, полное сообщение берётся из .output.
    deltas = sum(1 for _ in message.text)
    output = message.output
    calls = [call["name"] for call in output.tool_calls]
    rows.append((message.node, deltas, len(output.text), calls))

for number, (node, deltas, length, calls) in enumerate(rows, start=1):
    print(f"сообщение {number}: узел={node!r} дельт={deltas} знаков={length} вызовы={calls}")

print()
print("СООБЩЕНИЙ ПО УЗЛАМ:", dict(Counter(node for node, *_ in rows)))
print("ОТБОР ПО ИМЕНИ 'agent':", sum(1 for node, *_ in rows if node == "agent"))
print("ОТБОР ПО ИМЕНИ 'model':", sum(1 for node, *_ in rows if node == "model"))

final = stream.output
print("СООБЩЕНИЙ В ИТОГОВОМ СОСТОЯНИИ:", len(final["messages"]))
print("ОТВЕТ:", final["messages"][-1].text)
