"""Пример 1 урока 5: системный промпт как параметр агента.

Один и тот же вопрос уходит двум агентам: у первого системного промпта нет,
у второго он есть. Смотрим на разницу в ответе и на то, что осталось в состоянии.
"""

from langchain.agents import create_agent

from course_model import build_model

QUESTION = "Что такое очередь задач?"

model = build_model(temperature=0, max_tokens=1024)

plain = create_agent(model=model, tools=[])

strict = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "Вы отвечаете строго одним предложением, без вступлений, списков и примеров."
    ),
)

print("БЕЗ СИСТЕМНОГО ПРОМПТА")
result_plain = plain.invoke({"messages": [{"role": "user", "content": QUESTION}]})
print(result_plain["messages"][-1].text)
print()

print("С СИСТЕМНЫМ ПРОМПТОМ")
result_strict = strict.invoke({"messages": [{"role": "user", "content": QUESTION}]})
print(result_strict["messages"][-1].text)
print()

print("КЛЮЧИ СОСТОЯНИЯ ВТОРОГО АГЕНТА:", sorted(result_strict))
print(
    "ТИПЫ СООБЩЕНИЙ В СОСТОЯНИИ:",
    [type(message).__name__ for message in result_strict["messages"]],
)
