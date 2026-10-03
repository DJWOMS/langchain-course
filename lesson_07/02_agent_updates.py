"""Пример 2 урока 7: прогресс агента, режим потока updates.

Режим updates отдаёт по событию после каждого шага агента: что вернул узел model
и что вернул узел tools. Токенов в этом режиме нет, видно только, из каких
шагов состоит прогон.
"""

from langchain.agents import create_agent

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в городе."""
    return f"В городе {city} всегда солнечно!"


def describe(message) -> str:
    """Одна строка на сообщение: тип, вызовы инструментов, начало текста."""
    calls = getattr(message, "tool_calls", None)
    if calls:
        names = [call["name"] for call in calls]
        return f"{type(message).__name__} вызывает {names}"

    text = message.text.replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{type(message).__name__} {text!r}"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски, одним предложением.",
)

step = 0

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] != "updates":
        continue

    for node, update in chunk["data"].items():
        step += 1
        # У служебных ключей вроде __interrupt__ правка приходит не словарём.
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if not messages:
            print(f"шаг {step}: узел {node}, правка без сообщений: {update!r}")
            continue
        for message in messages:
            print(f"шаг {step}: узел {node}, {describe(message)}")

print(f"всего шагов: {step}")
