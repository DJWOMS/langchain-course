"""Пример 3 урока 10: инструмент читает состояние разговора.

Инструмент подсчитывает сообщения разговора к моменту вызова и возвращает
сводку. Это короткая память: она существует один запуск агента и обнуляется
вместе с ним.
"""

from collections import Counter

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from course_model import build_model


@tool
def conversation_summary(runtime: ToolRuntime) -> str:
    """Вернуть сводку по текущему разговору: сколько сообщений и каких."""
    messages = runtime.state["messages"]
    kinds = Counter(type(message).__name__ for message in messages)

    first = messages[0].text.replace("\n", " ") if messages else ""
    if len(first) > 40:
        first = first[:37] + "..."

    parts = ", ".join(f"{name} {count}" for name, count in sorted(kinds.items()))
    return f"Сообщений {len(messages)} ({parts}). Первое: {first!r}"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[conversation_summary],
    system_prompt=(
        "Вы помощник по разговору. Про состояние разговора отвечайте только по "
        "данным инструмента, ничего не придумывайте. Отвечайте по-русски и коротко."
    ),
)

DIALOG = [
    {"role": "user", "content": "Здравствуйте, у меня задвоился платёж по заказу 4412."},
    {"role": "assistant", "content": "Здравствуйте. Проверяю списания по заказу 4412."},
    {"role": "user", "content": "Сколько сообщений уже в нашем разговоре?"},
]

result = agent.invoke({"messages": DIALOG})

print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
print()
print("СОСТОЯНИЕ ПОСЛЕ ПРОГОНА")
for number, message in enumerate(result["messages"], start=1):
    text = message.text.replace("\n", " ")
    if len(text) > 50:
        text = text[:47] + "..."
    print(f"  {number}. {type(message).__name__:<14} {text!r}")
