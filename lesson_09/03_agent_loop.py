"""Пример 3. Тот же цикл, его выполняет агент.

Инструмент и вопрос те же, что в примере 2. Разница в том, что этапы 1-3
здесь не написаны: их делает create_agent.
"""

from course_model import build_model
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def get_order_status(order_id: str) -> str:
    """Узнать статус заказа в магазине по его номеру."""
    catalog = {
        "A-1001": "собран, ждёт курьера",
        "A-1002": "в пути, доставка завтра",
        "A-1003": "отменён покупателем",
    }
    return catalog.get(order_id, "заказ с таким номером не найден")


agent = create_agent(
    model=build_model(temperature=0),
    tools=[get_order_status],
    system_prompt="Вы отвечаете на вопросы о заказах. Номер заказа берите из вопроса.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Что с заказом A-1002?"}]})

print("СООБЩЕНИЙ В СОСТОЯНИИ:", len(result["messages"]))
for number, message in enumerate(result["messages"], start=1):
    kind = type(message).__name__
    if kind == "AIMessage" and message.tool_calls:
        detail = "просит вызвать: " + ", ".join(
            f"{call['name']}({call['args']})" for call in message.tool_calls
        )
    elif kind == "ToolMessage":
        detail = f"{message.name} -> {message.content}"
    else:
        detail = message.text
    print(f"{number}. {kind}: {detail}")

print()
print("ОТВЕТ ПОЛЬЗОВАТЕЛЮ:", result["messages"][-1].text)
