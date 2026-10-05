"""Пример 5. Схема аргументов на Pydantic вместо аннотаций.

Схема добавляет к подписи функции описание каждого поля,
закрытый список допустимых значений и границы чисел.
"""

import json
from typing import Literal

from course_model import build_model
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

ORDERS = [
    {"id": "A-1001", "customer": "Иванов", "status": "shipped", "sum": 4300},
    {"id": "A-1002", "customer": "Иванов", "status": "new", "sum": 1200},
    {"id": "A-1003", "customer": "Петрова", "status": "cancelled", "sum": 890},
    {"id": "A-1004", "customer": "Иванов", "status": "shipped", "sum": 15600},
]


class OrderSearch(BaseModel):
    """Параметры поиска по журналу заказов."""

    customer: str = Field(description="Фамилия покупателя ровно так, как в вопросе")
    status: Literal["new", "shipped", "cancelled"] = Field(
        default="new",
        description="Статус заказа: new это новый, shipped это отправлен, "
        "cancelled это отменён",
    )
    limit: int = Field(default=5, ge=1, le=50, description="Сколько заказов вернуть")


@tool(args_schema=OrderSearch)
def search_orders(customer: str, status: str = "new", limit: int = 5) -> str:
    """Найти заказы покупателя с указанным статусом."""
    found = [o for o in ORDERS if o["customer"] == customer and o["status"] == status]
    if not found:
        return "ничего не найдено"
    return "; ".join(f"{o['id']} на {o['sum']} рублей" for o in found[:limit])


print("СХЕМА, КОТОРУЮ УВИДИТ МОДЕЛЬ:")
print(json.dumps(search_orders.tool_call_schema.model_json_schema(), ensure_ascii=False, indent=2))

agent = create_agent(
    model=build_model(temperature=0),
    tools=[search_orders],
    system_prompt="Вы помощник менеджера магазина. Отвечайте одним предложением.",
)

question = "Какие отправленные заказы есть у Иванова?"
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

print()
print("ВОПРОС:", question)
for message in result["messages"]:
    if type(message).__name__ == "AIMessage" and message.tool_calls:
        for call in message.tool_calls:
            print("МОДЕЛЬ ПРИСЛАЛА АРГУМЕНТЫ:", call["args"])
    if type(message).__name__ == "ToolMessage":
        print("ИНСТРУМЕНТ ВЕРНУЛ:", message.content)

print("ОТВЕТ:", result["messages"][-1].text)
