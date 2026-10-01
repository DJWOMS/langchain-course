"""Пример 8 урока 6: схема плюс инструмент, близко к рабочей задаче.

Агент разбирает обращение, при необходимости ходит за статусом заказа и
возвращает готовую запись со вложенным объектом. Поле order необязательное: в
обращении номера заказа может не быть вовсе, и схема это допускает.

ВАЖНАЯ ДЕТАЛЬ ПРО ToolStrategy ВМЕСТЕ С ИНСТРУМЕНТАМИ. С ToolStrategy агент
выставляет модели tool_choice="any", то есть на каждом
шаге модель обязана вызвать какой-нибудь инструмент. Свободным текстом она
ответить не может: либо идёт в инструмент, либо отдаёт ответ по схеме. Это
видно в переписке первого обращения.
"""

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from course_model import build_model

ORDERS = {
    "4412": "в доставке, ожидается 18.09.2026, сумма 5300 рублей",
    "7781": "доставлен 12.09.2026, сумма 990 рублей",
}

MESSAGES = [
    "Где мой заказ 4412? Обещали вчера.",
    "Верните деньги за заказ 7781, куртка не подошла.",
    "Подскажите, вы доставляете в выходные?",
]


def find_order(order_id: str) -> str:
    """Возвращает статус и сумму заказа по его номеру."""
    return ORDERS.get(order_id, f"заказ {order_id} не найден")


class OrderFact(BaseModel):
    """Данные заказа, полученные инструментом."""

    order_id: str = Field(description="Номер заказа")
    status: str = Field(description="Статус заказа так, как его вернул инструмент")


class DeskAnswer(BaseModel):
    """Готовая запись службы поддержки по одному обращению."""

    category: Literal["оплата", "доставка", "возврат", "прочее"] = Field(
        description="Тема обращения"
    )
    urgency: Literal["низкая", "средняя", "высокая"] = Field(
        description="Срочность обращения"
    )
    order: OrderFact | None = Field(
        default=None,
        description="Данные заказа, если в обращении назван его номер",
    )
    reply: str = Field(description="Ответ клиенту, одно-два предложения, по-русски")


def show(message) -> str:
    """Одна строка на сообщение: тип, вызовы инструментов, начало текста."""
    text = message.text.replace("\n", " ")
    calls = [call["name"] for call in getattr(message, "tool_calls", [])]
    return f"{type(message).__name__:<14} {calls} {text[:70]!r}"


model = build_model(temperature=0, max_tokens=1024)
agent = create_agent(
    model=model,
    tools=[find_order],
    system_prompt=(
        "Вы оператор службы поддержки интернет-магазина. Если в обращении назван "
        "номер заказа, сначала посмотрите его статус инструментом, и только потом "
        "составляйте ответ. Отвечайте по-русски."
    ),
    response_format=DeskAnswer,
)

for number, text in enumerate(MESSAGES, start=1):
    print(f"ОБРАЩЕНИЕ {number}: {text}")
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})
    answer = result.get("structured_response")

    if answer is None:
        # Схема не собралась: печатаем переписку и идём дальше.
        print("  структурированного ответа нет")
        for position, message in enumerate(result["messages"], start=1):
            print(f"  {position}. {show(message)}")
        print()
        continue

    if number == 1:
        for position, message in enumerate(result["messages"], start=1):
            print(f"  {position}. {show(message)}")

    print("  тема:", answer.category, " срочность:", answer.urgency)
    print("  заказ:", answer.order.order_id if answer.order else "не назван")
    print("  статус:", answer.order.status if answer.order else "-")
    print("  ответ клиенту:", answer.reply)
    print(
        "  вызовов модели:",
        sum(1 for m in result["messages"] if type(m).__name__ == "AIMessage"),
    )
    print()
