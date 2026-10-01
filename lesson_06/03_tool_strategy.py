"""Пример 3 урока 6: ToolStrategy, названная явно.

Схема превращается в инструмент, модель "вызывает" его аргументами по схеме, а
агент разбирает аргументы и кладёт объект в structured_response. Параметр
tool_message_content меняет текст сообщения инструмента, который остаётся в
переписке.
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from course_model import build_model

TEXT = "Хочу вернуть куртку из заказа 7781, размер не подошёл."


class Ticket(BaseModel):
    """Разобранное обращение в службу поддержки."""

    category: Literal["оплата", "доставка", "возврат", "прочее"] = Field(
        description="Тема обращения"
    )
    urgency: Literal["низкая", "средняя", "высокая"] = Field(
        description="Срочность обращения"
    )
    summary: str = Field(description="Суть обращения одним предложением, по-русски")


def show(message) -> str:
    """Одна строка на сообщение: тип, вызовы инструментов, начало текста."""
    text = message.text.replace("\n", " ")
    calls = [call["name"] for call in getattr(message, "tool_calls", [])]
    return f"{type(message).__name__:<14} {calls} {text[:70]!r}"


model = build_model(temperature=0, max_tokens=1024)

default_agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(Ticket),
)

custom_agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=Ticket,
        tool_message_content="Обращение разобрано и передано в очередь возвратов.",
    ),
)

print("ТЕКСТ СООБЩЕНИЯ ИНСТРУМЕНТА ПО УМОЛЧАНИЮ")
result = default_agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
for number, message in enumerate(result["messages"], start=1):
    print(f"  {number}. {show(message)}")
print()

print("АРГУМЕНТЫ ВЫЗОВА, ИЗ КОТОРЫХ СОБРАН ОБЪЕКТ")
for message in result["messages"]:
    for call in getattr(message, "tool_calls", []):
        print(f"  {call['name']}: {call['args']}")
print()

print("СВОЙ ТЕКСТ СООБЩЕНИЯ ИНСТРУМЕНТА")
custom = custom_agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
for number, message in enumerate(custom["messages"], start=1):
    print(f"  {number}. {show(message)}")
print()

print("РАЗОБРАННЫЙ ОТВЕТ:", custom["structured_response"])
