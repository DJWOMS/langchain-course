"""Пример 5 урока 6: схема сломалась, агент повторил запрос.

В схеме два вида ограничений. Первые выражаются в JSON Schema, и модель видит
их вместе с описанием полей. Вторые заданы в валидаторе Pydantic, и модель о них
не знает вовсе: узнать она может только из текста ошибки. Здесь стоит второе,
поэтому первый вызов почти наверняка не пройдёт проверку.
"""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field, field_validator

from course_model import build_model

TEXT = "Оплатил заказ 4412 на 5300 рублей, товар не пришёл, верните деньги."


class Refund(BaseModel):
    """Заявка на возврат денег."""

    order_id: str = Field(description="Номер заказа из обращения")
    amount: float = Field(description="Сумма к возврату в рублях")

    @field_validator("order_id")
    @classmethod
    def check_order_id(cls, value: str) -> str:
        """Внутренний формат номера, которого нет в JSON Schema."""
        if not value.startswith("ORD-"):
            message = "Номер заказа записывается с префиксом ORD-, например ORD-1234"
            raise ValueError(message)
        return value


def show(message) -> str:
    """Одна строка на сообщение: тип, вызовы инструментов, начало текста."""
    text = message.text.replace("\n", " ")
    calls = [call["name"] for call in getattr(message, "tool_calls", [])]
    return f"{type(message).__name__:<14} {calls} {text[:110]!r}"


print("ЧТО ИЗ СХЕМЫ ВИДИТ МОДЕЛЬ")
print(" ", Refund.model_json_schema()["properties"])
print()

model = build_model(temperature=0, max_tokens=1024)
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(Refund),  # handle_errors=True по умолчанию
)

result = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})

print("ПЕРЕПИСКА")
for number, message in enumerate(result["messages"], start=1):
    print(f"  {number}. {show(message)}")
print()

print("ВЫЗОВОВ МОДЕЛИ:", sum(1 for m in result["messages"] if type(m).__name__ == "AIMessage"))
print("РАЗОБРАННЫЙ ОТВЕТ:", result["structured_response"])
