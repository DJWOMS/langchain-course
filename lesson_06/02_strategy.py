"""Пример 2 урока 6: какую стратегию агент выбрал сам.

Голая схема в response_format означает "выбери стратегию сам". Выбор виден по
двум вещам: по профилю модели, откуда фреймворк читает поддержку нативного
вывода, и по следам в переписке. Пара "AIMessage с вызовом инструмента плюс
ToolMessage" означает ToolStrategy.
"""

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from course_model import build_model

TEXT = "Когда приедет заказ 4412? Обещали вчера."


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

print("ИМЯ МОДЕЛИ:", getattr(model, "model_name", None))

profile = model.profile
if profile is None:
    print("ПРОФИЛЬ МОДЕЛИ: данных нет, значение None")
else:
    print("ПОЛЕЙ В ПРОФИЛЕ:", len(profile))
    print("ПОДДЕРЖКА structured_output:", profile.get("structured_output"))
print()

agent = create_agent(model=model, tools=[], response_format=Ticket)
result = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})

print("ПЕРЕПИСКА")
for number, message in enumerate(result["messages"], start=1):
    print(f"  {number}. {show(message)}")
print()

print("РАЗОБРАННЫЙ ОТВЕТ:", result["structured_response"])
