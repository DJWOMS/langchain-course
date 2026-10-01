"""Пример 7 урока 6: две схемы на выбор.

ToolStrategy принимает объединение типов, и каждая схема превращается в свой
инструмент. Модель выбирает одну из них по содержанию обращения. Третье
обращение написано так, что подходят обе схемы сразу: это тот случай, ради
которого во фреймворке есть отдельная ошибка про несколько ответов.
"""

from typing import Literal, Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from course_model import build_model

MESSAGES = [
    "Деньги списали дважды за заказ 4412, верните лишнее.",
    "Позвоните мне на +7 999 123-45-67 после шести вечера, вопрос по доставке.",
    "Деньги списали дважды за заказ 4412, и перезвоните на +7 999 123-45-67 вечером.",
]


class Ticket(BaseModel):
    """Обращение, на которое поддержка отвечает письмом."""

    category: Literal["оплата", "доставка", "возврат", "прочее"] = Field(
        description="Тема обращения"
    )
    urgency: Literal["низкая", "средняя", "высокая"] = Field(
        description="Срочность обращения"
    )
    summary: str = Field(description="Суть обращения одним предложением, по-русски")


class CallbackRequest(BaseModel):
    """Просьба перезвонить: клиент хочет разговор с человеком."""

    phone: str = Field(description="Номер телефона из обращения")
    reason: str = Field(description="Причина звонка одним предложением, по-русски")
    time_hint: str = Field(description="Когда клиенту удобно, словами из обращения")


def show(message) -> str:
    """Одна строка на сообщение: тип, вызовы инструментов, начало текста."""
    text = message.text.replace("\n", " ")
    calls = [call["name"] for call in getattr(message, "tool_calls", [])]
    return f"{type(message).__name__:<14} {calls} {text[:90]!r}"


model = build_model(temperature=0, max_tokens=1024)
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(Union[Ticket, CallbackRequest]),
)

for number, text in enumerate(MESSAGES, start=1):
    print(f"ОБРАЩЕНИЕ {number}: {text}")
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})

    for position, message in enumerate(result["messages"], start=1):
        print(f"  {position}. {show(message)}")

    answer = result.get("structured_response")
    print("  ВЫБРАННАЯ СХЕМА:", type(answer).__name__)
    print("  РАЗОБРАННЫЙ ОТВЕТ:", answer)
    print()
