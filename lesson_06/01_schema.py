"""Пример 1 урока 6: схема вместо разбора текста вручную.

Агенту задан параметр response_format со схемой на Pydantic. Разобранный ответ
приходит отдельным ключом состояния, а не текстом последнего сообщения.
"""

from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from course_model import build_model

TEXT = "Деньги списали дважды за один заказ 4412, верните лишнее сегодня же."


class Ticket(BaseModel):
    """Разобранное обращение в службу поддержки."""

    category: Literal["оплата", "доставка", "возврат", "прочее"] = Field(
        description="Тема обращения"
    )
    urgency: Literal["низкая", "средняя", "высокая"] = Field(
        description="Срочность обращения"
    )
    summary: str = Field(description="Суть обращения одним предложением, по-русски")


model = build_model(temperature=0, max_tokens=1024)
agent = create_agent(model=model, tools=[], response_format=Ticket)

result = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
answer = result["structured_response"]

print("ТИП ОТВЕТА:", type(answer).__name__)
print("КАТЕГОРИЯ:", answer.category)
print("СРОЧНОСТЬ:", answer.urgency)
print("СУТЬ:", answer.summary)
print()

print("КЛЮЧИ СОСТОЯНИЯ:", sorted(result))
print("ТИПЫ СООБЩЕНИЙ:", [type(message).__name__ for message in result["messages"]])
print("ТЕКСТ ПОСЛЕДНЕГО СООБЩЕНИЯ:", repr(result["messages"][-1].text))
