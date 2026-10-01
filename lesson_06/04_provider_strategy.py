"""Пример 4 урока 6: ProviderStrategy, названная явно.

Здесь схему держит не фреймворк, а сам провайдер: ему уходит параметр
response_format с JSON Schema. Поддержка есть не у каждого адреса, поэтому
вызов обёрнут в защитную ветку: отказ печатается строкой, а не роняет скрипт.
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
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


print("ЧТО УХОДИТ ПРОВАЙДЕРУ")
strategy = ProviderStrategy(Ticket)
sent = strategy.to_model_kwargs()["response_format"]
print("  тип формата:", sent["type"])
print("  имя схемы:", sent["json_schema"]["name"])
print("  поля схемы:", sorted(sent["json_schema"]["schema"]["properties"]))
print()

model = build_model(temperature=0, max_tokens=1024)
agent = create_agent(model=model, tools=[], response_format=ProviderStrategy(Ticket))

print("ПРОГОН С НАТИВНЫМ ВЫВОДОМ")
try:
    result = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
except Exception as error:  # noqa: BLE001
    # Адрес провайдера может не принимать response_format с JSON Schema.
    print("  отказ:", type(error).__name__)
    print("  текст:", str(error)[:300])
else:
    for number, message in enumerate(result["messages"], start=1):
        print(f"  {number}. {show(message)}")
    print()
    print("  РАЗОБРАННЫЙ ОТВЕТ:", result["structured_response"])
