"""Пример 4 урока 11: своя схема состояния и редьюсеры полей.

Схема наследуется от AgentState и добавляет три поля: warehouse приходит со
входа и читается инструментом, checked накапливает артикулы редьюсером
operator.add, last_sku хранит последний артикул своим редьюсером.

Запрос сделан так, чтобы модель вызвала инструмент дважды в одном ответе. Правки
от обоих вызовов приходят в одном шаге графа, и видно, как каждый редьюсер их сводит.
"""

import operator
from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from course_model import build_model

STOCK = {
    "A-100": {"Казань": 7, "Омск": 0},
    "B-200": {"Казань": 0, "Омск": 12},
}


def last_wins(_current: str, update: str) -> str:
    """Редьюсер: в состоянии остаётся та правка, что пришла последней."""
    return update


class StockState(AgentState):
    """Состояние агента склада: к messages добавлены три своих поля."""

    warehouse: str
    checked: Annotated[list[str], operator.add]
    last_sku: Annotated[str, last_wins]


@tool
def check_stock(sku: str, runtime: ToolRuntime[None, StockState]) -> Command:
    """Проверяет остаток товара по артикулу. Артикул вида A-100."""
    warehouse = runtime.state.get("warehouse", "неизвестен")
    left = STOCK.get(sku, {}).get(warehouse)

    if left is None:
        answer = f"Артикул {sku} на складе {warehouse} не заведён."
    else:
        answer = f"Артикул {sku}, склад {warehouse}: остаток {left} шт."

    return Command(
        update={
            "checked": [sku],
            "last_sku": sku,
            "messages": [
                ToolMessage(content=answer, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[check_stock],
    system_prompt=(
        "Вы кладовщик. Остатки берите только из инструмента, по одному вызову на "
        "артикул. Отвечайте по-русски одной строкой."
    ),
    state_schema=StockState,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Проверьте остатки по артикулам A-100 и B-200."}
        ],
        "warehouse": "Казань",
    }
)

first_call = next(m for m in result["messages"] if getattr(m, "tool_calls", None))

print("КЛЮЧИ СОСТОЯНИЯ:", sorted(result))
print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
print("ВЫЗОВОВ ИНСТРУМЕНТА В ПЕРВОМ ОТВЕТЕ МОДЕЛИ:", len(first_call.tool_calls))
print()
print("warehouse (поле входа):", result.get("warehouse"))
print("checked (operator.add):", result.get("checked"))
print("last_sku (last_wins):  ", result.get("last_sku"))
print()

print("ВЫЗОВЫ ИНСТРУМЕНТА В ИСТОРИИ")
for message in result["messages"]:
    if type(message).__name__ == "ToolMessage":
        print("  ", message.text.replace("\n", " "))
