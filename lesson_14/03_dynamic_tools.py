"""Пример 3 урока 14: набор инструментов под права собеседника и цена этого набора.

Агент один, инструментов восемь, задача одна и та же. Middleware оставляет в
запросе только те инструменты, которые разрешены роли из конфигурации запуска.
Наблюдатель печатает, сколько инструментов ушло в модель и во что провайдер
оценил вход.
"""

from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from course_model import build_model

ORDERS = [
    {"id": "Z-771", "city": "Казань", "status": "укомплектован"},
    {"id": "Z-773", "city": "Казань", "status": "застрял на складе"},
    {"id": "Z-774", "city": "Казань", "status": "застрял на складе"},
    {"id": "Z-775", "city": "Омск", "status": "в пути"},
]


@dataclass
class Access:
    """Конфигурация запуска: под какой ролью работает собеседник."""

    role: str


def read_orders(city: str) -> str:
    """Возвращает список заказов по городу вместе со статусом каждого."""
    rows = [o for o in ORDERS if o["city"] == city]

    if not rows:
        return f"Заказов по городу {city} нет."

    return "; ".join(f"{o['id']} - {o['status']}" for o in rows)


def read_stock(sku: str) -> str:
    """Возвращает остаток товара по артикулу на всех складах."""
    return f"Артикул {sku}: Казань 7 шт., Омск 0 шт."


def read_routes(city: str) -> str:
    """Возвращает список маршрутов, которые проходят через город."""
    return f"Через город {city} идут маршруты M-1 и M-4."


def read_notes(order_id: str) -> str:
    """Возвращает комментарии сотрудников к заказу."""
    return f"Комментариев к заказу {order_id} нет."


def write_order(order_id: str, status: str) -> str:
    """Меняет статус заказа на указанный."""
    return f"Заказ {order_id} переведён в статус {status}."


def cancel_order(order_id: str) -> str:
    """Отменяет заказ по номеру."""
    return f"Заказ {order_id} отменён."


def refund_payment(order_id: str) -> str:
    """Возвращает деньги за заказ на карту плательщика."""
    return f"По заказу {order_id} оформлен возврат."


def delete_order(order_id: str) -> str:
    """Удаляет заказ из системы без возможности восстановления."""
    return f"Заказ {order_id} удалён."


ALL_TOOLS = [
    read_orders,
    read_stock,
    read_routes,
    read_notes,
    write_order,
    cancel_order,
    refund_payment,
    delete_order,
]


@wrap_model_call
def tools_by_role(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Оставляет в запросе инструменты, разрешённые роли, и печатает цену набора."""
    role = request.runtime.context.role

    if role != "оператор":
        request = request.override(
            tools=[t for t in request.tools if t.name.startswith("read_")]
        )

    response = handler(request)
    # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
    usage = response.result[0].usage_metadata or {}

    print(
        f"{role:<10}  {len(request.tools):>6}  {usage.get('input_tokens', 0):>5}  "
        f"{', '.join(t.name for t in request.tools)}"
    )

    return response


agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=ALL_TOOLS,
    system_prompt=(
        "Вы помощник логистического отдела. Отвечайте по-русски, одной фразой. "
        "Данные берите инструментами, по памяти не отвечайте."
    ),
    middleware=[tools_by_role],
    context_schema=Access,
)

QUESTION = "Сколько заказов застряло на складе в Казани?"

print(f"{'роль':<10}  {'инстр':>6}  {'вход':>5}  инструменты в запросе")

for role in ("оператор", "наблюдатель"):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        context=Access(role=role),
    )
    answer = result["messages"][-1].text.replace("\n", " ")
    print(f"{role:<10}  ОТВЕТ: {answer}")
