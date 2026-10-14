"""Пример 6 урока 14: голый агент против собранного на одной и той же задаче.

Три вопроса подряд, у каждого агента свой thread, один набор инструментов, один
провайдер. Разница только в харнессе: у второго агента промпт собирается под роль,
в запрос уходят только разрешённые роли инструменты, а старые результаты
инструментов заменяются заглушкой по порогу.

Печатается число вызовов модели и сумма входных токенов за весь разговор.
"""

from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

REPORTS = {
    "Казань": "бой 9, подмочка 5, пересорт 3",
    "Омск": "бой 2, подмочка 1, пересорт 14",
    "Пермь": "бой 4, подмочка 12, пересорт 2",
}

FILLER = (
    "Приёмка идёт по электронной очереди, талон берут на въезде. Паллеты "
    "принимают по счёту мест и по весу брутто. Упаковку вскрывают при видимом "
    "повреждении обрешётки и снимают вскрытие на камеру. Температурный груз "
    "замеряют дважды. Документы возвращают водителю с отметкой о приёмке, копию "
    "подшивают в журнал склада и дублируют в учётную систему до 20:00 того же "
    "дня. Расхождение больше пяти мест останавливает приёмку до прихода "
    "старшего смены."
)

LONG_PROMPT = (
    "Вы помощник логистического отдела. Отвечайте по-русски, одной фразой. "
    "Данные берите инструментами, по памяти не отвечайте. "
    "Инструмент read_report отдаёт недельный отчёт по складу города вместе с "
    "регламентом приёмки. Инструмент read_stock отдаёт остаток по артикулу. "
    "Инструмент read_routes отдаёт маршруты через город. Инструмент "
    "write_order меняет статус заказа, его разрешено звать только оператору "
    "склада, и только после того, как оператор назвал номер заказа целиком. "
    "Инструмент cancel_order отменяет заказ, его разрешено звать только "
    "оператору склада и только после подтверждения суммы возврата. "
    "Наблюдателю запрещено менять данные: если наблюдатель просит отменить "
    "заказ или сменить статус, откажите и предложите обратиться к оператору."
)


@dataclass
class Access:
    """Конфигурация запуска: под какой ролью работает собеседник."""

    role: str


def read_report(city: str) -> str:
    """Возвращает полный недельный отчёт по складу города."""
    defects = REPORTS.get(city)

    if defects is None:
        return f"Отчёта по складу {city} нет."

    return f"Отчёт по складу {city}. Регламент: {FILLER} Брак за неделю: {defects}."


def read_stock(sku: str) -> str:
    """Возвращает остаток товара по артикулу на всех складах."""
    return f"Артикул {sku}: Казань 7 шт., Омск 0 шт."


def read_routes(city: str) -> str:
    """Возвращает список маршрутов, которые проходят через город."""
    return f"Через город {city} идут маршруты M-1 и M-4."


def write_order(order_id: str, status: str) -> str:
    """Меняет статус заказа на указанный."""
    return f"Заказ {order_id} переведён в статус {status}."


def cancel_order(order_id: str) -> str:
    """Отменяет заказ по номеру."""
    return f"Заказ {order_id} отменён."


ALL_TOOLS = [read_report, read_stock, read_routes, write_order, cancel_order]


@dynamic_prompt
def prompt_for_role(request: ModelRequest) -> str:
    """Собирает короткую инструкцию под роль собеседника."""
    base = (
        "Вы помощник логистического отдела. Отвечайте по-русски, одной фразой. "
        "Данные берите инструментами, по памяти не отвечайте."
    )

    if request.runtime.context.role == "оператор":
        return base + " Менять статус и отменять заказ разрешено."

    return base + " Менять данные вам нельзя, доступно только чтение."


@wrap_model_call
def tools_by_role(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Оставляет в запросе инструменты, разрешённые роли собеседника."""
    if request.runtime.context.role != "оператор":
        request = request.override(
            tools=[t for t in request.tools if t.name.startswith("read_")]
        )

    return handler(request)


def spent(messages) -> int:
    """Складывает входные токены по всем ответам модели в истории thread."""
    total = 0

    for message in messages:
        # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
        usage = getattr(message, "usage_metadata", None) or {}
        total += usage.get("input_tokens", 0)

    return total


bare = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=ALL_TOOLS,
    system_prompt=LONG_PROMPT,
    context_schema=Access,
    checkpointer=InMemorySaver(),
)

tuned = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=ALL_TOOLS,
    context_schema=Access,
    middleware=[
        prompt_for_role,
        tools_by_role,
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=300, keep=1)],
        ),
    ],
    checkpointer=InMemorySaver(),
)

QUESTIONS = [
    "Сколько брака за неделю на складе Казань?",
    "А на складе Омск?",
    "А на складе Пермь?",
]

print(f"{'агент':<10}  {'вызовов':>7}  {'вход всего':>10}  последний ответ")

for label, agent in (("голый", bare), ("собранный", tuned)):
    config = {"configurable": {"thread_id": f"together-{label}"}}

    for question in QUESTIONS:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config,
            context=Access(role="наблюдатель"),
        )

    messages = result["messages"]
    calls = sum(1 for message in messages if message.type == "ai")
    answer = messages[-1].text.replace("\n", " ")
    print(f"{label:<10}  {calls:>7}  {spent(messages):>10}  {answer}")
