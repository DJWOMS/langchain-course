"""Пример 6 урока 11: из чего собирается харнесс.

Два агента на одних и тех же инструментах. Первый голый: модель, инструменты,
системный промпт. Второму добавлены middleware из пяти категорий настройки,
всех, кроме среды исполнения.

Модель здесь не вызывается ни разу, печатается только схема графа: видно, во что
превращается каждый middleware и в каком порядке узлы стоят вокруг цикла.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model


def find_order(number: str) -> str:
    """Находит заказ по номеру и возвращает товар и город доставки."""
    return f"Заказ {number}: наушники, город доставки Казань."


def refund(number: str, reason: str) -> str:
    """Оформляет возврат денег по заказу. Действие необратимое."""
    return f"Возврат по заказу {number} оформлен, причина: {reason}."


SYSTEM_PROMPT = "Вы оператор службы доставки. Отвечайте по-русски и коротко."

model = build_model(temperature=0, max_tokens=512)
tools = [find_order, refund]

plain = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

configured = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        # Управление контекстом: сжать историю, пока она не переполнила окно.
        SummarizationMiddleware(
            model=model, trigger=("tokens", 4000), keep=("messages", 20)
        ),
        # Планирование: список задач, который модель обновляет инструментом write_todos.
        TodoListMiddleware(),
        # Отказоустойчивость: бюджет вызовов модели и повтор упавшего инструмента.
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolRetryMiddleware(max_retries=2),
        # Ограждения: почтовый адрес не уходит в модель как есть.
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # Вмешательство человека: возврат денег без человека не делается.
        HumanInTheLoopMiddleware(interrupt_on={"refund": True}),
    ],
    checkpointer=InMemorySaver(),
)

print("ГОЛЫЙ АГЕНТ")
print(plain.get_graph().draw_mermaid())
print()
print("АГЕНТ С MIDDLEWARE")
print(configured.get_graph().draw_mermaid())
