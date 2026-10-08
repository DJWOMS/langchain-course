"""Пример 5 урока 10: инструмент читает и пишет долгую память.

Два запуска агента подряд. Сообщения первого запуска во второй не попадают,
а хранилище у них общее, поэтому записанное в первом запуске достаётся во
втором.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from course_model import build_model

NAMESPACE = ("preferences",)


@dataclass
class SessionContext:
    """Конфигурация запуска: кто спрашивает."""

    user_id: str


@tool
def save_preference(preference: str, runtime: ToolRuntime[SessionContext]) -> str:
    """Запомнить пожелание пользователя о том, как ему отвечать."""
    if runtime.store is None:
        return "Хранилище не подключено, запомнить нечем."

    runtime.store.put(NAMESPACE, runtime.context.user_id, {"preference": preference})
    return f"Запомнил: {preference}"


@tool
def get_preference(runtime: ToolRuntime[SessionContext]) -> str:
    """Вернуть ранее сохранённое пожелание пользователя."""
    if runtime.store is None:
        return "Хранилище не подключено, вспомнить нечем."

    item = runtime.store.get(NAMESPACE, runtime.context.user_id)
    if item is None:
        return "Пожеланий не сохранено."

    return item.value["preference"]


store = InMemoryStore()

agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[save_preference, get_preference],
    system_prompt=(
        "Вы помощник. Пожелания пользователя о форме ответа сохраняйте "
        "инструментом и доставайте инструментом, по памяти не отвечайте. "
        "Отвечайте по-русски и коротко."
    ),
    context_schema=SessionContext,
    store=store,
)

context = SessionContext(user_id="u-101")

first = agent.invoke(
    {"messages": [{"role": "user", "content": "Запомните: отвечайте мне без списков."}]},
    context=context,
)
print("ЗАПУСК 1:", first["messages"][-1].text.replace("\n", " "))
print("  сообщений в состоянии:", len(first["messages"]))

second = agent.invoke(
    {"messages": [{"role": "user", "content": "Как я просил вам отвечать?"}]},
    context=context,
)
print("ЗАПУСК 2:", second["messages"][-1].text.replace("\n", " "))
print("  сообщений в состоянии:", len(second["messages"]))

print()
item = store.get(NAMESPACE, "u-101")
print("В ХРАНИЛИЩЕ ПОСЛЕ ДВУХ ЗАПУСКОВ:", item.value if item else "<пусто>")
