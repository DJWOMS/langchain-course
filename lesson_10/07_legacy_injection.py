"""Пример 7 урока 10: старые механизмы внедрения рядом с нынешним.

Два инструмента делают одно и то же. Первый собран по-старому, четырьмя разными
механизмами: InjectedState, InjectedStore, InjectedToolCallId и вызов
get_runtime внутри тела. Второй берёт то же самое из одного параметра
ToolRuntime. Схема для модели у обоих должна быть одинаковой.

Каждый агент получает свой единственный инструмент, иначе выбор между ними
остаётся за моделью и прогон перестаёт быть сравнением.
"""

from dataclasses import dataclass
from typing import Annotated

from langchain.agents import create_agent
from langchain.tools import (
    InjectedState,
    InjectedStore,
    InjectedToolCallId,
    ToolRuntime,
    tool,
)
from langgraph.runtime import get_runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from course_model import build_model

NAMESPACE = ("plans",)


@dataclass
class SessionContext:
    """Конфигурация запуска."""

    user_id: str


@tool
def old_style_report(
    topic: str,
    state: Annotated[dict, InjectedState],
    store: Annotated[BaseStore, InjectedStore],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Собрать служебную справку по теме обращения."""
    try:
        user_id = get_runtime(SessionContext).context.user_id
    except Exception as error:  # noqa: BLE001
        # Защитная ветка: get_runtime работает только внутри прогона графа.
        user_id = f"<{type(error).__name__}>"

    item = store.get(NAMESPACE, user_id)
    plan = item.value["plan"] if item else "неизвестен"

    return (
        f"тема={topic} пользователь={user_id} тариф={plan} "
        f"сообщений={len(state['messages'])} вызов={tool_call_id[:8]}"
    )


@tool
def new_style_report(topic: str, runtime: ToolRuntime[SessionContext]) -> str:
    """Собрать служебную справку по теме обращения."""
    user_id = runtime.context.user_id

    item = runtime.store.get(NAMESPACE, user_id) if runtime.store else None
    plan = item.value["plan"] if item else "неизвестен"

    return (
        f"тема={topic} пользователь={user_id} тариф={plan} "
        f"сообщений={len(runtime.state['messages'])} "
        f"вызов={runtime.tool_call_id[:8]}"
    )


store = InMemoryStore()
store.put(NAMESPACE, "u-101", {"plan": "premium"})

QUESTION = "Соберите справку по теме billing."

for report_tool in (old_style_report, new_style_report):
    agent = create_agent(
        model=build_model(temperature=0, max_tokens=512),
        tools=[report_tool],
        system_prompt=(
            "Вы помощник поддержки. Справку собирайте инструментом и возвращайте "
            "пользователю ровно ту строку, которую вернул инструмент, без правок."
        ),
        context_schema=SessionContext,
        store=store,
    )

    print(f"ИНСТРУМЕНТ {report_tool.name}")
    print("  видит модель:", sorted(report_tool.args))

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": QUESTION}]},
            context=SessionContext(user_id="u-101"),
        )
    except Exception as error:  # noqa: BLE001
        # Защитная ветка: прежняя схема могла перестать работать.
        print(f"  ОТКАЗ: {type(error).__name__}: {error}")
        continue

    tool_results = [
        message.text.replace("\n", " ")
        for message in result["messages"]
        if type(message).__name__ == "ToolMessage"
    ]
    print("  вернул инструмент:", tool_results)
    print("  ответ агента:", result["messages"][-1].text.replace("\n", " "))
