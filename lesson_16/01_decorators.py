"""Пример 1 урока 16: свой middleware двумя декораторами.

Хук-узел before_model считает вызовы модели и печатает, сколько сообщений
истории она получает. Хук-обёртка wrap_tool_call охватывает вызов инструмента:
замеряет время и печатает имя вместе с аргументами.

Обе функции становятся middleware и уходят в create_agent обычным
списком middleware.
"""

import time
from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ToolCallRequest,
    before_model,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from course_model import build_model

MODULES = {
    "/src/reports": 7305,
    "/src/billing": 2140,
    "/src/auth": 860,
}

turns = {"count": 0}


def module_size(path: str) -> str:
    """Возвращает число строк модуля репозитория по пути."""
    lines = MODULES.get(path)
    if lines is None:
        return f"модуля {path} в репозитории нет"
    return f"{path}: {lines} строк"


@before_model
def count_turns(state: AgentState, runtime: Runtime) -> dict | None:
    """Перед вызовом модели печатает его номер и длину истории."""
    turns["count"] += 1
    print(
        f"[before_model] вызов модели {turns['count']}, "
        f"сообщений в истории: {len(state['messages'])}"
    )
    return None


@wrap_tool_call
def time_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Замеряет время вызова инструмента и печатает имя с аргументами."""
    started = time.perf_counter()
    result = handler(request)
    spent_ms = (time.perf_counter() - started) * 1000
    print(
        f"[wrap_tool_call] {request.tool_call['name']}"
        f"({request.tool_call['args']}) за {spent_ms:.1f} мс"
    )
    return result


agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[module_size],
    system_prompt=(
        "Вы помощник по разбору легаси. Размеры модулей узнавайте инструментом. "
        "Отвечайте по-русски одной фразой."
    ),
    middleware=[count_turns, time_tool],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Что больше по числу строк, /src/reports или /src/auth?",
            }
        ]
    }
)

print()
print("имя middleware у декоратора:", count_turns.name, "и", time_tool.name)
print("ответ:", result["messages"][-1].text)
