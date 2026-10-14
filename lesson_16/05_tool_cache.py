"""Пример 5 урока 16: обёртка вокруг инструмента, отвечающая без запуска.

Хук wrap_tool_call решает, вызывать ли handler. Если такой вызов уже был,
middleware возвращает готовый ToolMessage и тело инструмента не выполняется.

Агент вызывается дважды с одним и тем же вопросом. Счётчик внутри инструмента
показывает, сколько раз тело инструмента выполнилось на самом деле.
"""

import json
from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.types import Command

from course_model import build_model

OWNERS = {"/src/reports": "нет", "/src/billing": "Ирина", "/src/auth": "Павел"}

runs = {"count": 0}
CACHE: dict[tuple[str, str], str] = {}


def module_owner(path: str) -> str:
    """Возвращает владельца модуля репозитория по пути."""
    runs["count"] += 1
    return f"{path}: владелец {OWNERS.get(path, 'неизвестен')}"


@wrap_tool_call
def cache_tool_calls(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Отдаёт прежний ответ инструмента, если вызов повторяется дословно."""
    key = (
        request.tool_call["name"],
        json.dumps(request.tool_call["args"], sort_keys=True, ensure_ascii=False),
    )

    if key in CACHE:
        print(f"[кеш] {key[0]}{key[1]}: ответ из кеша, инструмент не вызывается")
        return ToolMessage(
            content=CACHE[key],
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
        )

    result = handler(request)

    if isinstance(result, ToolMessage) and result.status == "success":
        CACHE[key] = result.text

    return result


agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[module_owner],
    system_prompt=(
        "Вы помощник по разбору легаси. Владельцев узнавайте инструментом. "
        "Отвечайте по-русски одной фразой."
    ),
    middleware=[cache_tool_calls],
)

QUESTION = "Кто владелец модуля /src/billing?"

for number in (1, 2):
    print(f"ВЫЗОВ АГЕНТА {number}")
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print(f"  запусков тела инструмента всего: {runs['count']}")
    print(f"  записей в кеше: {len(CACHE)}")
    print(f"  ответ: {result['messages'][-1].text}")
    print()
