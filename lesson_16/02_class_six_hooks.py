"""Пример 2 урока 16: один класс со всеми шестью хуками.

Middleware ничего не меняет, он только записывает в журнал каждое своё срабатывание.
По журналу видно, какие хуки срабатывают один раз за запуск, какие при каждом
вызове модели и как обёртки охватывают вызовы изнутри.

Внизу печатаются узлы собранного графа: хуки-узлы там есть, хуки-обёртки нет.
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from course_model import build_model

OWNERS = {"/src/reports": "нет", "/src/billing": "Ирина", "/src/auth": "Павел"}


def module_owner(path: str) -> str:
    """Возвращает владельца модуля репозитория по пути."""
    return f"{path}: владелец {OWNERS.get(path, 'неизвестен')}"


class AuditMiddleware(AgentMiddleware):
    """Пишет строку журнала на каждое срабатывание своих хуков."""

    def __init__(self) -> None:
        super().__init__()
        self.journal: list[str] = []

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self.journal.append("before_agent")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        self.journal.append(f"before_model, сообщений {len(state['messages'])}")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self.journal.append(f"wrap_model_call: вход, инструментов {len(request.tools)}")
        response = handler(request)
        self.journal.append(f"wrap_model_call: выход, сообщений {len(response.result)}")
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        self.journal.append(f"wrap_tool_call: вход, {name}")
        result = handler(request)
        self.journal.append(f"wrap_tool_call: выход, {name}")
        return result

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        last = state["messages"][-1]
        calls = len(getattr(last, "tool_calls", []) or [])
        self.journal.append(f"after_model, запрошено вызовов инструмента {calls}")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        self.journal.append("after_agent")
        return None


audit = AuditMiddleware()

agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[module_owner],
    system_prompt=(
        "Вы помощник по разбору легаси. Владельцев узнавайте инструментом. "
        "Отвечайте по-русски одной фразой."
    ),
    middleware=[audit],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Кто владелец модуля /src/billing?"}]}
)

print("ЖУРНАЛ MIDDLEWARE")
for number, line in enumerate(audit.journal, start=1):
    print(f"  {number:>2}. {line}")

print()
print("узлы собранного графа:")
for node in agent.get_graph().nodes:
    print(f"  {node}")

print()
print("ответ:", result["messages"][-1].text)
