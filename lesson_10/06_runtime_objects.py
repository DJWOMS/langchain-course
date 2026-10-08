"""Пример 6 урока 10: два объекта, Runtime и ToolRuntime, в одном прогоне.

Middleware получает Runtime, инструмент получает ToolRuntime. Конфигурация
запуска, хранилище и запись в поток у них общие, а состояние, объект config,
идентификатор вызова и список инструментов есть только у второго.

Чекпойнтер подключён, как в приложении с памятью сессии, а thread_id передан
в config: из него заполняется поле execution_info.thread_id. Память сессии
разбирается в уроке 12.
"""

from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from course_model import build_model


@dataclass
class SessionContext:
    """Конфигурация запуска."""

    user_id: str


@before_model
def show_runtime(state: AgentState, runtime: Runtime[SessionContext]) -> None:
    """Печатает поля Runtime перед вызовом модели."""
    info = runtime.execution_info
    print("MIDDLEWARE, объект Runtime")
    print("  context.user_id:", runtime.context.user_id)
    print("  store:", runtime.store)
    print("  server_info:", runtime.server_info)
    print("  есть ли поле config:", hasattr(runtime, "config"))
    print("  execution_info.thread_id:", info.thread_id if info else None)
    print("  сообщений в состоянии:", len(state["messages"]))
    return None


@tool
def whoami(runtime: ToolRuntime[SessionContext]) -> str:
    """Вернуть идентификатор текущего пользователя."""
    info = runtime.execution_info
    configurable = runtime.config.get("configurable", {})

    print("ИНСТРУМЕНТ, объект ToolRuntime")
    print("  context.user_id:", runtime.context.user_id)
    print("  tool_call_id:", runtime.tool_call_id)
    print("  сообщений в состоянии:", len(runtime.state["messages"]))
    print("  config configurable thread_id:", configurable.get("thread_id"))
    print("  execution_info.thread_id:", info.thread_id if info else None)
    print("  execution_info.node_attempt:", info.node_attempt if info else None)
    print("  инструментов в runtime.tools:", len(runtime.tools))

    return runtime.context.user_id


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[whoami],
    system_prompt="Вы помощник. Отвечайте по-русски одним предложением.",
    context_schema=SessionContext,
    middleware=[show_runtime],
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Под каким идентификатором я вошёл?"}]},
    config={"configurable": {"thread_id": "session-1"}},
    context=SessionContext(user_id="u-101"),
)

print()
print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
