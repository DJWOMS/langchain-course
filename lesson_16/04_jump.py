"""Пример 4 урока 16: прыжок к концу запуска и прыжок без объявления.

Middleware считает вызовы модели и после первого вызова возвращает
jump_to="end". Один и тот же код запускается дважды: в первом middleware хук
помечен hook_config(can_jump_to=["end"]), во втором пометки нет.

Пометка задаёт условное ребро в графе. Без неё узел соединён обычным ребром,
и jump_to из before_model никто не читает.
"""

from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

from course_model import build_model

OWNERS = {
    "/src/reports": "нет",
    "/src/billing": "Ирина",
    "/src/auth": "Павел",
    "/src/gateway": "Сергей",
}


def module_owner(path: str) -> str:
    """Возвращает владельца модуля репозитория по пути."""
    return f"{path}: владелец {OWNERS.get(path, 'неизвестен')}"


class TurnState(AgentState):
    """Состояние агента плюс счётчик вызовов модели."""

    turns: NotRequired[int]


class TurnLimitMiddleware(AgentMiddleware[TurnState]):
    """Завершает запуск после заданного числа вызовов модели."""

    state_schema = TurnState

    def __init__(self, max_turns: int) -> None:
        super().__init__()
        self.max_turns = max_turns

    def _verdict(self, state: TurnState) -> dict | None:
        """Общее тело хука: решение об остановке по числу сделанных вызовов модели."""
        if state.get("turns", 0) >= self.max_turns:
            print(f"  [{self.name}] запрошен прыжок к концу запуска")
            return {
                "messages": [
                    AIMessage("Лимит вызовов модели исчерпан, отвечаю тем, что успел собрать.")
                ],
                "jump_to": "end",
            }
        return None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: TurnState, runtime: Runtime) -> dict | None:
        return self._verdict(state)

    def after_model(self, state: TurnState, runtime: Runtime) -> dict | None:
        return {"turns": state.get("turns", 0) + 1}


class SilentTurnLimitMiddleware(TurnLimitMiddleware):
    """Тот же middleware, у которого хук не объявил, что умеет прыгать."""

    def before_model(self, state: TurnState, runtime: Runtime) -> dict | None:
        return self._verdict(state)


QUESTION = (
    "Назовите владельцев модулей /src/reports, /src/billing, /src/auth и "
    "/src/gateway."
)

model = build_model(temperature=0, max_tokens=256)


def run(label: str, layer: AgentMiddleware) -> None:
    """Запускает агента с одним middleware и печатает, чем кончился запуск."""
    agent = create_agent(
        model=model,
        tools=[module_owner],
        system_prompt=(
            "Вы помощник по разбору легаси. Владельцев узнавайте инструментом. "
            "Отвечайте по-русски одной фразой."
        ),
        middleware=[layer],
    )
    print(label)
    for edge in agent.get_graph().edges:
        if edge.source.endswith(".before_model"):
            kind = "условное" if edge.conditional else "обычное"
            print(f"  ребро {kind}: {edge.source} -> {edge.target}")

    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}], "turns": 0})
    ai = sum(1 for message in result["messages"] if message.type == "ai")
    tools = sum(1 for message in result["messages"] if message.type == "tool")
    print(f"  записано вызовов модели: {result['turns']}")
    print(f"  сообщений AI: {ai}, ответов инструмента: {tools}")
    print(f"  последнее сообщение: {result['messages'][-1].text}")
    print()


run("MIDDLEWARE С hook_config(can_jump_to=['end'])", TurnLimitMiddleware(max_turns=1))
run("ТОТ ЖЕ MIDDLEWARE БЕЗ ПОМЕТКИ", SilentTurnLimitMiddleware(max_turns=1))
