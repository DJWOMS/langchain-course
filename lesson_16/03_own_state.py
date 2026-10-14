"""Пример 3 урока 16: своё поле состояния у middleware.

Middleware объявляет подкласс AgentState с двумя своими полями и ставит его
атрибутом state_schema. Поля хранятся в состоянии агента наравне с messages:
их можно передать при вызове, прочитать в хуке и увидеть в результате.

Печать показывает, какие поля попали в результат: поля jump_to среди них нет.
"""

from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from course_model import build_model

TESTS = {"/src/reports": 0, "/src/billing": 41, "/src/auth": 12}


def module_tests(path: str) -> str:
    """Возвращает число тестов модуля репозитория по пути."""
    count = TESTS.get(path)
    if count is None:
        return f"модуля {path} в репозитории нет"
    return f"{path}: тестов {count}"


class BudgetState(AgentState):
    """Состояние агента плюс два счётчика middleware."""

    model_calls: NotRequired[int]
    tool_calls: NotRequired[int]


class BudgetMiddleware(AgentMiddleware[BudgetState]):
    """Считает вызовы модели и запрошенные ею вызовы инструментов."""

    state_schema = BudgetState

    def before_model(self, state: BudgetState, runtime: Runtime) -> dict | None:
        print(
            f"[before_model] сделано вызовов модели: {state.get('model_calls', 0)}, "
            f"заказано вызовов инструмента: {state.get('tool_calls', 0)}"
        )
        return None

    def after_model(self, state: BudgetState, runtime: Runtime) -> dict | None:
        last = state["messages"][-1]
        return {
            "model_calls": state.get("model_calls", 0) + 1,
            "tool_calls": state.get("tool_calls", 0)
            + len(getattr(last, "tool_calls", []) or []),
        }


agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[module_tests],
    system_prompt=(
        "Вы помощник по разбору легаси. Число тестов узнавайте инструментом, "
        "по одному модулю за вызов. Отвечайте по-русски одной фразой."
    ),
    middleware=[BudgetMiddleware()],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Сколько тестов в /src/reports и сколько в /src/auth?",
            }
        ],
        "model_calls": 0,
        "tool_calls": 0,
    }
)

print()
print("поля результата:", ", ".join(sorted(result)))
print("вызовов модели:", result["model_calls"])
print("вызовов инструмента:", result["tool_calls"])
print("ответ:", result["messages"][-1].text)
