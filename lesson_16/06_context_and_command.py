"""Пример 6 урока 16: свой контекст и правка состояния из обёртки.

Middleware читает конфигурацию запуска из request.runtime.context и дописывает
к системному сообщению правило тарифа. После ответа модели он считает
потраченные токены и записывает их в состояние, возвращая ExtendedModelResponse
с Command: у хука-обёртки других способов записать в состояние нет.

Тип контекста объявлен параметром класса: AgentMiddleware[UsageState,
SupportContext]. Сам класс контекста уходит в create_agent аргументом
context_schema, а значение приходит при вызове аргументом context.
"""

from collections.abc import Callable
from dataclasses import dataclass

from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import SystemMessage
from langgraph.types import Command

from course_model import build_model

OWNERS = {"/src/reports": "нет", "/src/billing": "Ирина", "/src/auth": "Павел"}

PLAN_RULES = {
    "free": "Тариф free: отвечайте одним предложением, без пояснений.",
    "pro": "Тариф pro: после ответа добавьте абзац с рекомендацией, что делать дальше.",
}


def module_owner(path: str) -> str:
    """Возвращает владельца модуля репозитория по пути."""
    return f"{path}: владелец {OWNERS.get(path, 'неизвестен')}"


@dataclass
class SupportContext:
    """Конфигурация запуска: кто спрашивает и по какому тарифу."""

    user_id: str
    plan: str


class UsageState(AgentState):
    """Состояние агента плюс расход токенов, накопленный middleware."""

    spent_tokens: NotRequired[int]


class PlanMiddleware(AgentMiddleware[UsageState, SupportContext]):
    """Правило тарифа в системном сообщении и счёт токенов в состоянии."""

    state_schema = UsageState

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        plan = request.runtime.context.plan
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({"type": "text", "text": PLAN_RULES[plan]})

        response = handler(request.override(system_message=SystemMessage(content=blocks)))

        usage = getattr(response.result[-1], "usage_metadata", None) or {}
        spent = request.state.get("spent_tokens", 0) + usage.get("total_tokens", 0)

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"spent_tokens": spent}),
        )


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[module_owner],
    system_prompt=(
        "Вы помощник по разбору легаси. Владельцев узнавайте инструментом. "
        "Отвечайте по-русски."
    ),
    middleware=[PlanMiddleware()],
    context_schema=SupportContext,
)

QUESTION = "Кто владелец модуля /src/reports?"

for plan in ("free", "pro"):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        context=SupportContext(user_id="u-101", plan=plan),
    )
    print(f"ТАРИФ {plan.upper()}")
    print(f"  токенов за запуск: {result['spent_tokens']}")
    print(f"  ответ: {result['messages'][-1].text}")
    print()
