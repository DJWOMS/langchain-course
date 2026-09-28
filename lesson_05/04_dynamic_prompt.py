"""Пример 4 урока 5: динамический промпт по контексту запуска.

Роль пользователя приходит в запуск параметром context, а декоратор
@dynamic_prompt собирает по ней системный промпт. Второй middleware стоит следом
и печатает то системное сообщение, которое в итоге уходит в модель.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)

from course_model import build_model

BASE = "Вы отвечаете на вопросы по разработке. Отвечайте по-русски, до трёх предложений."


@dataclass
class Context:
    """Форма данных, которые передаются в запуск агента."""

    user_role: str = "user"


@dynamic_prompt
def role_prompt(request: ModelRequest) -> str:
    """Собирает системный промпт по роли пользователя."""
    role = request.runtime.context.user_role

    if role == "expert":
        return f"{BASE} Отвечайте технически, термины не расшифровывайте."
    if role == "beginner":
        return f"{BASE} Объясняйте на бытовом примере, без терминов."
    return BASE


@wrap_model_call
def show_system(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает системное сообщение, которое уходит в модель."""
    print(f"  системное сообщение: {request.system_message.text!r}")
    return handler(request)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=300),
    tools=[],
    system_prompt="СТАТИЧЕСКИЙ ПРОМПТ, ЗАДАННЫЙ ПРИ СОЗДАНИИ АГЕНТА",
    middleware=[role_prompt, show_system],
    context_schema=Context,
)

QUESTION = "Зачем нужна очередь задач?"

for role in ("expert", "beginner"):
    print(f"РОЛЬ: {role}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        context=Context(user_role=role),
    )
    print(f"  ответ: {result['messages'][-1].text}")
    print()
