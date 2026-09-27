"""Пример 5 урока 4: выбор модели на лету внутри агента.

Агент здесь взят как готовая деталь, разбор в уроке 11, middleware в уроках
15 и 16. Разбирается одно место: как подменить модель на конкретном вызове.
"""

import os
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from course_model import build_model

fast = build_model(os.environ["MODEL_NAME"], temperature=0, max_tokens=256)
strong = build_model(
    os.getenv("MODEL_NAME_STRONG") or os.environ["MODEL_NAME"],
    temperature=0,
    max_tokens=512,
)


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


@wrap_model_call
def pick_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Короткий разговор ведёт дешёвая модель, разросшийся, дорогая."""
    message_count = len(request.state["messages"])
    chosen = strong if message_count > 2 else fast
    print(f"  [middleware] сообщений в состоянии: {message_count}, модель: {chosen.model_name}")
    return handler(request.override(model=chosen))


agent = create_agent(
    model=fast,
    tools=[get_weather],
    system_prompt="Вы помощник. Отвечайте по-русски.",
    middleware=[pick_model],
)

print("ПРОГОН АГЕНТА")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]}
)
print()

for message in result["messages"]:
    print(f"  {type(message).__name__}: {message.text[:70]!r}")

print()
print("ОТВЕТ:", result["messages"][-1].text[:120])
