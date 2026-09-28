"""Пример 2 урока 5: что физически уходит в модель на каждом шаге.

Middleware с декоратором @wrap_model_call стоит вокруг вызова модели и
печатает то, что в этот вызов уходит: системное сообщение плюс список сообщений
разговора. Инструмент здесь нужен затем, чтобы шагов было больше одного.
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from course_model import build_model

CALLS = {"n": 0}


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


def show(message) -> str:
    """Одна строка на сообщение: тип, начало текста, вызовы инструментов."""
    text = message.text.replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."

    # Поле tool_calls есть только у AIMessage.
    calls = getattr(message, "tool_calls", None)
    suffix = f"  tool_calls={[call['name'] for call in calls]}" if calls else ""

    return f"{type(message).__name__:<14} {text!r}{suffix}"


@wrap_model_call
def window(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает то, что уйдёт в модель, и пропускает вызов дальше."""
    CALLS["n"] += 1
    print(f"--- шаг {CALLS['n']}, в модель уходит ---")

    outgoing = list(request.messages)
    if request.system_message is not None:
        outgoing = [request.system_message, *outgoing]

    for number, message in enumerate(outgoing, start=1):
        print(f"  {number}. {show(message)}")

    print(f"  сообщений в запросе: {len(outgoing)}")
    print(f"  сообщений в состоянии: {len(request.state['messages'])}")
    # Инструмент провайдера описывается словарём, и поля name у него нет.
    names = [getattr(tool, "name", tool) for tool in request.tools]
    print(f"  инструменты в запросе: {names}")

    return handler(request)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[get_weather],
    system_prompt="Вы помощник. Отвечайте кратко.",
    middleware=[window],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]}
)

print()
print(
    "СОСТОЯНИЕ ПОСЛЕ ПРОГОНА:",
    [type(message).__name__ for message in result["messages"]],
)
print("ОТВЕТ:", result["messages"][-1].text)
