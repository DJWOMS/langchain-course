"""Пример 6 урока 5: промпт, который смотрит на состояние разговора.

Тот же вопрос задаётся дважды: в пустом разговоре и в разговоре, где уже
накопилась история. Промпт собирается по числу сообщений в состоянии.
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.messages import AIMessage, HumanMessage

from course_model import build_model

BASE = "Вы помощник службы поддержки. Отвечайте по-русски."


@dynamic_prompt
def length_aware_prompt(request: ModelRequest) -> str:
    """При длинном разговоре добавляет требование отвечать короче."""
    message_count = len(request.messages)

    if message_count > 6:
        return f"{BASE} Разговор затянулся, отвечайте одним предложением."
    return f"{BASE} Отвечайте подробно, с примером."


@wrap_model_call
def show_system(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает число сообщений и выбранный промпт."""
    print(f"  сообщений в состоянии: {len(request.state['messages'])}")
    print(f"  промпт: {request.system_message.text!r}")
    return handler(request)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[],
    middleware=[length_aware_prompt, show_system],
)

QUESTION = "Как отменить заказ?"

HISTORY = [
    HumanMessage("Здравствуйте"),
    AIMessage("Здравствуйте, чем помочь?"),
    HumanMessage("Заказ 4412 идёт долго"),
    AIMessage("Заказ в пути, срок доставки уточняется"),
    HumanMessage("А если я передумаю?"),
    AIMessage("Отмена возможна до передачи в доставку"),
]

print("КОРОТКИЙ РАЗГОВОР")
short = agent.invoke({"messages": [HumanMessage(QUESTION)]})
print(f"  ответ: {short['messages'][-1].text}")
print()

print("ДЛИННЫЙ РАЗГОВОР")
long_run = agent.invoke({"messages": [*HISTORY, HumanMessage(QUESTION)]})
print(f"  ответ: {long_run['messages'][-1].text}")
