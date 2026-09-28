"""Пример 5 урока 5: дописать к системному промпту, а не заменить его.

Два middleware добавляют по блоку к системному сообщению, третий печатает то,
что получилось. Работа идёт через content_blocks, поэтому блоки предыдущих
middleware остаются на месте.
"""

from collections.abc import Callable
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage

from course_model import build_model


@wrap_model_call
def add_project_facts(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Добавляет блок с фактами о проекте."""
    new_content = list(request.system_message.content_blocks) + [
        {
            "type": "text",
            "text": (
                "Проект: внутренний портал заявок. "
                "Стек: Python, FastAPI, PostgreSQL, очередь на Redis."
            ),
        }
    ]
    return handler(request.override(system_message=SystemMessage(content=new_content)))


@wrap_model_call
def add_today(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Добавляет блок с сегодняшней датой: сама модель её не знает."""
    new_content = list(request.system_message.content_blocks) + [
        {"type": "text", "text": f"Сегодня {date.today().isoformat()}."}
    ]
    return handler(request.override(system_message=SystemMessage(content=new_content)))


@wrap_model_call
def show_blocks(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает блоки системного сообщения в том виде, в каком они уйдут."""
    for number, block in enumerate(request.system_message.content_blocks, start=1):
        print(f"  блок {number}: {block}")
    return handler(request)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=300),
    tools=[],
    system_prompt="Вы помощник команды разработки. Отвечайте по-русски и коротко.",
    middleware=[add_project_facts, add_today, show_blocks],
)

print("СИСТЕМНОЕ СООБЩЕНИЕ, СОБРАННОЕ ТРЕМЯ MIDDLEWARE")
try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Какая сегодня дата и на чём написан наш проект?",
                }
            ]
        }
    )
    print()
    print("ОТВЕТ:", result["messages"][-1].text)
except Exception as error:
    # Системное сообщение из нескольких блоков провайдер может не принять.
    # Это нормальный исход для шлюза, а не ошибка примера.
    print()
    print(f"ПРОВАЙДЕР ОТКАЗАЛ: {type(error).__name__}: {error}")
