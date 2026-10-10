"""Пример 2 урока 13: знания о человеке дописываются в системный промпт.

Агент собран с двумя видами памяти: чекпойнтер сохраняет переписку одного
thread, в хранилище лежит то, что известно о человеке. Middleware перед каждым
вызовом модели достаёт записи из хранилища и дописывает их в системный промпт,
поэтому отдельного вызова модели для чтения памяти не нужно.

Два вызова агента идут в разных threads. История первого thread во второй не
попадает, а записи о человеке доступны обоим.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from course_model import build_model


@dataclass
class Session:
    """Конфигурация запуска: чьи записи доставать из хранилища."""

    user_id: str


store = InMemoryStore()

# Записи уже лежат в хранилище: их положил прошлый диалог, которого в этом
# запуске нет. Как они туда попадают, разбирается в примере 5.
store.put(("u-101", "facts"), "city", {"text": "Работает в Омске"})
store.put(("u-101", "facts"), "role", {"text": "Ведёт склад запчастей"})
store.put(("u-101", "preferences"), "answer_style", {"text": "Отвечать без списков"})


@wrap_model_call
def recall_memory(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Дописывает в системный промпт записи о пользователе из хранилища."""
    store = request.runtime.store
    if store is None:
        # Агент собран без хранилища: запрос уходит к модели без изменений.
        return handler(request)

    user_id = request.runtime.context.user_id
    items = store.search((user_id,), limit=10)
    if not items:
        return handler(request)

    known = "\n".join(f"- {item.value['text']}" for item in items)
    blocks = list(request.system_message.content_blocks) + [
        {"type": "text", "text": f"Что известно о пользователе:\n{known}"}
    ]
    return handler(request.override(system_message=SystemMessage(content=blocks)))


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[],
    system_prompt="Вы рабочий помощник. Отвечайте по-русски и коротко.",
    middleware=[recall_memory],
    context_schema=Session,
    checkpointer=InMemorySaver(),
    store=store,
)

session = Session(user_id="u-101")

first = agent.invoke(
    {"messages": [{"role": "user", "content": "Посчитайте, сколько будет 17 умножить на 3."}]},
    config={"configurable": {"thread_id": "thread-1"}},
    context=session,
)
print("THREAD 1:", first["messages"][-1].text.replace("\n", " "))
print("  сообщений в thread:", len(first["messages"]))
print()

second = agent.invoke(
    {"messages": [{"role": "user", "content": "Что вы считали минуту назад и в каком городе я работаю?"}]},
    config={"configurable": {"thread_id": "thread-2"}},
    context=session,
)
print("THREAD 2:", second["messages"][-1].text.replace("\n", " "))
print("  сообщений в thread:", len(second["messages"]))
