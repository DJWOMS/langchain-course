"""Пример 5 урока 14: встроенное редактирование контекста и его временность.

Три вопроса в одном thread, каждый с тяжёлым результатом инструмента. Middleware
ContextEditingMiddleware заменяет старые результаты заглушкой, когда история
превышает порог.

Наблюдатель стоит после редактора в списке middleware, поэтому получает уже
правленый запрос: то, что ушло в модель. В конце печатается состояние thread, где лежит
неправленая история.
"""

from itertools import count
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

REPORTS = {
    "Казань": "бой 9, подмочка 5, пересорт 3",
    "Омск": "бой 2, подмочка 1, пересорт 14",
    "Пермь": "бой 4, подмочка 12, пересорт 2",
}

FILLER = (
    "Приёмка идёт по электронной очереди, талон берут на въезде. Паллеты "
    "принимают по счёту мест и по весу брутто. Упаковку вскрывают при видимом "
    "повреждении обрешётки и снимают вскрытие на камеру. Температурный груз "
    "замеряют дважды. Документы возвращают водителю с отметкой о приёмке, копию "
    "подшивают в журнал склада и дублируют в учётную систему до 20:00 того же "
    "дня. Расхождение больше пяти мест останавливает приёмку до прихода "
    "старшего смены."
)


def warehouse_report(city: str) -> str:
    """Возвращает полный недельный отчёт по складу города."""
    defects = REPORTS.get(city)

    if defects is None:
        return f"Отчёта по складу {city} нет."

    return f"Отчёт по складу {city}. Регламент: {FILLER} Брак за неделю: {defects}."


step = count(1)


@wrap_model_call
def observer(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Печатает то, что дошло до модели после работы редактора контекста."""
    tool_messages = [m for m in request.messages if m.type == "tool"]
    cleared = sum(
        1
        for m in tool_messages
        if m.response_metadata.get("context_editing", {}).get("cleared")
    )
    shapes = ", ".join(f"{len(m.text):>4}" for m in tool_messages)

    print(
        f"{next(step):>5}  {len(request.messages):>9}  {len(tool_messages):>4}  "
        f"{cleared:>8}  {count_tokens_approximately(request.messages):>7}  {shapes}"
    )

    return handler(request)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[warehouse_report],
    system_prompt=(
        "Вы помощник руководителя склада. Отвечайте по-русски, одной фразой. "
        "Цифры берите инструментом, по памяти не отвечайте."
    ),
    middleware=[
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=300, keep=1)],
        ),
        observer,
    ],
    checkpointer=InMemorySaver(),
)

CONFIG = {"configurable": {"thread_id": "edit-1"}}

QUESTIONS = [
    "Сколько брака за неделю на складе Казань?",
    "А на складе Омск?",
    "А на складе Пермь?",
]

print(
    f"{'вызов':>5}  {'сообщений':>9}  {'tool':>4}  {'очищено':>8}  "
    f"{'оценка':>7}  длины результатов инструмента"
)

for question in QUESTIONS:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        CONFIG,
    )

print()
print("ПОСЛЕДНИЙ ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))

print()
print("В THREAD ПОСЛЕ ТРЁХ ВОПРОСОВ:")
for message in result["messages"]:
    if message.type == "tool":
        print(f"  tool  {len(message.text):>4} знаков  {message.text[:38]}")
