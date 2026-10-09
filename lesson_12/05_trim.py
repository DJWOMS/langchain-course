"""Пример 5 урока 12: обрезка истории перед вызовом модели.

Middleware before_model оставляет в thread последние KEEP сообщений и удаляет
всё, что старше. Диалог тот же, что в примере 4, и последний вопрос проверяет,
пережил ли обрезку факт из первой реплики.
"""

from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from course_model import build_model

KEEP = 4


@before_model
def trim_history(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Оставляет в thread последние KEEP сообщений, начиная с реплики человека."""
    messages = state["messages"]
    if len(messages) <= KEEP:
        return None

    kept = messages[-KEEP:]
    # История, начатая ответом модели или результатом инструмента, ломает часть
    # провайдеров: первым сообщением они ждут реплику человека.
    while kept and kept[0].type != "human":
        kept = kept[1:]
    if not kept:
        return None

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}


agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[],
    system_prompt=(
        "Вы помощник склада. Отвечайте по-русски, тремя-четырьмя развёрнутыми "
        "фразами. Если нужного факта в диалоге не было, так и скажите."
    ),
    middleware=[trim_history],
    checkpointer=InMemorySaver(),
)

CONFIG = {"configurable": {"thread_id": "trim-1"}}

QUESTIONS = [
    "Запомните: моя смена начинается в 7:40.",
    "Опишите порядок приёмки паллет на складе.",
    "А как оформлять брак, найденный при приёмке?",
    "Во сколько начинается моя смена?",
]

print(f"{'вызов':>5}  {'сообщений в thread':>18}  первое сообщение thread")

for number, question in enumerate(QUESTIONS, start=1):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        CONFIG,
    )
    first = result["messages"][0].text.replace("\n", " ")
    if len(first) > 40:
        first = first[:37] + "..."
    print(f"{number:>5}  {len(result['messages']):>18}  {first!r}")

print()
print("ПОСЛЕДНИЙ ВОПРОС:", QUESTIONS[-1])
print("ПОСЛЕДНИЙ ОТВЕТ: ", result["messages"][-1].text.replace("\n", " "))
