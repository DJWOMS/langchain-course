"""Пример 2 урока 12: checkpoint состояния thread.

Один запрос, на который агенту нужен инструмент. После вызова читается то,
что осталось в thread: последний checkpoint и вся история checkpoints. Видно,
на каких шагах графа чекпойнтер делает запись.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

SHIFTS = {"Казань": "7:40", "Омск": "9:00"}


def shift_start(city: str) -> str:
    """Возвращает время начала смены на складе в городе."""
    time = SHIFTS.get(city)
    if time is None:
        return f"Склада в городе {city} нет."
    return f"Смена на складе в городе {city} начинается в {time}."


agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[shift_start],
    system_prompt="Вы помощник склада. Отвечайте по-русски, одной короткой фразой.",
    checkpointer=InMemorySaver(),
)

CONFIG = {"configurable": {"thread_id": "shift-1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "Во сколько смена в Казани?"}]},
    CONFIG,
)

snapshot = agent.get_state(CONFIG)

print("ПОСЛЕДНИЙ CHECKPOINT")
print("  ключи values:  ", sorted(snapshot.values))
print("  сообщений:     ", len(snapshot.values["messages"]))
print("  next:          ", snapshot.next)
print("  шаг:           ", snapshot.metadata["step"])
print("  источник:      ", snapshot.metadata["source"])
print("  checkpoint_id: ", snapshot.config["configurable"]["checkpoint_id"])
print("  checkpoint_ns: ", repr(snapshot.config["configurable"]["checkpoint_ns"]))
print("  создан:        ", snapshot.created_at)
# У самого первого checkpoint в thread родителя нет, parent_config равен None.
parent = snapshot.parent_config
print(
    "  родитель:      ",
    parent["configurable"]["checkpoint_id"] if parent else None,
)
print()

print("ИСТОРИЯ CHECKPOINTS, сверху самый свежий")
print(f"  {'шаг':>4}  {'источник':<8}  {'следующий узел':<16}  сообщений")
for item in agent.get_state_history(CONFIG):
    following = ", ".join(item.next) or "-"
    count = len(item.values.get("messages", []))
    print(
        f"  {item.metadata['step']:>4}  {item.metadata['source']:<8}  "
        f"{following:<16}  {count}"
    )
