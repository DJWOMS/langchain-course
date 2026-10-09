"""Пример 3 урока 12: thread на диске и режим долговечности записи.

SqliteSaver держит checkpoints в файле. Программа открывает соединение, задаёт
вопрос и закрывает соединение, а потом открывает тот же файл заново, другим
соединением и другим объектом агента. Thread при этом остаётся на месте.

Первый сеанс пишет в режиме durability="sync", второй в режиме "exit", и по
числу checkpoints после каждого видна разница.

Файл session.sqlite удаляется на старте, чтобы вывод не зависел от того,
сколько раз пример уже запускали.

SqliteSaver ставится отдельным пакетом: pip install langgraph-checkpoint-sqlite
"""

from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from course_model import build_model

DB_PATH = Path(__file__).with_name("session.sqlite")
DB_PATH.unlink(missing_ok=True)

CONFIG = {"configurable": {"thread_id": "shift-1"}}
SYSTEM_PROMPT = (
    "Вы помощник склада. Отвечайте по-русски, одной короткой фразой. "
    "Если нужного факта в диалоге не было, так и скажите."
)


def session(question, durability):
    """Отдельный сеанс работы: своё соединение, свой агент, общий файл thread."""
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        agent = create_agent(
            model=build_model(temperature=0, max_tokens=256),
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            CONFIG,
            durability=durability,
        )
        checkpoints = len(list(agent.get_state_history(CONFIG)))

    answer = result["messages"][-1].text.replace("\n", " ")
    print(f"вопрос:      {question}")
    print(f"ответ:       {answer}")
    print(f"durability:  {durability}")
    print(f"checkpoints в thread: {checkpoints}")
    print(f"файл thread:          {DB_PATH.name}, {DB_PATH.stat().st_size} байт")
    print()


session("Запомните: моя смена начинается в 7:40.", "sync")
session("Во сколько начинается моя смена?", "exit")

with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
    tuples = list(checkpointer.list(CONFIG))
    print("ЗАПИСИ THREAD В ФАЙЛЕ")
    print("  записей в файле:", len(tuples))
    print("  самая свежая:   ", tuples[0].config["configurable"]["checkpoint_id"])
    print("  самая старая:   ", tuples[-1].config["configurable"]["checkpoint_id"])
