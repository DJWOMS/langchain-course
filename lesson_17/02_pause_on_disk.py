"""Пример 2 урока 17: пауза хранится в файле чекпойнтера и переживает закрытие соединения.

Первый сеанс доводит агента до паузы и закрывает соединение с файлом чекпойнтера.
Второй сеанс открывает тот же файл другим соединением, собирает агента
заново и читает из чекпойнтера, на каком узле остановлен запуск. Возобновление идёт
из второго сеанса, тем же thread_id.

Файл pause.sqlite удаляется на старте, чтобы вывод не зависел от того,
сколько раз пример уже запускали.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from course_model import build_model

DB_PATH = Path(__file__).with_name("pause.sqlite")
DB_PATH.unlink(missing_ok=True)

CONFIG = {"configurable": {"thread_id": "cleanup-1"}}
EXECUTED = []


def delete_backups(older_than_days: int) -> str:
    """Удаляет резервные копии старше указанного числа дней."""
    EXECUTED.append(f"delete_backups({older_than_days})")
    return f"удалено 14 копий старше {older_than_days} дней"


def build_agent(checkpointer):
    """Собирает агента с паузой на удаление. Сборка одна и та же в обоих сеансах."""
    return create_agent(
        model=build_model(temperature=0),
        tools=[delete_backups],
        system_prompt="Вы администратор хранилища. Отвечайте по-русски, одной фразой.",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"delete_backups": {"allowed_decisions": ["approve", "reject"]}},
            ),
        ],
        checkpointer=checkpointer,
    )


print("СЕАНС 1: агент доходит до паузы")

with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
    agent = build_agent(checkpointer)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Удалите резервные копии старше 30 дней."}]},
        CONFIG,
        version="v2",
    )
    print("  запуск остановлен:", bool(result.interrupts))
    print("  выполнено:        ", EXECUTED)

print("  соединение закрыто, файл чекпойнтера:", DB_PATH.name, DB_PATH.stat().st_size, "байт")
print()

print("СЕАНС 2: другое соединение, другой объект агента, тот же thread_id")

with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
    agent = build_agent(checkpointer)
    snapshot = agent.get_state(CONFIG)

    print("  следующий узел:   ", snapshot.next)
    print("  ждущих прерываний:", len(snapshot.interrupts))

    action = snapshot.interrupts[0].value["action_requests"][0]
    print("  агент просит:     ", action["name"], action["args"])

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        CONFIG,
        version="v2",
    )

    print("  выполнено:        ", EXECUTED)
    print("  ответ агента:     ", result.value["messages"][-1].text.replace("\n", " "))
