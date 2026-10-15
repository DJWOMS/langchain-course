"""Пример 4 урока 17: пауза по условию.

Предикат when получает ToolCallRequest и читает аргументы вызова.
Запись внутрь рабочей папки проходит без паузы, запись за её пределы
останавливает запуск.

Задание запускается дважды в двух threads, меняется только путь в задании.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from course_model import build_model

EXECUTED = []
WORKSPACE = "/workspace/"


def write_file(path: str, content: str) -> str:
    """Записывает текст в файл по указанному пути."""
    EXECUTED.append(f"write_file({path})")
    return f"записано {len(content)} знаков в {path}"


def outside_workspace(request: ToolCallRequest) -> bool:
    """Возвращает True для записи за пределы рабочей папки, такой вызов встаёт на паузу."""
    path = request.tool_call["args"].get("path", "")
    return not path.startswith(WORKSPACE)


agent = create_agent(
    model=build_model(temperature=0),
    tools=[write_file],
    system_prompt="Вы помощник разработчика. Отвечайте по-русски, одной фразой.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {
                    "allowed_decisions": ["approve", "reject"],
                    "when": outside_workspace,
                },
            },
        ),
    ],
    checkpointer=InMemorySaver(),
)


def run(thread_id, task):
    """Запуск одного задания: печатает, встала пауза или нет."""
    EXECUTED.clear()
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config,
        version="v2",
    )

    print(f"THREAD {thread_id}")
    print("  задание:          ", task)
    print("  запуск остановлен:", bool(result.interrupts))

    if result.interrupts:
        action = result.interrupts[0].value["action_requests"][0]
        print("  агент просит:     ", action["name"], action["args"]["path"])
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "reject", "message": "Путь вне рабочей папки."}]}),
            config,
            version="v2",
        )

    print("  выполнено:        ", EXECUTED)
    print("  ответ агента:     ", result.value["messages"][-1].text.replace("\n", " "))
    print()


run("inside", "Запишите строку hello в файл /workspace/notes.txt")
run("outside", "Запишите строку hello в файл /etc/notes.txt")
