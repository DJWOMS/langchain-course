"""Пример 1 урока 12: история одного thread, соседний thread её не получает.

Агент собран с чекпойнтером в памяти. Три запроса: два уходят в thread
"seat-1", третий в thread "seat-2". Между вызовами код не передаёт историю
руками, её подставляет чекпойнтер по thread_id из конфигурации вызова.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

agent = create_agent(
    model=build_model(temperature=0, max_tokens=256),
    tools=[],
    system_prompt=(
        "Вы помощник склада. Отвечайте по-русски, одной короткой фразой. "
        "Если нужного факта в диалоге не было, так и скажите."
    ),
    checkpointer=InMemorySaver(),
)


def ask(question, thread_id):
    """Один запрос в названный thread. В invoke уходит только новая реплика."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )
    answer = result["messages"][-1].text.replace("\n", " ")
    print(f"[{thread_id}] вопрос: {question}")
    print(f"[{thread_id}] ответ:  {answer}")
    print(f"[{thread_id}] сообщений в thread после вызова: {len(result['messages'])}")
    print()


ask("Запомните: моя смена начинается в 7:40.", "seat-1")
ask("Во сколько начинается моя смена?", "seat-1")
ask("Во сколько начинается моя смена?", "seat-2")
