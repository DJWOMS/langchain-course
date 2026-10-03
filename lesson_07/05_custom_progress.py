"""Пример 5 урока 7: свои сообщения о прогрессе из инструмента.

Пока работает инструмент, модель ничего не присылает, и для пользователя это
выглядит как зависание. Режим custom даёт инструменту собственный канал в поток: что угодно,
что вы передали в writer, приходит в цикл потока.
"""

from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from course_model import build_model

PAGES = 3


def load_orders(day: str) -> str:
    """Загружает заказы за указанный день и возвращает сводку."""
    writer = get_stream_writer()

    loaded = 0
    for page in range(1, PAGES + 1):
        loaded += 40
        writer({"stage": "loading", "page": page, "of": PAGES, "rows": loaded})

    writer({"stage": "done", "rows": loaded})
    return f"за {day} заказов {loaded}, из них оплачено 97"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[load_orders],
    system_prompt="Отвечайте по-русски, одним предложением.",
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Сколько заказов за вчера?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "custom":
        event = chunk["data"]
        if event["stage"] == "loading":
            print(f"[прогресс] страница {event['page']} из {event['of']}, строк {event['rows']}")
        else:
            print(f"[прогресс] загрузка закончена, строк {event['rows']}")

    elif chunk["type"] == "updates":
        for node, update in chunk["data"].items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for message in messages:
                calls = getattr(message, "tool_calls", None)
                if calls:
                    print(f"[шаг] {node}: вызов {[call['name'] for call in calls]}")
                else:
                    text = message.text.replace("\n", " ")
                    print(f"[шаг] {node}: {text[:70]!r}")
