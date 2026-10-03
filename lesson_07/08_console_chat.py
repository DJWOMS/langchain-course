"""Пример 8 урока 7: что из потока показывают пользователю.

Три режима сразу: шаги агента, токены модели и прогресс инструмента. Пользователь
видит текст ответа по мере готовности, строку статуса на вызове инструмента и
прогресс загрузки. Обрывки JSON с аргументами в окно не попадают.

Заодно печатается то, ради чего поток и заводят: сколько секунд прошло до первого
показанного знака и сколько до конца ответа.
"""

import time

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk
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

    return f"за {day} заказов {loaded}, из них оплачено 97, отменено 6"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[load_orders],
    system_prompt="Вы помощник аналитика. Отвечайте по-русски, до двух предложений.",
)

started = time.perf_counter()
first_text_at = None

try:
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Что со вчерашними заказами?"}]},
        stream_mode=["updates", "messages", "custom"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            # В поток режима messages попадают и готовые ToolMessage, и обрывки
            # вызовов. Пользователю показывается только текст ответа модели.
            if not isinstance(token, AIMessageChunk) or not token.text:
                continue
            if metadata.get("langgraph_node") != "model":
                continue
            if first_text_at is None:
                first_text_at = time.perf_counter() - started
            print(token.text, end="", flush=True)

        elif chunk["type"] == "custom":
            event = chunk["data"]
            print(f"\n[загрузка] страница {event['page']} из {event['of']}, строк {event['rows']}")

        elif chunk["type"] == "updates":
            for _node, update in chunk["data"].items():
                messages = update.get("messages", []) if isinstance(update, dict) else []
                for message in messages:
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for call in message.tool_calls:
                            print(f"\n[инструмент] {call['name']}({call['args']})")

except Exception as error:
    # Отказ провайдера посреди потока: часть ответа уже показана пользователю.
    print(f"\n[сбой] {type(error).__name__}: {error}")

print()
if first_text_at is None:
    # Защитная ветка: текста в потоке не было вовсе, показывать было нечего.
    print("первый знак ответа: текста в потоке не было")
else:
    print(f"первый знак ответа через: {first_text_at:.2f} с")
print(f"весь ответ через: {time.perf_counter() - started:.2f} с")
