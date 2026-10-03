"""Пример 4 урока 7: режим messages, токены модели и метаданные chunk.

Режим messages отдаёт пары (chunk сообщения, метаданные). Метаданные говорят, из
какого узла графа пришёл chunk. Аргументы инструмента приходят обрывками текста и
собираются в JSON только к концу вызова.
"""

from collections import Counter

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски, одним предложением.",
)

by_node = Counter()
by_type = Counter()
arg_pieces = []
metadata_keys = []
shown = 0

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] != "messages":
        continue

    token, metadata = chunk["data"]
    node = metadata.get("langgraph_node", "?")
    by_node[node] += 1
    by_type[type(token).__name__] += 1
    if not metadata_keys:
        metadata_keys = sorted(metadata)

    blocks = [block["type"] for block in token.content_blocks]

    if isinstance(token, AIMessageChunk):
        for piece in token.tool_call_chunks:
            arg_pieces.append(piece["args"])

    # Пустые chunks рассуждения пропускаются, см. пример 1.
    if blocks and shown < 6:
        shown += 1
        print(f"  {shown}. узел {node}, {type(token).__name__}, блоки {blocks}, text={token.text!r}")

print()
print("CHUNKS ПО УЗЛАМ:", dict(by_node))
print("ТИПЫ СООБЩЕНИЙ В ПОТОКЕ:", dict(by_type))
print("КЛЮЧИ МЕТАДАННЫХ:", metadata_keys)

if arg_pieces:
    print("АРГУМЕНТЫ ИНСТРУМЕНТА ПО CHUNKS:", arg_pieces)
    print("ОНИ ЖЕ, СКЛЕЕННЫЕ:", "".join(piece or "" for piece in arg_pieces))
else:
    # Защитная ветка: модель обошлась без инструмента или отдала вызов целиком.
    print("АРГУМЕНТЫ ИНСТРУМЕНТА ПО CHUNKS: chunks не было")
