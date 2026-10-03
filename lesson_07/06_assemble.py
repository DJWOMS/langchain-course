"""Пример 6 урока 7: chunk это не сообщение.

Разобрать вызов инструмента по одному chunk нельзя: в chunk лежит обрывок JSON.
Целое собирается двумя способами, и оба показаны здесь: сложением chunks в цикле
и чтением готового сообщения из режима updates.
"""

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в городе."""
    return f"В городе {city} всегда солнечно!"


def show_assembled(message: AIMessageChunk, reason: str) -> None:
    """Печатает то, что видно только у собранного сообщения."""
    text = message.text.replace("\n", " ")
    print(f"СОБРАННОЕ СООБЩЕНИЕ ({reason})")
    print(f"  вызовы инструментов: {[call['name'] for call in message.tool_calls]}")
    print(f"  аргументы: {[call['args'] for call in message.tool_calls]}")
    print(f"  текст: {text[:70]!r}")
    print(f"  расход: {message.usage_metadata}")


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски, одним предложением.",
)

full = None
partial_calls = 0

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, _metadata = chunk["data"]
        if not isinstance(token, AIMessageChunk):
            continue

        if token.tool_call_chunks:
            partial_calls += 1
            piece = token.tool_call_chunks[0]
            print(f"chunk вызова: name={piece['name']!r}, args={piece['args']!r}")

        full = token if full is None else full + token

        if token.chunk_position == "last":
            show_assembled(full, "по признаку chunk_position")
            full = None

    elif chunk["type"] == "updates":
        for node, update in chunk["data"].items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for message in messages:
                if isinstance(message, AIMessage) and message.tool_calls:
                    print(f"из режима updates, узел {node}: {message.tool_calls}")
                elif isinstance(message, ToolMessage):
                    print(f"из режима updates, узел {node}: ответ {message.text!r}")

if full is not None:
    # Защитная ветка: класс модели обошёл базовый класс LangChain,
    # и последний chunk пришёл без признака chunk_position.
    show_assembled(full, "признака конца не было")

print(f"chunks с обрывками вызова: {partial_calls}")
