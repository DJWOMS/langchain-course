"""Пример 3 урока 3: одно и то же содержимое в трёх видах.

Сети здесь нет и ключ не нужен. Сообщения собраны руками ровно в том виде,
в каком их присылают два разных провайдера, и дальше видно, что с ними делает
свойство content_blocks.

Запуск: python 03_content_blocks.py
"""

from langchain.messages import AIMessage


def show(title, message):
    """Печатает три взгляда на одно и то же сообщение."""
    print(title)
    print("  ТИП content:    ", type(message.content).__name__)
    print("  content:        ", message.content)
    print("  content_blocks: ", message.content_blocks)
    print("  text:           ", repr(message.text))
    print()


# 1. Содержимое строкой. Так выглядит ответ обычной текстовой модели.
show("СТРОКА", AIMessage("Париж, столица Франции."))

# 2. Блоки в формате Anthropic: рассуждение зовётся thinking.
show(
    "БЛОКИ ПРОВАЙДЕРА ANTHROPIC",
    AIMessage(
        content=[
            {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
            {"type": "text", "text": "..."},
        ],
        response_metadata={"model_provider": "anthropic"},
    ),
)

# 3. То же самое в формате OpenAI: рассуждение зовётся reasoning и лежит в summary.
show(
    "БЛОКИ ПРОВАЙДЕРА OPENAI",
    AIMessage(
        content=[
            {
                "type": "reasoning",
                "id": "rs_abc123",
                "summary": [
                    {"type": "summary_text", "text": "summary 1"},
                    {"type": "summary_text", "text": "summary 2"},
                ],
            },
            {"type": "text", "text": "...", "id": "msg_abc123"},
        ],
        response_metadata={"model_provider": "openai"},
    ),
)
