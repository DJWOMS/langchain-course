"""Пример 7 урока 3: картинка внутри сообщения.

Первая половина работает без сети и показывает, что фреймворк кладёт в
сообщение. Вторая отправляет это сообщение модели курса и печатает, что ответил
провайдер: ответ модели или отказ. Отказ здесь нормальный исход примера, ошибки
в коде он не означает.

Картинка служебная: квадрат восемь на восемь точек в формате PNG, записанный
в base64 прямо в файле, чтобы пример не зависел от чужого сайта.

Запуск: python 07_multimodal.py
"""

from langchain.messages import HumanMessage

from course_model import build_model

# Квадрат 8x8 в формате PNG, закодированный base64: все 64 точки цвета
# RGB (220, 50, 47), красный.
RED_SQUARE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR4nGO4Y6SPFTEM"
    "LQkAItlPQVLZ9OgAAAAASUVORK5CYII="
)


def preview(block):
    """Обрезает длинные строки внутри блока: base64 занял бы весь экран."""
    if not isinstance(block, dict):
        return repr(block)

    trimmed = {}
    for key, value in block.items():
        if isinstance(value, str) and len(value) > 24:
            trimmed[key] = value[:24] + "..."
        else:
            trimmed[key] = value
    return trimmed


message = HumanMessage(
    content_blocks=[
        {"type": "text", "text": "Какого цвета этот квадрат? Ответьте одним словом."},
        {"type": "image", "base64": RED_SQUARE_PNG, "mime_type": "image/png"},
    ]
)

print("ЧТО ЛЕЖИТ В content:")
if isinstance(message.content, str):
    print(" ", repr(message.content))
else:
    for block in message.content:
        print(" ", preview(block))

print()
print("ЧТО ОТДАЁТ content_blocks:")
for block in message.content_blocks:
    print(" ", preview(block))

print()
print("ЗАПРОС К МОДЕЛИ КУРСА")

model = build_model(temperature=0)

try:
    response = model.invoke([message])
    print("  ОТВЕТ:", response.text.strip())
    print("  РАСХОД:", response.usage_metadata)
except Exception as error:  # noqa: BLE001
    print(f"  отказ: {type(error).__name__}")
    print(f"  текст: {error}")
