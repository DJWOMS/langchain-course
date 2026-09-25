"""Пример 5 урока 3: где в сообщении лежит рассуждение модели.

Два места, куда смотреть: блоки типа reasoning в content_blocks и счётчик
output_token_details["reasoning"] в usage_metadata. Любого из них в ответе
может не оказаться, поэтому у обоих есть ветка на этот случай.

Запуск: python 05_reasoning.py
"""

from course_model import build_model

QUESTION = (
    "У Ани в два раза больше книг, чем у Бори, а вместе у них 51 книга. "
    "Сколько книг у Ани?"
)

model = build_model(temperature=0)
response = model.invoke(QUESTION)

reasoning_blocks = [b for b in response.content_blocks if b["type"] == "reasoning"]
text_blocks = [b for b in response.content_blocks if b["type"] == "text"]

print("ТИПЫ БЛОКОВ:       ", [block["type"] for block in response.content_blocks])
print("БЛОКОВ РАССУЖДЕНИЯ:", len(reasoning_blocks))
print("БЛОКОВ ТЕКСТА:     ", len(text_blocks))
print()

if reasoning_blocks:
    joined = " ".join(block.get("reasoning", "") for block in reasoning_blocks)
    print("РАССУЖДЕНИЕ, ПЕРВЫЕ 300 ЗНАКОВ:")
    print(" ", joined[:300])
else:
    print("Блоков рассуждения в сообщении нет.")

print()
print("ОТВЕТ:", response.text.strip()[:200])
print()

usage = response.usage_metadata or {}
details = usage.get("output_token_details") or {}

print("ВХОД: ", usage.get("input_tokens"))
print("ВЫХОД:", usage.get("output_tokens"))
print("ИЗ НИХ НА РАССУЖДЕНИЕ:", details.get("reasoning", "провайдер не разделил"))
