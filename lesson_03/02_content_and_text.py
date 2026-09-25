"""Пример 2 урока 3: что лежит в ответе модели, кроме текста.

Запуск: python 02_content_and_text.py
"""

from course_model import build_model

model = build_model(temperature=0)

response = model.invoke("Ответьте одним словом: столица Японии")

print("ТИП СООБЩЕНИЯ:  ", type(response).__name__)
print("ТИП content:    ", type(response.content).__name__)
print("content:        ", repr(response.content))
print("ТИП text:       ", type(response.text).__name__)
print("text:           ", repr(response.text))
print("content_blocks: ", response.content_blocks)
print("tool_calls:     ", response.tool_calls)
print("id:             ", response.id)
print()
print("КЛЮЧИ response_metadata:", sorted(response.response_metadata))
print("usage_metadata:         ", response.usage_metadata)
