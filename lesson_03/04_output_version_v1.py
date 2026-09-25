"""Пример 4 урока 3: как положить стандартные блоки внутрь content.

По умолчанию content_blocks это отдельное свойство, а content остаётся таким,
каким его прислал провайдер. Параметр output_version="v1" меняет это: стандартные
блоки попадают в сам content, и сообщение можно отдавать наружу как есть.

Запуск: python 04_output_version_v1.py
"""

from course_model import build_model

QUESTION = "Ответьте одним словом: столица Италии"

default_model = build_model(temperature=0)
v1_model = build_model(temperature=0, output_version="v1")

default_response = default_model.invoke(QUESTION)
v1_response = v1_model.invoke(QUESTION)

print("ПО УМОЛЧАНИЮ")
print("  ТИП content:    ", type(default_response.content).__name__)
print("  content:        ", repr(default_response.content))
print("  content_blocks: ", default_response.content_blocks)
print()
print('С output_version="v1"')
print("  ТИП content:    ", type(v1_response.content).__name__)
print("  content:        ", repr(v1_response.content))
print("  content_blocks: ", v1_response.content_blocks)
print()
print("ТЕКСТ У ОБОИХ ОДИНАКОВЫЙ:", default_response.text.strip() == v1_response.text.strip())
