"""Пример 1 урока 8: поток событий у одной модели, без агента.

Метод stream_events(version="v3") у модели возвращает не итератор, а объект
ChatModelStream с проекциями: .text, .reasoning, .tool_calls,
.output. Итоговое сообщение и расход токенов лежат в .output.
"""

from course_model import build_model

model = build_model(temperature=0, max_tokens=1024)

stream = model.stream_events(
    "Назовите три версии протокола HTTP, по одной в строке, без пояснений.",
    version="v3",
)

print("ТИП ОБЪЕКТА ПРОГОНА:", type(stream).__name__)

deltas = []
for delta in stream.text:
    deltas.append(delta)

print("ДЕЛЬТ ПРИШЛО:", len(deltas))
print("ПЕРВЫЕ ТРИ ДЕЛЬТЫ:", deltas[:3])

final = stream.output
print("ТИП ИТОГОВОГО СООБЩЕНИЯ:", type(final).__name__)
print("ЗНАКОВ В ОТВЕТЕ:", len(final.text))
print("РАСХОД:", final.usage_metadata)
print("ОТВЕТ:")
print(final.text)
