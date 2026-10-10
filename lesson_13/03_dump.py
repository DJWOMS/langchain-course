"""Пример 3 урока 13: память, в которую пишут всё подряд.

Модель здесь не нужна, важно само правило записи: каждая реплика пользователя
уходит в хранилище отдельной записью с новым ключом. Так выглядит "пусть
агент запоминает диалог", если не задумываться об отборе.

После трёх сессий пример печатает число записей и знаков в них, пять записей,
которые уйдут в промпт при limit=5, и записи про город и про кофе.
"""

import uuid

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
NAMESPACE = ("u-101", "memories")

SESSIONS = {
    "понедельник": [
        "Работаю в Казани",
        "Кофе пью без сахара",
        "Сегодня болит голова, отвечайте покороче",
        "Веду склад запчастей",
    ],
    "среда": [
        "Кофе пью без сахара",
        "В офисе холодно",
        "Завтра встреча с поставщиком в десять",
        "Смена с 8 до 17",
    ],
    "пятница": [
        "Переехал в Омск",
        "Кофе теперь пью с молоком",
        "Голова прошла",
        "Встреча с поставщиком прошла нормально",
    ],
}

for day, turns in SESSIONS.items():
    for text in turns:
        # Ключ новый на каждую реплику, поэтому ничего не перезаписывается.
        store.put(NAMESPACE, str(uuid.uuid4()), {"text": text, "session": day})

everything = store.search(NAMESPACE, limit=100)
print("ЗАПИСЕЙ В ПАМЯТИ:", len(everything))
print("ЗНАКОВ В ТЕКСТАХ:", sum(len(item.value["text"]) for item in everything))
print()

print("ЧТО УЙДЁТ В ПРОМПТ ПРИ search(..., limit=5):")
for item in store.search(NAMESPACE, limit=5):
    print(f"  [{item.value['session']}] {item.value['text']}")
print()

city_records = [item for item in everything if "Казани" in item.value["text"] or "Омск" in item.value["text"]]
print("ЗАПИСИ ПРО ГОРОД:")
for item in city_records:
    print(f"  [{item.value['session']}] {item.value['text']}")
print()

coffee_records = [item for item in everything if "Кофе" in item.value["text"]]
print("ЗАПИСИ ПРО КОФЕ:", len(coffee_records))
for item in coffee_records:
    print(f"  [{item.value['session']}] {item.value['text']}")
