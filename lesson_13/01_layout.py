"""Пример 1 урока 13: раскладка хранилища по пространствам имён.

Записи кладутся в хранилище из кода, модель не нужна. Пример показывает
запросы к хранилищу: всё, что записано про человека, одна категория, отбор
по содержимому и список заведённых пространств имён.

Отдельно показан потолок выдачи: аргумент limit обрезает список, и в ответе
нет признака того, что записей было больше.
"""

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Пространство имён, это кортеж строк. Здесь первый уровень, владелец записи,
# второй, вид памяти. Уровней может быть сколько угодно.
FACTS = [
    ("city", {"text": "Работает в Омске", "source": "диалог"}),
    ("role", {"text": "Ведёт склад запчастей", "source": "анкета"}),
    ("shift", {"text": "Смена с 8 до 17", "source": "анкета"}),
    ("team", {"text": "В подчинении четыре человека", "source": "диалог"}),
]

PREFERENCES = [
    ("answer_style", {"text": "Отвечать без списков", "source": "диалог"}),
]

for key, value in FACTS:
    store.put(("u-101", "facts"), key, value)

for key, value in PREFERENCES:
    store.put(("u-101", "preferences"), key, value)

# Второй пользователь, чтобы было видно, что записи не перемешиваются.
store.put(("u-202", "facts"), "city", {"text": "Работает в Казани", "source": "анкета"})


def show(title, items):
    """Печатает заголовок и найденные записи одной строкой на запись."""
    print(title)
    for item in items:
        print(f"  {'/'.join(item.namespace)} :: {item.key} :: {item.value['text']}")
    print()


show("ВСЁ ПРО ОДНОГО ЧЕЛОВЕКА, поиск по префиксу ('u-101',):", store.search(("u-101",)))

show("ТОЛЬКО ФАКТЫ, полное пространство имён:", store.search(("u-101", "facts")))

show(
    "ОТБОР ПО СОДЕРЖИМОМУ, filter по полю source:",
    store.search(("u-101", "facts"), filter={"source": "анкета"}),
)

show(
    "ПОТОЛОК ВЫДАЧИ, limit=2 при четырёх записях:",
    store.search(("u-101", "facts"), limit=2),
)

print("ПРОСТРАНСТВА ИМЁН ОДНОГО ЧЕЛОВЕКА:")
for namespace in store.list_namespaces(prefix=("u-101",), max_depth=2):
    print(" ", namespace)
print()

print("ВСЕ ПРОСТРАНСТВА ИМЁН ХРАНИЛИЩА:")
for namespace in store.list_namespaces():
    print(" ", namespace)
print()

item = store.get(("u-101", "facts"), "city")
print("ОДНА ЗАПИСЬ ЦЕЛИКОМ:")
print(item.dict())
