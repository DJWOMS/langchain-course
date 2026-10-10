"""Пример 4 урока 13: те же сессии, но запись идёт в профиль.

От примера 3 этот пример отличается схемой и ключом. Набор полей закрыт схемой,
и всё, чего в схеме нет, в память не попадает. У профиля один ключ, поэтому
новое значение поля заменяет прежнее, и второй записи рядом не появляется.

Список PROPOSALS, это поля и значения, которые модель могла бы предложить к
записи по репликам трёх сессий. Здесь он задан данными, чтобы вывод не
менялся от запуска к запуску.
"""

from pydantic import BaseModel, ConfigDict, ValidationError

from langgraph.store.memory import InMemoryStore

NAMESPACE = ("u-101", "profile")
KEY = "main"


class UserProfile(BaseModel):
    """Закрытый набор того, что агенту разрешено помнить о человеке."""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    role: str | None = None
    shift: str | None = None
    drink: str | None = None
    answer_style: str | None = None


PROPOSALS = [
    ("понедельник", "city", "Казань"),
    ("понедельник", "drink", "кофе без сахара"),
    ("понедельник", "mood", "болит голова"),
    ("понедельник", "role", "склад запчастей"),
    ("среда", "drink", "кофе без сахара"),
    ("среда", "office_temperature", "холодно"),
    ("среда", "next_meeting", "завтра в десять"),
    ("среда", "shift", "с 8 до 17"),
    ("пятница", "city", "Омск"),
    ("пятница", "drink", "кофе с молоком"),
    ("пятница", "mood", "голова прошла"),
]

store = InMemoryStore()


def save_field(field, value):
    """Кладёт одно поле в профиль. Возвращает пару: принято ли и почему."""
    item = store.get(NAMESPACE, KEY)
    current = dict(item.value) if item else {}
    candidate = {**current, field: value}

    try:
        profile = UserProfile(**candidate)
    except ValidationError:
        return False, "поля нет в схеме"

    if current.get(field) == value:
        return False, "значение не изменилось"

    store.put(NAMESPACE, KEY, profile.model_dump(exclude_none=True))
    return True, "записано"


for day, field, value in PROPOSALS:
    accepted, reason = save_field(field, value)
    mark = "+" if accepted else "-"
    print(f"{mark} [{day}] {field} = {value} :: {reason}")

print()
item = store.get(NAMESPACE, KEY)
print("ЗАПИСЕЙ В ПРОСТРАНСТВЕ ИМЁН:", len(store.search(NAMESPACE)))
print("ПРОФИЛЬ:")
for field, value in item.value.items():
    print(f"  {field}: {value}")
print()
print("СОЗДАН:  ", item.created_at.isoformat())
print("ИЗМЕНЁН: ", item.updated_at.isoformat())
