"""Пример 6 урока 13: память на диске, срок годности и ручная чистка.

Хранилище лежит файлом SQLite, поэтому записанное одним соединением достаётся
другим, как достанется завтрашним запуском программы.

Разовая заметка пишется с аргументом ttl, а метод sweep_ttl удаляет всё, чему
срок вышел. Профиль пишется без ttl и остаётся. Устаревшее руками убирается
методом delete.

Запуск занимает около пяти секунд: пример ждёт, пока истечёт срок заметки.
"""

import time
from pathlib import Path

from langgraph.store.sqlite import SqliteStore

DB_PATH = Path(__file__).with_name("memory.db")

# Пример начинается с чистого файла, иначе его вывод зависел бы от прошлых запусков.
DB_PATH.unlink(missing_ok=True)


def show(title, items):
    """Печатает заголовок и записи одной строкой на запись."""
    print(title)
    for item in items:
        print(f"  {'/'.join(item.namespace)} :: {item.key} :: {item.value['value']}")
    print()


with SqliteStore.from_conn_string(str(DB_PATH)) as store:
    # setup() создаёт таблицы. Если его не вызвать, SqliteStore выполнит его сам
    # при первом запросе, поэтому второе соединение ниже обходится без него.
    store.setup()

    store.put(("u-101", "profile"), "city", {"value": "Омск"})
    store.put(("u-101", "profile"), "answer_style", {"value": "без списков"})

    # ttl задаётся в минутах, 0.05 это три секунды. В продакшне здесь стоят
    # недели: это срок, после которого заметка перестаёт быть правдой.
    store.put(
        ("u-101", "notes"),
        "promo-september",
        {"value": "Просил напомнить про акцию поставщика"},
        ttl=0.05,
    )
    print("ФАЙЛ ЗАПИСАН:", DB_PATH.name)
    print()

# Соединение закрыто. Дальше с тем же файлом работает другое соединение, так
# же к нему подключится завтрашний запуск программы.
with SqliteStore.from_conn_string(str(DB_PATH)) as store:
    show("ПОСЛЕ ПЕРЕОТКРЫТИЯ ФАЙЛА:", store.search(("u-101",)))

    time.sleep(5)

    print("УДАЛЕНО ПО СРОКУ ГОДНОСТИ:", store.sweep_ttl())
    print()
    show("ОСТАЛОСЬ:", store.search(("u-101",)))

    store.delete(("u-101", "profile"), "answer_style")
    show("ПОСЛЕ РУЧНОГО УДАЛЕНИЯ ОДНОЙ ЗАПИСИ:", store.search(("u-101",)))

    print("ПРОСТРАНСТВА ИМЁН, КОТОРЫЕ ОСТАЛИСЬ:", store.list_namespaces(prefix=("u-101",)))

# Пример удаляет файл, чтобы повторный запуск начинался с того же места.
DB_PATH.unlink(missing_ok=True)
