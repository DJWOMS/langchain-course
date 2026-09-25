"""Пример 8 урока 3: сохранить диалог на диск и продолжить его завтра.

dumpd превращает сообщение в обычный словарь, load собирает из словаря объект
сообщения обратно. Между ними стоит любой способ хранения, здесь это файл JSON.

Второй вызов load показывает форму, которую стоит писать всегда: список
разрешённых классов задан явно параметром allowed_objects. Предупреждения
фреймворка здесь не глушатся, а ловятся и печатаются, чтобы в выводе не было
абсолютных путей с машины автора.

Файл dialogue.json появится рядом со скриптом.

Запуск: python 08_serialization.py
"""

import json
import warnings
from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.load import dumpd, load

from course_model import build_model

HISTORY_FILE = Path(__file__).with_name("dialogue.json")

model = build_model(temperature=0)

# День первый: один ход разговора.
history = [
    SystemMessage("Вы отвечаете одним предложением, без вступлений."),
    HumanMessage("Я собираю ноутбук для монтажа видео, бюджет 120 тысяч. Что важнее всего?"),
]
history.append(model.invoke(history))

print("ДО СОХРАНЕНИЯ:", [type(message).__name__ for message in history])
print("ОТВЕТ МОДЕЛИ: ", history[-1].text.strip())
print()

# История уходит в файл: dumpd отдаёт словарь, json.dumps пишет его на диск.
HISTORY_FILE.write_text(
    json.dumps([dumpd(message) for message in history], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("РАЗМЕР ФАЙЛА:", HISTORY_FILE.stat().st_size, "байт")
print("ВОПРОС ПОЛЬЗОВАТЕЛЯ В ВИДЕ СЛОВАРЯ:")
print(json.dumps(dumpd(history[1]), ensure_ascii=False, indent=2))
print()


def restore(items, **kwargs):
    """Собирает сообщения из словарей и возвращает их вместе с предупреждениями."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        messages = [load(item, **kwargs) for item in items]
    return messages, caught


def print_warnings(title, caught):
    """Печатает предупреждения без абсолютного пути: имя файла и номер строки."""
    print(title)
    if not caught:
        print("  предупреждений нет")
    for warning in caught:
        print(f"  {warning.category.__name__}, {Path(warning.filename).name}:{warning.lineno}")
        print(f"    {warning.message}")


# День второй: другой запуск программы, история берётся из файла.
raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

# Так писать не стоит: список разрешённых классов не задан.
by_default, default_warnings = restore(raw)
print_warnings("ВЫЗОВ БЕЗ allowed_objects", default_warnings)
print()

# Так стоит: разрешены только классы сообщений.
restored, strict_warnings = restore(raw, allowed_objects="messages")
print_warnings('ВЫЗОВ С allowed_objects="messages"', strict_warnings)
print()

print("ПОСЛЕ ВОССТАНОВЛЕНИЯ:", [type(message).__name__ for message in restored])
print(
    "СОДЕРЖИМОЕ СОВПАЛО:",
    [message.content for message in restored] == [message.content for message in history],
)
print(
    "ОБА СПОСОБА ДАЛИ ОДНО И ТО ЖЕ:",
    [message.content for message in by_default] == [message.content for message in restored],
)

restored.append(HumanMessage("Повторите ваш совет одним предложением."))

print("ПРОДОЛЖЕНИЕ РАЗГОВОРА:", model.invoke(restored).text.strip())
