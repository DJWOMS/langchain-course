"""Пример 1. Что декоратор @tool делает с обычной функцией.

Модель здесь не вызывается. Пример печатает объект инструмента и вызывает его двумя способами.
"""

import json

from langchain.tools import tool


@tool
def get_order_status(order_id: str) -> str:
    """Узнать статус заказа в магазине по его номеру."""
    catalog = {
        "A-1001": "собран, ждёт курьера",
        "A-1002": "в пути, доставка завтра",
        "A-1003": "отменён покупателем",
    }
    return catalog.get(order_id, "заказ с таким номером не найден")


print("ТИП ОБЪЕКТА:", type(get_order_status).__name__)
print("ИМЯ:", get_order_status.name)
print("ОПИСАНИЕ:", get_order_status.description)
print("АРГУМЕНТЫ:", get_order_status.args)

print()
print("СХЕМА, КОТОРУЮ УВИДИТ МОДЕЛЬ:")
schema = get_order_status.tool_call_schema.model_json_schema()
print(json.dumps(schema, ensure_ascii=False, indent=2))

print()
print("ВЫЗОВ СЛОВАРЁМ АРГУМЕНТОВ:")
print(repr(get_order_status.invoke({"order_id": "A-1002"})))

print()
print("ВЫЗОВ ТАК, КАК ЭТО ДЕЛАЕТ АГЕНТ:")
fake_call = {
    "name": "get_order_status",
    "args": {"order_id": "A-1003"},
    "id": "call_demo_1",
    "type": "tool_call",
}
message = get_order_status.invoke(fake_call)
print("тип результата:", type(message).__name__)
print("tool_call_id:", message.tool_call_id)
print("status:", message.status)
print("содержимое:", message.content)
