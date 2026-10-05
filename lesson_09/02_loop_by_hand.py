"""Пример 2. Цикл вызова инструмента, собранный руками.

Показывает, из чего состоит шаг агента: модель возвращает вызов инструмента,
вызов исполняется вручную, результат кладётся обратно в список сообщений.
"""

from course_model import build_model
from langchain.messages import ToolMessage
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


model = build_model(temperature=0)
model_with_tools = model.bind_tools([get_order_status])

messages = [{"role": "user", "content": "Что с заказом A-1002?"}]

# Этап 1. Модель решает, нужен ли инструмент, и возвращает вызов.
ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)

print("ТЕКСТ ПЕРВОГО ОТВЕТА:", repr(ai_message.text))
print("ЗАПРОШЕНО ВЫЗОВОВ:", len(ai_message.tool_calls))
for call in ai_message.tool_calls:
    print(f"  {call['name']} {call['args']} id={call['id']}")

# Этап 2. Инструменты исполняются вручную, ответы собираются в список.
tool_messages = []
for call in ai_message.tool_calls:
    if call["name"] == get_order_status.name:
        tool_messages.append(get_order_status.invoke(call))
    else:
        # Модель может назвать инструмент, которого нет в списке. Ответ нужен и на такой вызов.
        tool_messages.append(
            ToolMessage(
                content=f"инструмента {call['name']} нет",
                tool_call_id=call["id"],
                status="error",
            )
        )

messages.extend(tool_messages)

print()
print("ОТВЕТОВ ИНСТРУМЕНТА:", len(tool_messages))
for message in tool_messages:
    print(f"  id={message.tool_call_id} -> {message.content}")

ids_requested = {call["id"] for call in ai_message.tool_calls}
ids_answered = {message.tool_call_id for message in tool_messages}
print("ВЫЗОВЫ И ОТВЕТЫ СОВПАДАЮТ:", ids_requested == ids_answered)

# Этап 3. Результаты уходят обратно в модель за финальным ответом.
final = model_with_tools.invoke(messages)
messages.append(final)

print()
print("СООБЩЕНИЙ В ИСТОРИИ:", len(messages))
print("ТИПЫ:", [type(m).__name__ if hasattr(m, "type") else "dict" for m in messages])
print("ФИНАЛЬНЫЙ ОТВЕТ:", final.text)
