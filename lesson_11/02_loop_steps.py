"""Пример 2 урока 11: цикл модель-инструменты по шагам.

Тот же агент, что в примере 1, но запущенный потоком в режиме updates: после
каждого шага графа приходит правка состояния вместе с именем узла, который её
сделал. Видно, из каких узлов собран запуск и на чём он заканчивается.
"""

from langchain.agents import create_agent

from course_model import build_model

ORDERS = {
    "4412": {"city": "Казань", "item": "наушники"},
    "5190": {"city": "Омск", "item": "клавиатура"},
}

DELIVERY_DAYS = {"Казань": 2, "Омск": 4}


def find_order(number: str) -> str:
    """Находит заказ по номеру и возвращает товар и город доставки."""
    order = ORDERS.get(number)
    if order is None:
        return f"Заказ {number} не найден."
    return f"Заказ {number}: {order['item']}, город доставки {order['city']}."


def delivery_days(city: str) -> str:
    """Возвращает срок доставки в город в днях."""
    days = DELIVERY_DAYS.get(city)
    if days is None:
        return f"Срок доставки в город {city} неизвестен."
    return f"Доставка в город {city} занимает {days} дня."


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[find_order, delivery_days],
    system_prompt=(
        "Вы оператор службы доставки. Отвечайте по-русски, коротко и по делу. "
        "Номера заказов и сроки берите только из инструментов."
    ),
)

step = 0
last_message = None

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Когда приедет мой заказ 4412?"}]},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] != "updates":
        continue

    for node, update in chunk["data"].items():
        step += 1
        # У служебных ключей вроде __interrupt__ правка приходит не словарём.
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if not messages:
            print(f"шаг {step}: узел {node}, правка без сообщений: {update!r}")
            continue
        for message in messages:
            last_message = message
            calls = getattr(message, "tool_calls", None)
            if calls:
                names = [call["name"] for call in calls]
                print(f"шаг {step}: узел {node}, запрошены инструменты {names}")
            else:
                text = message.text.replace("\n", " ")
                if len(text) > 50:
                    text = text[:47] + "..."
                print(f"шаг {step}: узел {node}, {type(message).__name__} {text!r}")

print()
print("ШАГОВ ВСЕГО:", step)
print(
    "ПОСЛЕДНЕЕ СООБЩЕНИЕ ЗАПРОСИЛО ИНСТРУМЕНТЫ:",
    bool(getattr(last_message, "tool_calls", None)),
)
