"""Пример 1 урока 11: первый агент на create_agent.

Два инструмента, между которыми есть зависимость: город доставки известен только
после поиска заказа. Сколько раз вызвать инструменты и в каком порядке, определяют
ответы модели.

Печатается не только ответ, но и то, что вернул агент целиком: ключи состояния и
вся цепочка сообщений.
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

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Когда приедет мой заказ 4412?"}]}
)

print("ТИП РЕЗУЛЬТАТА:", type(result).__name__)
print("КЛЮЧИ СОСТОЯНИЯ:", sorted(result))
print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
print()

print("ЦЕПОЧКА СООБЩЕНИЙ")
model_calls = 0
for number, message in enumerate(result["messages"], start=1):
    kind = type(message).__name__
    if kind == "AIMessage":
        model_calls += 1
    calls = getattr(message, "tool_calls", None)
    if calls:
        names = [f"{call['name']}({call['args']})" for call in calls]
        print(f"  {number}. {kind:<12} вызывает {names}")
        continue
    text = message.text.replace("\n", " ")
    if len(text) > 60:
        text = text[:57] + "..."
    print(f"  {number}. {kind:<12} {text!r}")

print()
print("ВЫЗОВОВ МОДЕЛИ:", model_calls)
