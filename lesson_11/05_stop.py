"""Пример 5 урока 11: три способа остановить агента.

Один и тот же вопрос задаётся трижды. Первый запуск заканчивается сам: модель
перестаёт просить инструменты. Второму поставлен потолок шагов графа, и он падает
с GraphRecursionError. Третьему поставлен бюджет вызовов модели, и он заканчивается
без исключения, но последнее сообщение в истории добавлено middleware.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.errors import GraphRecursionError

from course_model import build_model

ORDERS = {"4412": {"city": "Казань", "item": "наушники"}}
DELIVERY_DAYS = {"Казань": 2}

QUESTION = {"messages": [{"role": "user", "content": "Когда приедет мой заказ 4412?"}]}


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


SYSTEM_PROMPT = (
    "Вы оператор службы доставки. Отвечайте по-русски, коротко и по делу. "
    "Номера заказов и сроки берите только из инструментов."
)

model_kwargs = {"temperature": 0, "max_tokens": 512}
tools = [find_order, delivery_days]


def describe(result):
    """Сколько сообщений в истории и чем она закончилась."""
    last = result["messages"][-1]
    text = last.text.replace("\n", " ")
    if len(text) > 70:
        text = text[:67] + "..."
    return f"сообщений: {len(result['messages'])}, последнее: {type(last).__name__} {text!r}"


print("1. БЕЗ ОГРАНИЧЕНИЙ")
plain = create_agent(
    model=build_model(**model_kwargs), tools=tools, system_prompt=SYSTEM_PROMPT
)
print("  ", describe(plain.invoke(QUESTION)))

print()
print("2. ПОТОЛОК ШАГОВ ГРАФА: recursion_limit=2")
try:
    plain.invoke(QUESTION, config={"recursion_limit": 2})
except GraphRecursionError as error:
    print("   отказ:", type(error).__name__)
    print("   текст:", str(error).split("\n")[0])

print()
print("3. БЮДЖЕТ ВЫЗОВОВ МОДЕЛИ: run_limit=1, exit_behavior='end'")
limited = create_agent(
    model=build_model(**model_kwargs),
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    middleware=[ModelCallLimitMiddleware(run_limit=1, exit_behavior="end")],
)
print("  ", describe(limited.invoke(QUESTION)))
