"""Пример 3 урока 11: сценарий и агент на одной задаче.

Одни и те же данные и две сборки. В сценарии порядок шагов задан кодом: найти
номер, взять заказ, взять срок, попросить модель сформулировать ответ. В агенте
порядок выбирает модель.

Два запроса. Первый целиком подходит под сценарий. Во втором номера заказа нет, и
видно, чем сборки отличаются. Рядом с каждым ответом печатается цена: сколько раз
вызвали модель и сколько токенов на это ушло.
"""

import re

from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler

from course_model import build_model

ORDERS = {
    "4412": {"city": "Казань", "item": "наушники"},
    "5190": {"city": "Омск", "item": "клавиатура"},
}

DELIVERY_DAYS = {"Казань": 2, "Омск": 4}


def lookup_order(number):
    """Данные заказа или None. Общий источник для сценария и для инструмента."""
    return ORDERS.get(number)


def lookup_days(city):
    """Срок доставки в днях или None."""
    return DELIVERY_DAYS.get(city)


def find_order(number: str) -> str:
    """Находит заказ по номеру и возвращает товар и город доставки."""
    order = lookup_order(number)
    if order is None:
        return f"Заказ {number} не найден."
    return f"Заказ {number}: {order['item']}, город доставки {order['city']}."


def delivery_days(city: str) -> str:
    """Возвращает срок доставки в город в днях."""
    days = lookup_days(city)
    if days is None:
        return f"Срок доставки в город {city} неизвестен."
    return f"Доставка в город {city} занимает {days} дня."


SYSTEM_PROMPT = (
    "Вы оператор службы доставки. Отвечайте по-русски, коротко и по делу. "
    "Номера заказов и сроки берите только из фактов, которые вам дали."
)

# Агенту факты во входе не дают, он берёт их из инструментов, как в примере 1.
AGENT_PROMPT = (
    "Вы оператор службы доставки. Отвечайте по-русски, коротко и по делу. "
    "Номера заказов и сроки берите только из инструментов."
)

model = build_model(temperature=0, max_tokens=512)

agent = create_agent(
    model=model,
    tools=[find_order, delivery_days],
    system_prompt=AGENT_PROMPT,
)


def run_workflow(question, callback):
    """Порядок шагов задан здесь, в коде, и не зависит от вопроса."""
    calls = 0

    found = re.search(r"\d{4}", question)
    if found is None:
        return "Не вижу номера заказа, уточните его.", calls

    order = lookup_order(found.group())
    if order is None:
        return f"Заказ {found.group()} не найден.", calls

    days = lookup_days(order["city"])
    facts = (
        f"Заказ {found.group()}: {order['item']}, город {order['city']}, "
        f"срок доставки {days} дня."
    )

    calls += 1
    answer = model.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Факты: {facts}\nВопрос: {question}"},
        ],
        config={"callbacks": [callback]},
    )
    return answer.text, calls


def run_agent(question, callback):
    """Порядок шагов выбирает модель, код задаёт только набор инструментов."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [callback]},
    )
    calls = sum(
        1 for message in result["messages"] if type(message).__name__ == "AIMessage"
    )
    return result["messages"][-1].text, calls


def total_tokens(callback):
    """Сумма токенов по всем моделям, которые вызвали под этим обработчиком."""
    return sum(usage["total_tokens"] for usage in callback.usage_metadata.values())


QUESTIONS = [
    "Когда приедет мой заказ 4412?",
    "Сколько дней идёт доставка в Омск?",
]

for question in QUESTIONS:
    print("=" * 70)
    print("ВОПРОС:", question)

    for title, runner in (("сценарий", run_workflow), ("агент  ", run_agent)):
        callback = UsageMetadataCallbackHandler()
        answer, calls = runner(question, callback)
        print(
            f"  {title} | вызовов модели: {calls} | токенов: {total_tokens(callback)}"
        )
        print(f"           {answer.replace(chr(10), ' ')}")
