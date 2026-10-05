"""Пример 8. return_direct: выход из цикла сразу после инструмента.

Два прогона. В первом единственный инструмент помечен return_direct=True,
и агент отдаёт его вывод как есть. Во втором в системном промпте указано
вызвать оба инструмента на одном шаге. Если модель так и сделает, выход
сразу не срабатывает.
"""

from course_model import build_model
from langchain.agents import create_agent
from langchain.tools import tool

PAYMENT_LINK = "https://pay.example.com/invoice/A-1002?sum=1200"


@tool(return_direct=True)
def create_payment_link(order_id: str) -> str:
    """Выдать ссылку на оплату заказа. Ответ показывается пользователю дословно."""
    return f"Ссылка на оплату заказа {order_id}: {PAYMENT_LINK}"


@tool
def get_order_sum(order_id: str) -> str:
    """Узнать сумму заказа в рублях."""
    return "1200"


def report(title, agent, question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    print(title)
    print("  типы сообщений:", [type(m).__name__ for m in messages])
    print("  вызовов модели:", sum(1 for m in messages if type(m).__name__ == "AIMessage"))
    last = messages[-1]
    print("  последнее сообщение:", type(last).__name__)
    print("  его содержимое:", last.content)
    print()


report(
    "ПРОГОН 1: один инструмент с return_direct=True",
    create_agent(
        model=build_model(temperature=0),
        tools=[create_payment_link],
        system_prompt="Вы помощник магазина. Ссылку на оплату выдавайте инструментом.",
    ),
    "Дайте ссылку на оплату заказа A-1002.",
)

report(
    "ПРОГОН 2: рядом обычный инструмент, вызваны оба",
    create_agent(
        model=build_model(temperature=0),
        tools=[create_payment_link, get_order_sum],
        system_prompt=(
            "Вы помощник магазина. Когда просят ссылку на оплату, сначала одним ответом "
            "вызовите оба инструмента: сумму заказа и ссылку на оплату."
        ),
    ),
    "Дайте ссылку на оплату заказа A-1002 и скажите его сумму.",
)
