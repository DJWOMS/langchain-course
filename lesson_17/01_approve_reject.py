"""Пример 1 урока 17: пауза перед действием с последствиями.

У агента два инструмента: поиск карточки заказа и возврат денег. Поиск
выполняется без паузы, перед возвратом запуск останавливается до решения
человека.

Диалог идёт дважды, в двух разных threads: в первом человек подтверждает
возврат, во втором отказывает и называет причину. Список EXECUTED собирает
имена вызовов, которые дошли до тела функции.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from course_model import build_model

EXECUTED = []

ORDERS = {
    "A-118": {"customer": "Игнатов", "sum": 4500, "status": "доставлен"},
}


def search_orders(order_id: str) -> str:
    """Возвращает карточку заказа по его номеру."""
    EXECUTED.append(f"search_orders({order_id})")
    order = ORDERS.get(order_id)

    if order is None:
        return f"заказ {order_id} не найден"

    return f"{order_id}: {order['customer']}, {order['sum']} руб., {order['status']}"


def refund_order(order_id: str, amount: int) -> str:
    """Возвращает покупателю деньги за заказ."""
    EXECUTED.append(f"refund_order({order_id}, {amount})")
    return f"возврат {amount} руб. по заказу {order_id} проведён"


SYSTEM_PROMPT = (
    "Вы оператор поддержки магазина. Сначала посмотрите карточку заказа, "
    "затем верните деньги на всю его сумму. Отвечайте по-русски, одной фразой."
)

agent = create_agent(
    model=build_model(temperature=0),
    tools=[search_orders, refund_order],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "refund_order": {"allowed_decisions": ["approve", "reject"]},
                "search_orders": False,
            },
            description_prefix="Операция ждёт подтверждения",
        ),
    ],
    checkpointer=InMemorySaver(),
)


def run(thread_id, decision):
    """Ведёт один диалог до паузы и возобновляет его решением человека."""
    EXECUTED.clear()
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Заказ A-118, покупатель просит возврат."}]},
        config,
        version="v2",
    )

    print(f"THREAD {thread_id}")
    print("  выполнено до паузы:", EXECUTED)
    print("  запуск остановлен: ", bool(result.interrupts))

    request = result.interrupts[0].value

    for action, review in zip(request["action_requests"], request["review_configs"]):
        print("  агент просит:      ", action["name"], action["args"])
        print("  разрешены решения: ", review["allowed_decisions"])
        print("  описание:          ", action["description"].replace("\n", " | "))

    print("  решение человека:  ", decision["type"])

    result = agent.invoke(Command(resume={"decisions": [decision]}), config, version="v2")

    answer = " ".join(result.value["messages"][-1].text.split())

    print("  выполнено всего:   ", EXECUTED)
    print("  ответ агента:      ", answer or "(последнее сообщение пришло без текста)")
    print()


run("refund-yes", {"type": "approve"})
run(
    "refund-no",
    {
        "type": "reject",
        "message": "Возврат не согласован: заказ доставлен, претензий по качеству нет.",
    },
)
