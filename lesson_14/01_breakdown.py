"""Пример 1 урока 14: из чего собран запрос к модели и сколько весит каждая часть.

Middleware печатает строку на каждый вызов модели: сколько сообщений уходит,
во что оценены системное сообщение и сообщения каждого типа, сколько отправлено
инструментов и какой вход насчитал провайдер.

Задача выбрана так, чтобы агент вызвал инструменты несколько раз, и в таблице
стало видно, какая именно часть запроса растёт от вызова к вызову.
"""

from collections import Counter
from itertools import count
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages.utils import count_tokens_approximately

from course_model import build_model

ORDERS = {
    "Z-771": {"city": "Казань", "status": "укомплектован"},
    "Z-772": {"city": "Омск", "status": "ждёт оплаты"},
}

DELIVERY = {"Казань": 2, "Омск": 5}

RULES = (
    "Регламент приёмки груза на складе города {city}.\n"
    "1. Машина встаёт под разгрузку по электронной очереди, талон берут на "
    "въезде, время в талоне сдвигать нельзя.\n"
    "2. Паллеты принимают по счёту мест и по весу брутто, расхождение больше "
    "двух процентов фиксируют актом в тот же день.\n"
    "3. Упаковку вскрывают только при видимом повреждении обрешётки, вскрытие "
    "снимают на камеру и прикладывают файл к акту.\n"
    "4. Температурный груз замеряют дважды: в момент открытия дверей и через "
    "пятнадцать минут после начала разгрузки.\n"
    "5. Брак раскладывают по трём коробам: бой, подмочка, пересорт. Короб "
    "опечатывают и передают на карантинную зону до конца смены.\n"
    "6. Документы возвращают водителю с отметкой о приёмке, копию подшивают в "
    "журнал склада и дублируют в учётную систему до 20:00 того же дня.\n"
    "7. Расхождение по количеству больше пяти мест останавливает приёмку: "
    "вызывают старшего смены и представителя перевозчика.\n"
)


def find_order(order_id: str) -> str:
    """Находит заказ по номеру вида Z-771 и возвращает город и статус."""
    order = ORDERS.get(order_id)

    if order is None:
        return f"Заказ {order_id} не найден."

    return f"Заказ {order_id}: город {order['city']}, статус {order['status']}."


def delivery_days(city: str) -> str:
    """Сообщает срок доставки в днях до указанного города."""
    days = DELIVERY.get(city)

    if days is None:
        return f"Город {city} не обслуживается."

    return f"Доставка до города {city} занимает {days} дн."


def acceptance_rules(city: str) -> str:
    """Возвращает полный регламент приёмки груза для склада города."""
    return RULES.format(city=city)


step = count(1)
sent_tools = []


@wrap_model_call
def breakdown(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Считает состав запроса до вызова модели, печатает его с расходом после ответа."""
    sent_tools[:] = [tool.name for tool in request.tools]
    prompt = (
        count_tokens_approximately([request.system_message])
        if request.system_message
        else 0
    )

    by_type = Counter()
    for message in request.messages:
        by_type[message.type] += count_tokens_approximately([message])

    estimate = prompt + sum(by_type.values())
    response = handler(request)
    # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
    usage = response.result[0].usage_metadata or {}

    print(
        f"{next(step):>5}  {len(request.messages):>9}  {prompt:>6}  "
        f"{by_type['human']:>6}  {by_type['ai']:>4}  {by_type['tool']:>6}  "
        f"{len(request.tools):>6}  {estimate:>6}  {usage.get('input_tokens', 0):>5}"
    )

    return response


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[find_order, delivery_days, acceptance_rules],
    system_prompt=(
        "Вы помощник логистического отдела. Отвечайте по-русски. Номер заказа "
        "всегда проверяйте инструментом, по памяти не отвечайте. Если в ответе "
        "нужен срок доставки, берите его инструментом для города из заказа. "
        "Правила приёмки пересказывайте своими словами, целиком регламент не "
        "цитируйте."
    ),
    middleware=[breakdown],
)

print(
    f"{'вызов':>5}  {'сообщений':>9}  {'промпт':>6}  {'human':>6}  {'ai':>4}  "
    f"{'tool':>6}  {'инстр':>6}  {'оценка':>6}  {'вход':>5}"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Заказ Z-771: когда он приедет и что учесть при приёмке?"
                ),
            }
        ]
    }
)

print()
print("ИНСТРУМЕНТЫ В ЗАПРОСЕ:", ", ".join(sent_tools))
print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
