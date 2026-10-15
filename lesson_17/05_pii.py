"""Пример 5 урока 17: персональные данные до модели и после инструмента.

Первый показывает три стратегии сразу: почта затирается, номер карты
маскируется, адрес заменяется хешем. Печатается сообщение человека
из состояния агента, то есть то, что ушло в модель.

Второй показывает стратегию block со своим detector: ключ доступа
в тексте поднимает исключение PIIDetectionError, и запуск падает до вызова модели.

Третий чистит ответ инструмента: выгрузка из базы приходит
с почтой клиента, и почта затирается до следующего вызова модели.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import PIIDetectionError, PIIMiddleware

from course_model import build_model

SYSTEM_PROMPT = "Вы помощник службы поддержки. Отвечайте по-русски, одной фразой."


def human_text(messages):
    """Возвращает текст последнего сообщения человека из состояния агента."""
    return [m for m in messages if m.type == "human"][-1].text


print("ЗАПУСК 1: три стратегии на входе")

agent = create_agent(
    model=build_model(temperature=0),
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        PIIMiddleware("credit_card", strategy="mask"),
        PIIMiddleware("ip", strategy="hash"),
    ],
)

question = (
    "Почта клиента ivanov@example.com, карта 5105-1051-0510-5100, "
    "заходил с адреса 192.168.1.42. Подтвердите, что заявка принята."
)
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

print("  было: ", question)
print("  ушло: ", human_text(result["messages"]))
print()

print("ЗАПУСК 2: стратегия block со своим detector")

blocking_agent = create_agent(
    model=build_model(temperature=0),
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
    ],
)

try:
    blocking_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Вот мой ключ sk-abcdefghijklmnopqrstuvwxyz012345, проверьте доступ.",
                }
            ]
        }
    )
except PIIDetectionError as error:
    print("  исключение:", type(error).__name__)
    print("  текст:     ", error)
    print("  тип данных:", error.pii_type)
    print("  найдено:   ", [match["value"] for match in error.matches])

print()

print("ЗАПУСК 3: чистится ответ инструмента")


def customer_card(order_id: str) -> str:
    """Возвращает карточку клиента по номеру заказа."""
    return f"заказ {order_id}: Игнатов Пётр, почта ignatov@example.com, статус доставлен"


tool_agent = create_agent(
    model=build_model(temperature=0),
    tools=[customer_card],
    system_prompt=SYSTEM_PROMPT + " Данные о клиенте берите инструментом.",
    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=False,
            apply_to_tool_results=True,
        ),
    ],
)

result = tool_agent.invoke(
    {"messages": [{"role": "user", "content": "Что известно по заказу A-118?"}]}
)

tool_messages = [m for m in result["messages"] if m.type == "tool"]
print("  инструмент вернул:", customer_card("A-118"))
print("  в истории лежит:  ", tool_messages[-1].content)
print("  ответ агента:     ", result["messages"][-1].text.replace("\n", " "))
