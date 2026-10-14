"""Пример 2 урока 14: инструкция собирается под собеседника и под ход разговора.

Один и тот же агент отвечает на один и тот же вопрос трижды. Меняются два
источника: конфигурация запуска (кто спрашивает) и состояние разговора (сколько
сообщений уже накопилось). Промпт печатается перед каждым ответом, чтобы было
видно, что именно ушло в модель.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from course_model import build_model

QUESTION = "Где сейчас заказ Z-771 и когда он приедет?"

HISTORY = [
    {"role": "user", "content": "Добрый день, нужна помощь по заказам."},
    {"role": "assistant", "content": "Здравствуйте. Назовите номер заказа."},
    {"role": "user", "content": "Сначала общий вопрос: вы работаете в выходные?"},
    {"role": "assistant", "content": "Склад принимает и выдаёт груз по субботам до 15:00."},
    {"role": "user", "content": "Понял, спасибо."},
    {"role": "assistant", "content": "Рад помочь. Что-то ещё?"},
]


@dataclass
class Session:
    """Конфигурация запуска: кто пришёл и из какого города."""

    role: str
    city: str


def find_order(order_id: str) -> str:
    """Находит заказ по номеру вида Z-771 и возвращает город, статус и срок."""
    if order_id != "Z-771":
        return f"Заказ {order_id} не найден."

    return (
        "Заказ Z-771: город Казань, статус укомплектован, ячейка хранения B-14, "
        "перевозчик СДЭК, срок доставки 2 дн."
    )


@dynamic_prompt
def prompt_for_session(request: ModelRequest) -> str:
    """Собирает системный промпт из конфигурации запуска и длины разговора."""
    session = request.runtime.context

    parts = ["Вы помощник службы доставки. Отвечайте по-русски."]

    if session.role == "оператор":
        parts.append(
            "Собеседник, это оператор склада: называйте ячейку хранения и "
            "перевозчика."
        )
    else:
        parts.append(
            "Собеседник, это клиент: внутренние коды ячеек и имя перевозчика "
            "не называйте."
        )

    parts.append(f"Город собеседника: {session.city}.")

    if len(request.messages) > 4:
        parts.append("Разговор длинный, отвечайте одной фразой.")

    prompt = " ".join(parts)
    print("ПРОМПТ:", prompt)

    return prompt


agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[find_order],
    middleware=[prompt_for_session],
    context_schema=Session,
)

RUNS = [
    ("клиент, начало разговора", [], Session(role="клиент", city="Казань")),
    ("оператор, начало разговора", [], Session(role="оператор", city="Казань")),
    ("оператор, разговор идёт давно", HISTORY, Session(role="оператор", city="Казань")),
]

for label, history, session in RUNS:
    print()
    print("ЗАПУСК:", label)
    result = agent.invoke(
        {"messages": [*history, {"role": "user", "content": QUESTION}]},
        context=session,
    )
    print("ОТВЕТ: ", result["messages"][-1].text.replace("\n", " "))
