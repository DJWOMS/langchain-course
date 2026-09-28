"""Пример 7 урока 5: примеры в промпте, три способа их положить.

Задача одна: свести обращение к строке "категория|срочность|очередь".
Сравниваются три варианта: одна инструкция, инструкция с примерами текстом и
инструкция с примерами, положенными парами сообщений перед вопросом.

Что здесь доказывает код, а что остаётся на усмотрение модели:

1) доказуемо, где лежат коды очередей. Проверки печатаются строками "код Q-14
   в ...": в инструкции его нет, в двух других вариантах есть
2) доказуемо, что уходит в модель на вызове третьего варианта и что остаётся
   в состоянии после него. Два списка печатаются рядом
3) ответ модели, это иллюстрация. Вердикт check_queue и строка ИТОГ дают
   замер одного прогона, а не свойство модели
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AIMessage, HumanMessage

from course_model import build_model

INSTRUCTION = (
    "Вы классифицируете обращения в поддержку. Ответ, это одна строка вида "
    "категория|срочность|очередь, без пояснений. "
    "Категория: оплата, доставка, возврат. Срочность: низкая, высокая. "
    "Очередь, это внутренний код команды, которая берёт обращение в работу. "
    "Код очереди зависит только от категории."
)

# Таблица очередей задана в одном месте: из неё собираются примеры и по ней же
# проверяется ответ. В инструкцию она не попадает намеренно.
QUEUES = {
    "оплата": "Q-14",
    "доставка": "Q-3",
    "возврат": "Q-22",
}

EXAMPLES = [
    ("не приходит чек на почту", f"оплата|низкая|{QUEUES['оплата']}"),
    ("курьер не пришёл во второй раз", f"доставка|высокая|{QUEUES['доставка']}"),
    ("пришла куртка не того цвета, заберите", f"возврат|низкая|{QUEUES['возврат']}"),
]

TICKETS = [
    "оплата не проходит, карта рабочая",
    "когда приедет заказ 4412?",
    "хочу вернуть куртку, размер не подошёл",
]

WITH_EXAMPLES = INSTRUCTION + "\n\nПримеры:\n" + "\n".join(
    f"{question} -> {answer}" for question, answer in EXAMPLES
)

# Сюда middleware кладёт список сообщений последнего вызова, чтобы его можно было
# сравнить с тем, что осталось в состоянии после прогона.
SENT_TO_MODEL: list[str] = []


def check_queue(answer: str) -> str:
    """Говорит, взят код очереди из таблицы или придуман моделью.

    Категория берётся из самого ответа, поэтому проверка не зависит от того,
    угадала модель категорию или нет: сверяется только пара категория, код.
    """
    parts = [part.strip() for part in answer.split("|")]

    if len(parts) != 3:
        return "форма ответа другая"

    category, _, queue = parts
    expected = QUEUES.get(category)

    if expected is None:
        return "категория не из списка"

    return "код из таблицы" if queue == expected else "кода нет в таблице"


@wrap_model_call
def inject_examples(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Кладёт примеры парами сообщений перед разговором, на один вызов."""
    shots = []
    for question, answer in EXAMPLES:
        shots.append(HumanMessage(question))
        shots.append(AIMessage(answer))

    messages = [*shots, *request.messages]
    SENT_TO_MODEL[:] = [
        f"{type(message).__name__:<14} {message.text!r}" for message in messages
    ]

    return handler(request.override(messages=messages))


model = build_model(temperature=0, max_tokens=1024)

VARIANTS = [
    ("одна инструкция", create_agent(model=model, tools=[], system_prompt=INSTRUCTION)),
    (
        "примеры текстом",
        create_agent(model=model, tools=[], system_prompt=WITH_EXAMPLES),
    ),
    (
        "примеры сообщениями",
        create_agent(
            model=model,
            tools=[],
            system_prompt=INSTRUCTION,
            middleware=[inject_examples],
        ),
    ),
]

CODE = QUEUES["оплата"]

print("ГДЕ ЛЕЖАТ КОДЫ ОЧЕРЕДЕЙ")
for category, queue in QUEUES.items():
    print(f"  {category:<10} {queue}")
print(f"  код {CODE} в инструкции:            {CODE in INSTRUCTION}")
print(f"  код {CODE} в инструкции с примерами: {CODE in WITH_EXAMPLES}")
print(f"  код {CODE} в парах сообщений:        {any(CODE in a for _, a in EXAMPLES)}")
print()

for title, agent in VARIANTS:
    print(title.upper())
    hits = 0
    for ticket in TICKETS:
        result = agent.invoke({"messages": [HumanMessage(ticket)]})
        answer = result["messages"][-1].text.replace("\n", " ")
        verdict = check_queue(answer)

        if verdict == "код из таблицы":
            hits += 1

        print(f"  {ticket:<38} -> {answer!r:<28} {verdict}")
    print(f"  ИТОГ: код из таблицы {hits} из {len(TICKETS)}")
    print()

print("ЧТО УШЛО В МОДЕЛЬ НА ПОСЛЕДНЕМ ВЫЗОВЕ ТРЕТЬЕГО ВАРИАНТА")
for line in SENT_TO_MODEL:
    print(f"  {line}")
print(f"  всего сообщений: {len(SENT_TO_MODEL)}")
print()

print("ЧТО ОСТАЛОСЬ В СОСТОЯНИИ ПОСЛЕ ЭТОГО ЖЕ ВЫЗОВА")
for message in result["messages"]:
    print(f"  {type(message).__name__:<14} {message.text!r}")
print(f"  всего сообщений: {len(result['messages'])}")
