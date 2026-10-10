"""Пример 5 урока 13: агент пишет в память по правилу.

Системный промпт перечисляет, что считается устойчивым фактом. Схема
инструмента сужает поле до перечисления, и вызов с чужим полем отсекается
проверкой аргументов до запуска функции. Проверка ALLOWED_FIELDS внутри
инструмента повторяет схему и страхует от её правки.

Ключ записи, это имя поля, поэтому повторный факт перезаписывает прежний и
хранилище не растёт от диалога к диалогу.
"""

from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from course_model import build_model

ALLOWED_FIELDS = ("city", "role", "shift", "answer_style")


@dataclass
class Session:
    """Конфигурация запуска: чьи записи править."""

    user_id: str


@tool
def remember(
    field: Literal["city", "role", "shift", "answer_style"],
    value: str,
    runtime: ToolRuntime[Session],
) -> str:
    """Запомнить устойчивый факт о пользователе.

    Поля: city это город работы; role это чем человек занимается; shift это
    рабочие часы; answer_style это как человеку отвечать. Разовые состояния
    (самочувствие, погода, планы на завтра) этим инструментом не сохраняются.
    """
    if runtime.store is None:
        return "Хранилище не подключено, запомнить нечем."

    if field not in ALLOWED_FIELDS:
        return f"Поле {field} не сохраняется, допустимы: {', '.join(ALLOWED_FIELDS)}."

    namespace = (runtime.context.user_id, "profile")
    runtime.store.put(namespace, field, {"value": value})
    return f"Запомнил {field}."


store = InMemoryStore()

agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[remember],
    system_prompt=(
        "Вы рабочий помощник. В память записывайте только то, что останется "
        "верным через месяц: город работы, занятие, рабочие часы, пожелание "
        "о форме ответа. Самочувствие, погоду и планы на ближайшие дни не "
        "записывайте. Отвечайте по-русски и коротко."
    ),
    context_schema=Session,
    store=store,
)

MESSAGE = (
    "Я переехал в Омск и теперь веду склад запчастей. Смена с 8 до 17. "
    "Сегодня болит голова, а завтра в десять встреча с поставщиком."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": MESSAGE}]},
    context=Session(user_id="u-101"),
)

print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
print()

print("ВЫЗОВЫ ИНСТРУМЕНТА:")
for message in result["messages"]:
    for call in getattr(message, "tool_calls", []) or []:
        print(f"  {call['name']}({call['args']})")
print()

print("ЧТО ЛЕГЛО В ПАМЯТЬ:")
for item in store.search(("u-101", "profile")):
    print(f"  {item.key}: {item.value['value']}")
