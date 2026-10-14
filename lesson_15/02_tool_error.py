"""Пример 2 урока 15: падение инструмента с одним middleware и без него.

Инструмент читает учётную книгу и падает на годах, которых нет в архиве.
Первый запуск идёт без middleware, второй с ToolErrorMiddleware.

Middleware не чинит инструмент. Он превращает исключение в сообщение об ошибке,
которое получает модель, и решение остаётся за ней.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware

from course_model import build_model


def read_ledger(year: int) -> str:
    """Читает учётную книгу склада за указанный год."""
    if year < 2019:
        raise ValueError(f"архив за {year} год не оцифрован")
    return f"Книга за {year} год: 412 приходных записей, 87 списаний."


def on_error(exc, request):
    """Что получит модель вместо исключения. None означает "пропустить дальше"."""
    if isinstance(exc, ValueError):
        name = request.tool_call["name"]
        return (
            f"Инструмент {name} завершился с ошибкой {type(exc).__name__}: "
            f"такого года в архиве нет. Самый ранний оцифрованный год, это 2019."
        )
    # Остальные исключения идут дальше и роняют агента.
    return None


SYSTEM_PROMPT = (
    "Вы архивариус склада. Отвечайте по-русски, двумя-тремя фразами. "
    "Если инструмент сообщил об ошибке, поступайте по тому, что в ней написано."
)

QUESTION = "Сколько списаний было в 2017 году?"

model = build_model(temperature=0, max_tokens=384)

plain = create_agent(model=model, tools=[read_ledger], system_prompt=SYSTEM_PROMPT)

guarded = create_agent(
    model=model,
    tools=[read_ledger],
    system_prompt=SYSTEM_PROMPT,
    middleware=[ToolErrorMiddleware(on_error)],
)

print("ЗАПУСК БЕЗ MIDDLEWARE")
try:
    result = plain.invoke({"messages": [{"role": "user", "content": QUESTION}]})
except Exception as exc:  # noqa: BLE001 - пример печатает тип падения
    print(f"  запуск упал: {type(exc).__name__}: {exc}")
else:
    print("  запуск дошёл до конца:", result["messages"][-1].text)

print()
print("ЗАПУСК С MIDDLEWARE")
result = guarded.invoke({"messages": [{"role": "user", "content": QUESTION}]})

for message in result["messages"]:
    if message.type == "tool":
        print(f"  {message.name}: status={message.status}")

print("  ответ:", result["messages"][-1].text.replace("\n", " "))
