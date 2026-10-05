"""Пример 6. Что происходит, когда инструмент падает.

Две сборки одного агента с одним и тем же ломающимся инструментом.
Первая, без обработки: исключение выходит наружу из agent.invoke.
Вторая, с middleware wrap_tool_call: исключение превращается
в ToolMessage, и агент доходит до ответа.
"""

from collections.abc import Callable

from course_model import build_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest


@tool
def get_balance(account: str) -> str:
    """Узнать остаток на счёте по его номеру."""
    if account != "40817":
        # Внешняя система отвечает отказом. В рабочем приложении это был бы
        # таймаут базы, 500 от API или отсутствующая запись.
        raise ValueError(f"счёт {account} не найден в реестре")
    return "остаток 12 400 рублей"


QUESTION = "Сколько денег на счёте 99999?"
SYSTEM = "Вы банковский помощник. Остаток узнавайте только инструментом."


def build_agent(middleware):
    return create_agent(
        model=build_model(temperature=0),
        tools=[get_balance],
        system_prompt=SYSTEM,
        middleware=middleware,
    )


print("СБОРКА 1: без обработки ошибок")
try:
    result = build_agent([]).invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print("агент дошёл до ответа:", result["messages"][-1].text)
except Exception as error:
    print("agent.invoke упал:", type(error).__name__)
    print("текст:", error)

print()


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Превращает исключение инструмента в сообщение, понятное модели."""
    try:
        return handler(request)
    except Exception as error:
        return ToolMessage(
            content=f"Инструмент не отработал: {error}. Не повторяйте вызов, "
            "скажите пользователю, что счёт не найден.",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


print("СБОРКА 2: то же самое с middleware wrap_tool_call")
result = build_agent([handle_tool_errors]).invoke(
    {"messages": [{"role": "user", "content": QUESTION}]}
)

for message in result["messages"]:
    kind = type(message).__name__
    if kind == "ToolMessage":
        print(f"ToolMessage status={message.status}: {message.content}")
    elif kind == "AIMessage" and message.tool_calls:
        print("AIMessage просит:", [c["args"] for c in message.tool_calls])

print("ОТВЕТ ПОЛЬЗОВАТЕЛЮ:", result["messages"][-1].text)
