"""Пример 8 урока 10: старый способ, config configurable, рядом с новым.

Один и тот же агент запускается дважды. В первом запуске идентификатор
пользователя приходит аргументом context, во втором тем способом, которым
это делали до версии 1: словарём configurable внутри config. Инструмент
возвращает оба источника одной строкой, поэтому видно, что в каждом запуске
пришло, а что нет.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from course_model import build_model


@dataclass
class SessionContext:
    """Конфигурация запуска."""

    user_id: str


@tool
def whoami(runtime: ToolRuntime[SessionContext]) -> str:
    """Вернуть идентификатор текущего пользователя из обоих источников."""
    # Защитная ветка: без аргумента context поле runtime.context равно None.
    from_context = getattr(runtime.context, "user_id", None)
    from_configurable = runtime.config.get("configurable", {}).get("user_id")

    return f"context={from_context} configurable={from_configurable}"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[whoami],
    system_prompt=(
        "Вы помощник. Идентификатор пользователя берите инструментом и "
        "возвращайте ровно ту строку, которую он вернул, без правок."
    ),
    context_schema=SessionContext,
)

QUESTION = "Под каким идентификатором я вошёл?"


def run(title, **invoke_kwargs):
    """Один запуск агента со своим способом передачи данных."""
    print(title)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]}, **invoke_kwargs
    )

    for message in result["messages"]:
        if type(message).__name__ == "ToolMessage":
            print("  инструмент вернул:", message.text.replace("\n", " "))

    print("  ответ агента:", result["messages"][-1].text.replace("\n", " "))


run("ЗАПУСК 1, новый способ: аргумент context", context=SessionContext(user_id="u-101"))
print()
run(
    "ЗАПУСК 2, старый способ: config configurable",
    config={"configurable": {"user_id": "u-303"}},
)
