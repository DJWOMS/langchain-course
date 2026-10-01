"""Пример 6 урока 6: три режима handle_errors на одной сломанной схеме.

Схема та же, что в примере 5, поэтому первый ответ модели проверку не проходит.
Меняется только то, что агент делает дальше: повторяет с текстом ошибки,
повторяет со своим текстом или отдаёт исключение наружу.

ОТДЕЛЬНО ПРО ПЕРВЫЙ РЕЖИМ. Документация показывает handle_errors=ValueError как
способ повторять только на ошибках разбора. В закреплённой версии пакета ошибка
разбора приходит классом StructuredOutputValidationError, он наследуется от
StructuredOutputError, а тот от Exception, но не от ValueError. Значит условие
не совпадает и повтора не будет. Прогон ниже показывает, что получается.
"""

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    StructuredOutputValidationError,
    ToolStrategy,
)
from pydantic import BaseModel, Field, field_validator

from course_model import build_model

TEXT = "Оплатил заказ 4412 на 5300 рублей, товар не пришёл, верните деньги."


class Refund(BaseModel):
    """Заявка на возврат денег."""

    order_id: str = Field(description="Номер заказа из обращения")
    amount: float = Field(description="Сумма к возврату в рублях")

    @field_validator("order_id")
    @classmethod
    def check_order_id(cls, value: str) -> str:
        """Внутренний формат номера, которого нет в JSON Schema."""
        if not value.startswith("ORD-"):
            message = "Номер заказа записывается с префиксом ORD-, например ORD-1234"
            raise ValueError(message)
        return value


model = build_model(temperature=0, max_tokens=1024)


def run(title: str, handle_errors) -> None:
    """Прогоняет одно обращение с заданным режимом обработки ошибок."""
    print(title)
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(schema=Refund, handle_errors=handle_errors),
    )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
    except Exception as error:  # noqa: BLE001
        print("  исключение:", type(error).__name__)
        print("  текст:", str(error)[:160])
        print()
        return

    calls = sum(1 for m in result["messages"] if type(m).__name__ == "AIMessage")
    print("  вызовов модели:", calls)
    for message in result["messages"]:
        if type(message).__name__ == "ToolMessage":
            print("  сообщение инструмента:", repr(message.text[:120]))
    print("  разобранный ответ:", result.get("structured_response"))
    print()


run("РЕЖИМ 1: handle_errors=ValueError, как показано в документации", ValueError)
run(
    "РЕЖИМ 2: handle_errors=StructuredOutputValidationError, настоящий класс ошибки",
    StructuredOutputValidationError,
)
run(
    "РЕЖИМ 3: handle_errors со своим текстом",
    "Номер заказа записывается с префиксом ORD-. Повторите ответ.",
)
