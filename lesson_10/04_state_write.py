"""Пример 4 урока 10: инструмент пишет в состояние.

Обычный инструмент возвращает строку, и она становится сообщением для модели.
Этот инструмент возвращает Command с полем update и записывает в состояние,
какой команде передано обращение. Поле escalated_to добавлено в своей схеме
состояния, поэтому после прогона его видно рядом с messages.
"""

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from course_model import build_model


class SupportState(AgentState):
    """Состояние агента поддержки: к messages добавлено поле передачи."""

    escalated_to: str


@tool
def escalate(team: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """Передать обращение другой команде. Название команды на латинице."""
    return Command(
        update={
            "escalated_to": team,
            "messages": [
                ToolMessage(
                    content=f"Обращение передано команде {team}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[escalate],
    system_prompt=(
        "Вы первая линия поддержки. Вопросы про деньги и списания передавайте "
        "команде billing, вопросы про доставку команде logistics. Отвечайте "
        "по-русски и коротко."
    ),
    state_schema=SupportState,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "С меня дважды списали деньги за заказ 4412."}
        ]
    }
)

print("ОТВЕТ:", result["messages"][-1].text.replace("\n", " "))
print("КЛЮЧИ СОСТОЯНИЯ:", sorted(result))
print("ПОЛЕ escalated_to:", result.get("escalated_to", "<поля нет>"))
print()
print("СООБЩЕНИЯ В СОСТОЯНИИ")
for number, message in enumerate(result["messages"], start=1):
    text = message.text.replace("\n", " ")
    if len(text) > 50:
        text = text[:47] + "..."
    print(f"  {number}. {type(message).__name__:<14} {text!r}")
