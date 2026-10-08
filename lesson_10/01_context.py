"""Пример 1 урока 10: конфигурация запуска через context и context_schema.

Один и тот же вопрос от двух разных пользователей. Идентификатора пользователя
нет в схеме для модели, поэтому модель не может его подставить. Он приходит в
агента отдельным аргументом context, а инструмент читает его через runtime.context.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

from course_model import build_model


@dataclass
class SessionContext:
    """Конфигурация запуска: кто спрашивает и из какого отдела."""

    user_id: str
    department: str


TICKETS = {
    "u-101": ["T-5501 не приходит письмо о доставке", "T-5502 задвоился платёж"],
    "u-202": ["T-6100 не открывается отчёт по продажам"],
}


@tool
def list_my_tickets(runtime: ToolRuntime[SessionContext]) -> str:
    """Вернуть список обращений текущего пользователя."""
    user_id = runtime.context.user_id
    tickets = TICKETS.get(user_id, [])

    if not tickets:
        return "Открытых обращений нет."

    return "\n".join(tickets)


agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[list_my_tickets],
    system_prompt="Вы помощник службы поддержки. Отвечайте по-русски и коротко.",
    context_schema=SessionContext,
)

QUESTION = "Какие у меня открыты обращения?"

for user_id in ("u-101", "u-202"):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": QUESTION}]},
        context=SessionContext(user_id=user_id, department="support"),
    )
    answer = result["messages"][-1].text.replace("\n", " ")
    print(f"ПОЛЬЗОВАТЕЛЬ {user_id}: {answer}")

print()
print("АРГУМЕНТЫ, КОТОРЫЕ ВИДИТ МОДЕЛЬ:", list_my_tickets.args)
