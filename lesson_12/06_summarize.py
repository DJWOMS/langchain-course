"""Пример 6 урока 12: суммаризация вместо обрезки.

Тот же диалог, что в примерах 4 и 5. Старые сообщения здесь
заменяются сводкой, её пишет отдельный вызов модели внутри встроенного
middleware SummarizationMiddleware.

Middleware срабатывает при шести сообщениях в thread. После этого в thread
остаются два последних сообщения, а перед ними встаёт сводка.

Сводку пишет middleware по собственному промпту, system_prompt агента на неё
не действует. Поэтому промпт сводки задан отдельно, параметром summary_prompt,
и на русском языке.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

SUMMARY_PROMPT = (
    "Кратко перескажите по-русски историю переписки ниже. Сохраните все "
    "факты, которые пользователь просил запомнить, и принятые в диалоге "
    "решения.\n\n"
    "Переписка:\n{messages}"
)

agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[],
    system_prompt=(
        "Вы помощник склада. Отвечайте по-русски, тремя-четырьмя развёрнутыми "
        "фразами. Если нужного факта в диалоге не было, так и скажите."
    ),
    middleware=[
        SummarizationMiddleware(
            model=build_model(temperature=0, max_tokens=512),
            trigger=("messages", 6),
            keep=("messages", 2),
            summary_prompt=SUMMARY_PROMPT,
        )
    ],
    checkpointer=InMemorySaver(),
)

CONFIG = {"configurable": {"thread_id": "sum-1"}}

QUESTIONS = [
    "Запомните: моя смена начинается в 7:40.",
    "Опишите порядок приёмки паллет на складе.",
    "А как оформлять брак, найденный при приёмке?",
    "Во сколько начинается моя смена?",
]


def find_summary(messages):
    """Сообщение со сводкой, если middleware её уже собрал."""
    for message in messages:
        if message.additional_kwargs.get("lc_source") == "summarization":
            return message
    return None


print(f"{'вызов':>5}  {'сообщений в thread':>18}  сводка в thread")

for number, question in enumerate(QUESTIONS, start=1):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        CONFIG,
    )
    summary = find_summary(result["messages"])
    print(f"{number:>5}  {len(result['messages']):>18}  {summary is not None}")

print()
summary = find_summary(result["messages"])
if summary is None:
    print("СВОДКА: middleware не сработал, условие срабатывания не выполнилось")
else:
    print("СВОДКА:")
    print(summary.text)
    print()
    # После сводки в thread лежит хвост, сохранённый по keep, и ответ на последний вопрос.
    print("ХВОСТ, СОХРАНЁННЫЙ ПОСЛЕ СВОДКИ:")
    for message in result["messages"][1:-1]:
        print(f"  [{message.type}] {message.text.replace(chr(10), ' ')}")

print()
print("ПОСЛЕДНИЙ ВОПРОС:", QUESTIONS[-1])
print("ПОСЛЕДНИЙ ОТВЕТ: ", result["messages"][-1].text.replace("\n", " "))
