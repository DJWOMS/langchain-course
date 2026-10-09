"""Пример 4 урока 12: чем длиннее thread, тем дороже каждый следующий вызов.

Четыре запроса подряд в один thread. После каждого печатается, сколько
сообщений лежит в thread, во что провайдер оценил вход этого запроса, сколько
входных токенов набежало с начала диалога и какой объём истории насчитал
счётчик LangChain.
"""

from langchain.agents import create_agent
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver

from course_model import build_model

agent = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[],
    system_prompt=(
        "Вы помощник склада. Отвечайте по-русски, тремя-четырьмя развёрнутыми "
        "фразами."
    ),
    checkpointer=InMemorySaver(),
)

CONFIG = {"configurable": {"thread_id": "window-1"}}

QUESTIONS = [
    "Запомните: моя смена начинается в 7:40.",
    "Опишите порядок приёмки паллет на складе.",
    "А как оформлять брак, найденный при приёмке?",
    "Во сколько начинается моя смена?",
]

spent = 0
print(f"{'вызов':>5}  {'сообщений':>9}  {'вход':>6}  {'вход всего':>10}  {'история':>7}")

for number, question in enumerate(QUESTIONS, start=1):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        CONFIG,
    )
    answer = result["messages"][-1]
    # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
    usage = answer.usage_metadata or {}
    incoming = usage.get("input_tokens", 0)
    spent += incoming
    history = count_tokens_approximately(result["messages"])
    print(
        f"{number:>5}  {len(result['messages']):>9}  {incoming:>6}  "
        f"{spent:>10}  {history:>7}"
    )

print()
print("ПОСЛЕДНИЙ ВОПРОС:", QUESTIONS[-1])
print("ПОСЛЕДНИЙ ОТВЕТ: ", result["messages"][-1].text.replace("\n", " "))
