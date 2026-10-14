"""Пример 6 урока 15: поиск по файлам без единого своего инструмента.

FilesystemFileSearchMiddleware вовсе не встраивается в цикл. Он приносит
агенту два инструмента, glob_search и grep_search, и ограничивает их корнем,
за который они не выходят.

Корень здесь, это папка с примерами урока, то есть агент ищет по тем файлам,
которые вы только что запускали. Поиск идёт средствами Python: так результат
не зависит от того, установлен у вас ripgrep или нет.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

from course_model import build_model

ROOT = Path(__file__).parent

SYSTEM_PROMPT = (
    "Вы разбираете незнакомый репозиторий. Пользуйтесь поиском по файлам, "
    "а не догадками. Отвечайте по-русски, двумя-тремя фразами."
)

QUESTION = (
    "Сколько в репозитории файлов с расширением .py и в каких из них "
    "встречается ToolRetryMiddleware?"
)

agent = create_agent(
    model=build_model(temperature=0, max_tokens=512),
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        FilesystemFileSearchMiddleware(root_path=str(ROOT), use_ripgrep=False),
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

print("что агент спрашивал у поиска:")
for message in result["messages"]:
    if message.type == "ai":
        for call in message.tool_calls:
            print(f"  {call['name']}({call['args']})")

print()
print("ответ:", result["messages"][-1].text.replace("\n", " "))
