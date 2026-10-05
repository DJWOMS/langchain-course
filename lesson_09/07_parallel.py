"""Пример 7. Несколько вызовов инструмента на одном шаге идут одновременно.

Инструмент спит секунду. Пример печатает, сколько вызовов модель
запросила на одном шаге и сколько времени заняло их исполнение.
"""

import threading
import time

from course_model import build_model
from langchain.agents import create_agent
from langchain.tools import tool

LOG = []
LOCK = threading.Lock()
START = time.perf_counter()


@tool
def check_stock(city: str) -> str:
    """Узнать остаток товара на складе города. Запрос идёт около секунды."""
    began = time.perf_counter() - START
    time.sleep(1.0)
    ended = time.perf_counter() - START
    with LOCK:
        LOG.append((city, began, ended, threading.current_thread().name))
    return f"на складе {city} осталось 7 штук"


agent = create_agent(
    model=build_model(temperature=0),
    tools=[check_stock],
    system_prompt=(
        "Вы отвечаете про остатки на складах. Если в вопросе несколько городов, "
        "запросите все города сразу, одним ответом."
    ),
)

question = "Сколько товара на складах в Казани, Самаре и Перми?"


def run(title, config=None):
    LOG.clear()
    began = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    spent = time.perf_counter() - began

    calls_per_turn = [
        len(m.tool_calls)
        for m in result["messages"]
        if type(m).__name__ == "AIMessage" and m.tool_calls
    ]
    print(title)
    print("  шагов с вызовами:", len(calls_per_turn), "вызовов по шагам:", calls_per_turn)
    print("  всего исполнений инструмента:", len(LOG))
    print(f"  время всего прогона: {spent:.2f} с")
    for city, start_at, end_at, thread_name in sorted(LOG, key=lambda row: row[1]):
        print(f"  {city}: {start_at:.2f} -> {end_at:.2f} ({thread_name})")
    print()


run("ПРОГОН 1: как есть")
run("ПРОГОН 2: max_concurrency=1", config={"max_concurrency": 1})
