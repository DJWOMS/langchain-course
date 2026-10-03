"""Пример 7 урока 7: служебный вызов модели в потоке не нужен.

Внутри инструмента работает вторая модель, служебная. Её ответ пользователю не
нужен, а в поток режима messages она попадает наравне с основной. Метка nostream
убирает её из потока, не отменяя самого вызова.
"""

from collections import Counter

from langchain.agents import create_agent

from course_model import build_model

INTERNAL = {"model": None}


def classify(text: str) -> str:
    """Определяет тему обращения одним словом."""
    answer = INTERNAL["model"].invoke(
        [{"role": "user", "content": f"Одним словом назовите тему обращения: {text}"}]
    )
    return answer.text.strip()


def run(title: str, tagged: bool) -> None:
    """Прогоняет агента и считает chunks потока по узлам графа."""
    print(title)

    service = build_model(temperature=0, max_tokens=1024)
    INTERNAL["model"] = service.with_config({"tags": ["nostream"]}) if tagged else service

    agent = create_agent(
        model=build_model(temperature=0, max_tokens=1024),
        tools=[classify],
        system_prompt="Отвечайте по-русски, одним предложением. Тему определяйте инструментом.",
    )

    by_node = Counter()
    answer = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "У меня дважды списали деньги за заказ 4412."}]},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue
        token, metadata = chunk["data"]
        by_node[metadata.get("langgraph_node", "?")] += 1
        if metadata.get("langgraph_node") == "model":
            answer += token.text

    print(f"  chunks по узлам: {dict(by_node)}")
    print(f"  ответ пользователю: {answer.strip()[:70]!r}")


run("СЛУЖЕБНАЯ МОДЕЛЬ БЕЗ МЕТКИ", tagged=False)
print()
run("СЛУЖЕБНАЯ МОДЕЛЬ С МЕТКОЙ nostream", tagged=True)
