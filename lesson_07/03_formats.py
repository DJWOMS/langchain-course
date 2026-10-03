"""Пример 3 урока 7: две формы chunk, v1 по умолчанию и v2 по требованию.

Один и тот же агент, один и тот же набор режимов, разная форма chunk. В v1 форма
зависит от того, сколько режимов вы запросили и нужны ли подграфы. В v2 форма
всегда одна: словарь с ключами type, ns и data.
"""

from langchain.agents import create_agent

from course_model import build_model


def get_weather(city: str) -> str:
    """Возвращает погоду в городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски, одним предложением.",
)

PAYLOAD = {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]}
MODES = ["updates", "messages"]

print("ФОРМА v1, ПО УМОЛЧАНИЮ")
count = 0
for chunk in agent.stream(PAYLOAD, stream_mode=MODES):
    count += 1
    if count <= 3:
        head = chunk[0] if isinstance(chunk, tuple) else None
        print(f"  chunk {count}: {type(chunk).__name__}, первый элемент {head!r}")
print(f"  всего chunks: {count}")

print()
print("ФОРМА v2")
count = 0
for chunk in agent.stream(PAYLOAD, stream_mode=MODES, version="v2"):
    count += 1
    if count <= 3:
        print(
            f"  chunk {count}: {type(chunk).__name__}, ключи {sorted(chunk)}, "
            f"type={chunk['type']!r}, ns={chunk['ns']!r}"
        )
print(f"  всего chunks: {count}")
