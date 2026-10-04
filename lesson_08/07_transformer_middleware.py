"""Пример 7 урока 8: тот же преобразователь, зарегистрированный через middleware.

Middleware объявляет фабрики преобразователей атрибутом transformers, и агент
добавляет их при сборке графа. Вызывающему коду про них знать не нужно:
stream_events вызывается без аргумента transformers, а проекция в extensions
уже есть.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from course_model import build_model
from tool_activity import ToolActivityTransformer


class ToolActivityMiddleware(AgentMiddleware):
    """Middleware без единого хука: он нужен только ради проекции потока."""

    transformers = (ToolActivityTransformer,)


def get_weather(city: str) -> str:
    """Возвращает погоду в указанном городе."""
    return f"В городе {city} всегда солнечно!"


agent = create_agent(
    model=build_model(temperature=0, max_tokens=1024),
    tools=[get_weather],
    system_prompt="Отвечайте по-русски и коротко.",
    middleware=[ToolActivityMiddleware()],
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Какая погода в Сан-Франциско?"}]},
    version="v3",
)

print("КЛЮЧИ EXTENSIONS:", sorted(stream.extensions))

for record in stream.extensions["tool_activity"]:
    print(f"  {record['event']:<18} {record['tool']}")

print()
print("ОТВЕТ:", stream.output["messages"][-1].text)

# Второй прогон того же агента: видно ли записи именованного канала
# в общем потоке событий.
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Какая погода в Бостоне?"}]},
    version="v3",
)

methods = [event["method"] for event in stream]
print("СОБЫТИЙ ВСЕГО:", len(methods))
print("ИЗ НИХ custom:tool_activity:", methods.count("custom:tool_activity"))
