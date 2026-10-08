"""Пример 2 урока 10: что из подписи инструмента попадает в схему для модели.

Сети здесь нет, модель не вызывается. Для каждого инструмента сравниваются две
схемы: полная, по которой фреймворк собирает вызов, и та, что уходит в модель.
Последние два инструмента названы зарезервированными именами, и по разнице
между схемами видно, к чему это приводит.
"""

from typing import Any

from langchain.tools import ToolRuntime, tool


@tool
def weather_plain(city: str) -> str:
    """Вернуть погоду в городе."""
    return f"В городе {city} ясно."


@tool
def weather_with_runtime(city: str, runtime: ToolRuntime) -> str:
    """Вернуть погоду в городе и номер вызова инструмента."""
    return f"В городе {city} ясно. Вызов {runtime.tool_call_id}."


@tool
def weather_bad_runtime(city: str, runtime: str) -> str:
    """Вернуть погоду в городе с пометкой о режиме работы."""
    return f"В городе {city} ясно. Режим {runtime}."


@tool
def weather_bad_config(city: str, config: dict) -> str:
    """Вернуть погоду в городе с учётом настроек."""
    return f"В городе {city} ясно. Настройки {config}."


def full_schema_fields(some_tool: Any) -> list[str]:
    """Имена полей полной схемы инструмента.

    У инструмента, собранного из функции, args_schema это модель Pydantic.
    Защитная ветка нужна на случай схемы, заданной словарём JSON Schema.
    """
    schema = some_tool.args_schema

    if isinstance(schema, dict):
        return sorted(schema.get("properties", {}))

    return sorted(schema.model_fields)


TOOLS = [
    weather_plain,
    weather_with_runtime,
    weather_bad_runtime,
    weather_bad_config,
]

print(f"{'инструмент':<22} {'полная схема':<34} схема для модели")
for item in TOOLS:
    full = ", ".join(full_schema_fields(item))
    visible = ", ".join(sorted(item.args))
    print(f"{item.name:<22} {full:<34} {visible}")

print()
print("ПОДРОБНО ПРО weather_with_runtime")
print("  тип поля runtime в полной схеме:", weather_with_runtime.args_schema.model_fields["runtime"].annotation)
print("  описание для модели:", weather_with_runtime.description)
