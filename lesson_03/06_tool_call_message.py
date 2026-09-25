"""Пример 6 урока 3: как устроено сообщение с вызовом инструмента.

Цикл собран руками, без агента: модель просит вызвать инструмент, вызов идёт
здесь же, результат возвращается сообщением роли tool. Так видно, из чего агент
из урока 2 состоит внутри.

Запуск: python 06_tool_call_message.py
"""

from langchain.tools import tool

from course_model import build_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"В городе {city} сейчас +17 и дождь."


model_with_tools = build_model(temperature=0).bind_tools([get_weather])

messages = [{"role": "user", "content": "Какая погода в Казани?"}]
ai_message = model_with_tools.invoke(messages)

print("ТИП СООБЩЕНИЯ:", type(ai_message).__name__)
print("content:      ", repr(ai_message.content))
print("text:         ", repr(ai_message.text))
print("tool_calls:   ", ai_message.tool_calls)
print()

if not ai_message.tool_calls:
    print("Модель не попросила инструмент, продолжать цикл не с чем.")
    print("Так бывает: вызов инструментов держат не все модели и не все шлюзы.")
    raise SystemExit(0)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    print("ИМЯ:      ", tool_call["name"])
    print("АРГУМЕНТЫ:", tool_call["args"])
    print("ID:       ", tool_call["id"])

    tool_message = get_weather.invoke(tool_call)

    print("ЧТО ВЕРНУЛ ИНСТРУМЕНТ:", type(tool_message).__name__)
    print("  content:     ", repr(tool_message.content))
    print("  tool_call_id:", tool_message.tool_call_id)
    print("  name:        ", tool_message.name)
    print()

    messages.append(tool_message)

final = model_with_tools.invoke(messages)

print("СОСТАВ ИСТОРИИ:", [type(m).__name__ if not isinstance(m, dict) else m["role"] for m in messages])
print("ФИНАЛЬНЫЙ ОТВЕТ:", final.text)
