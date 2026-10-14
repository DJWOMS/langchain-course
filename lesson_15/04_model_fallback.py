"""Пример 4 урока 15: запасная модель вместо упавшей основной.

Основная модель заведомо сломана: имя модели такое, которого у провайдера нет.
Первый запуск показывает, чем это кончается без middleware, второй, что делает
ModelFallbackMiddleware.

Запасная модель передаётся готовым объектом, а не строкой: строку middleware отдаёт
в init_chat_model сам, без адреса и ключа шлюза курса.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

from course_model import build_model

SYSTEM_PROMPT = "Вы помощник разработчика. Отвечайте по-русски одной фразой."
QUESTION = "Назовите две причины, по которым вызов модели может упасть."

broken = build_model("no-such-model-in-this-gateway", temperature=0, max_tokens=256)
spare = build_model(temperature=0, max_tokens=256)

plain = create_agent(model=broken, tools=[], system_prompt=SYSTEM_PROMPT)

with_fallback = create_agent(
    model=broken,
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    middleware=[ModelFallbackMiddleware(spare)],
)


def answered_by(message):
    """Имя модели из ответа. Провайдер может его не прислать."""
    return message.response_metadata.get("model_name", "провайдер не сообщил")


print("ЗАПУСК БЕЗ MIDDLEWARE")
try:
    result = plain.invoke({"messages": [{"role": "user", "content": QUESTION}]})
except Exception as exc:  # noqa: BLE001 - пример печатает тип падения
    print(f"  запуск упал: {type(exc).__name__}")
else:
    print("  запуск дошёл до конца:", result["messages"][-1].text)

print()
print("ЗАПУСК С MIDDLEWARE")
result = with_fallback.invoke({"messages": [{"role": "user", "content": QUESTION}]})
answer = result["messages"][-1]
print("  отвечала модель:", answered_by(answer))
print("  ответ:", answer.text.replace("\n", " "))
