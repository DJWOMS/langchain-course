"""Пример 6 урока 4: инструмент на стороне провайдера.

Отказ здесь, это нормальный исход: серверные инструменты есть не у каждого
провайдера и не на каждом адресе, совместимом с OpenAI. Пример печатает то,
что вернулось, каким бы оно ни было.
"""

from course_model import build_model

model = build_model(temperature=0)

profile = model.profile or {}
print("ЧТО ГОВОРИТ ПРОФИЛЬ ДО ЗАПРОСА")
print(f"  tool_calling: {profile.get('tool_calling', 'поля нет')}")
print("  отдельного поля про серверные инструменты в профиле нет")
print()

# Инструмент провайдера описывается словарём с типом, а не функцией Python.
server_tool = {"type": "web_search"}
model_with_tools = model.bind_tools([server_tool])

print("ЗАПРОС С ИНСТРУМЕНТОМ web_search")

try:
    response = model_with_tools.invoke("What was a positive news story from today?")
except Exception as error:
    print(f"  провайдер отказал: {type(error).__name__}")
    print(f"  {str(error)[:300]}")
    print("  Инструмента на стороне этого провайдера нет, идём своими инструментами")
    print("  (урок 9), а поиск в сети делаем сами.")
else:
    print("  типы блоков:", [block["type"] for block in response.content_blocks])
    print()

    for block in response.content_blocks:
        if block["type"] == "server_tool_call":
            print(f"  вызов:     {block.get('name')} {block.get('args')}")
        elif block["type"] == "server_tool_result":
            print(f"  результат: {block.get('status')}")
        elif block["type"] == "text":
            print(f"  текст:     {block['text'][:150]!r}")
            for annotation in block.get("annotations") or []:
                print(f"    источник: {annotation.get('url')}")

    print()
    print("  вызовов инструментов для нас:", len(response.tool_calls))
    print("  ToolMessage отправлять не нужно: провайдер всё сделал у себя")
