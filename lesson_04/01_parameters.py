"""Пример 1 урока 4: три места, где задаётся параметр модели.
"""

from course_model import build_model

QUESTION = "Опишите очередь задач в трёх предложениях."


def show(title, response):
    """Печатает то, по чему видно, подействовал параметр или нет."""
    usage = response.usage_metadata or {}
    finish = response.response_metadata.get("finish_reason")
    print(title)
    print(f"  выход:  {usage.get('output_tokens')} токенов, finish_reason={finish!r}")
    print(f"  начало: {response.text[:60]!r}")
    print()


# 1. Параметр при создании модели: действует на все вызовы этой модели.
short = build_model(temperature=0, max_tokens=32)
show("MAX_TOKENS=32 ПРИ СОЗДАНИИ", short.invoke(QUESTION))

# 2. Параметр, прикреплённый через bind: получается обёртка вокруг модели
#    с добавленным аргументом. Исходная модель не меняется.
longer = short.bind(max_tokens=200)
show("ТА ЖЕ МОДЕЛЬ, BIND(MAX_TOKENS=200)", longer.invoke(QUESTION))

# 3. Параметр на один вызов: уходит в запрос поверх настроек модели.
show("ТА ЖЕ МОДЕЛЬ, MAX_TOKENS=200 НА ВЫЗОВ", short.invoke(QUESTION, max_tokens=200))

# 4. Исходная модель не изменилась ни от bind, ни от параметра на вызов.
show("ИСХОДНАЯ МОДЕЛЬ ПОСЛЕ ОБОИХ ПЕРЕОПРЕДЕЛЕНИЙ", short.invoke(QUESTION))

# 5. Настройки видно и без запроса к провайдеру: они лежат на объекте модели.
print("НАСТРОЙКИ НА ОБЪЕКТЕ МОДЕЛИ")
print(f"  имя модели:  {short.model_name}")
print(f"  max_tokens:  {short.max_tokens}")
print(f"  temperature: {short.temperature}")
print(f"  timeout:     {short.request_timeout}")
print(f"  max_retries: {short.max_retries}")
