"""Пример 2 урока 4: профиль модели, чтение и правка.
"""

from langchain_openai import ChatOpenAI

from course_model import build_model

FIELDS = (
    "max_input_tokens",
    "max_output_tokens",
    "tool_calling",
    "structured_output",
    "reasoning_output",
    "reasoning_effort_levels",
    "temperature",
    "image_inputs",
    "pdf_inputs",
)

# ПОДСТАВЬТЕ СВОИ ЧИСЛА, ниже они условные. Профиль пишется руками там, где данных
# по имени модели в пакете интеграции нет. Значения берутся со страницы вашего
# провайдера, а не из головы: фреймворк их не проверяет и на запрос не влияет.
COURSE_PROFILE = {
    "max_input_tokens": 128000,
    "max_output_tokens": 8192,
    "tool_calling": True,
    "structured_output": True,
    "reasoning_output": True,
    "temperature": False,
    "image_inputs": False,
}


def show_profile(title, profile):
    """Печатает профиль по одному полю на строку, с пометкой отсутствующих."""
    print(title)

    if not profile:
        print("  профиля нет: данных по этому имени модели в пакете нет")
        print()
        return

    for field in FIELDS:
        if field in profile:
            print(f"  {field:<24} {profile[field]}")
        else:
            print(f"  {field:<24} поля нет")

    print()


show_profile("ПРОФИЛЬ МОДЕЛИ КУРСА", build_model().profile)

# Ключ здесь не используется: профиль читается из данных пакета, запрос
# к провайдеру не уходит.
known = ChatOpenAI(model="gpt-5.5", api_key="not-used-no-request-is-made")
show_profile("ПРОФИЛЬ gpt-5.5 ИЗ ДАННЫХ ПАКЕТА", known.profile)

custom = build_model(profile=COURSE_PROFILE)
show_profile("ПРОФИЛЬ, ЗАДАННЫЙ РУКАМИ", custom.profile)

# Правка готового профиля: новый словарь и копия модели. Так исходная модель
# не меняется, что важно, если её объект уже кому-то отдали.
patched_profile = custom.profile | {"max_input_tokens": 64000}
patched = custom.model_copy(update={"profile": patched_profile})

print("ПОСЛЕ ПРАВКИ ЧЕРЕЗ MODEL_COPY")
print(f"  у копии:    {patched.profile['max_input_tokens']}")
print(f"  у исходной: {custom.profile['max_input_tokens']}")
print()

# Профиль, это справка о модели, а не рычаг управления ею.
print("ЧТО ПРОФИЛЬ НЕ ДЕЛАЕТ")
print(f"  в профиле temperature: {custom.profile['temperature']}")
print("  применит провайдер параметр или нет, профиль не решает")
