"""Урок 1, пример 3. Размер контекстного окна как данные, а не как слухи.

LangChain отдаёт характеристики модели через атрибут profile. Данные для него
приходят из проекта models.dev и лежат внутри пакета интеграции, поэтому запроса
в сеть здесь не будет и денег этот пример не стоит.

Запуск: python 03_context_window.py

Основания:
1) атрибут profile и поля max_input_tokens, tool_calling и прочие,
   oss/langchain/models.mdx, раздел "Model profiles" (нужен langchain>=1.1)
2) источник данных models.dev и то, что профиль поставляется вместе с пакетом,
   там же
3) профиль можно задать своими руками параметром profile,
   там же, врезка "Updating or overwriting profile data"
4) определение контекстного окна: максимальное число токенов, которое можно
   передать модели, oss/concepts/context.mdx, врезка в начале страницы
"""

from langchain_openai import ChatOpenAI

from course_model import build_model

model = build_model()
profile = model.profile

print("ПРОФИЛЬ МОДЕЛИ КУРСА")
if profile is None:
    print("Профиль не найден: для этого имени модели данных в пакете нет.")
    print("Так бывает у шлюзов, где имя модели выглядит как provider/model.")
    print("Размер окна тогда берётся со страницы вашего провайдера,")
    print("а в код его кладут параметром profile при создании модели.")
else:
    print(f"  max_input_tokens:  {profile.get('max_input_tokens')}")
    print(f"  max_output_tokens: {profile.get('max_output_tokens')}")
    print(f"  tool_calling:      {profile.get('tool_calling')}")
    print(f"  reasoning_output:  {profile.get('reasoning_output')}")

print()
print("ТРИ ИМЕНИ, У КОТОРЫХ ПРОФИЛЬ ЕСТЬ ВСЕГДА")
print("Ключ здесь не используется: profile читается из данных пакета,")
print("запрос к провайдеру не уходит.")

for name in ("gpt-4o-mini", "gpt-5-nano", "gpt-5.5"):
    known = ChatOpenAI(model=name, api_key="not-used-no-request-is-made")
    known_profile = known.profile or {}
    print(
        f"  {name:<12} вход до {known_profile.get('max_input_tokens')} токенов, "
        f"выход до {known_profile.get('max_output_tokens')}"
    )
