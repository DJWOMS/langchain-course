"""Урок 1, пример 8. Диалог на три хода: токены и деньги.

История отправляется заново на каждом ходу, поэтому вход растёт от хода к ходу,
а вместе с ним растёт счёт. Итог собирает UsageMetadataCallbackHandler: он
складывает расход по всем вызовам и раскладывает его по именам моделей.

Запуск: python 08_cost.py

Основания:
1) UsageMetadataCallbackHandler и передача его через config={"callbacks": [...]},
   oss/langchain/models.mdx, раздел "Token usage"
2) поля usage_metadata, в том числе input_token_details с cache_read,
   oss/langchain/messages.mdx, раздел "AI message", подраздел "Token usage"
3) кеширование префикса включено на стороне провайдера и видно в usage,
   oss/langchain/models.mdx, раздел "Prompt caching"
4) ставки за миллион токенов, отдельные для входа с попаданием в кеш, входа без
   попадания и выхода, https://api-docs.deepseek.com/quick_start/pricing,
   обращение 22.09.2026
5) попадание в кеш требует полного совпадения префикса,
   https://api-docs.deepseek.com/guides/kv_cache, обращение 31.08.2026
"""

from langchain.messages import AIMessage, HumanMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

from course_model import build_model

# ПОДСТАВЬТЕ СВОИ СТАВКИ. Здесь цены модели курса на 22.09.2026, доллары за
# миллион токенов, ночной тариф. У вашего провайдера они другие и меняются
# часто.
PRICE_INPUT_MISS = 0.15
PRICE_INPUT_HIT = 0.003
PRICE_OUTPUT = 0.60

QUESTIONS = [
    "Назовите три задачи, где агент с инструментами выигрывает у одного запроса к модели.",
    "Возьмите первую из них и опишите, какие инструменты понадобятся.",
    "А теперь оцените, сколько вызовов модели уйдёт на один прогон такой задачи.",
]

callback = UsageMetadataCallbackHandler()
model = build_model(temperature=0)

history = []

for number, question in enumerate(QUESTIONS, start=1):
    history.append(HumanMessage(question))
    response = model.invoke(history, config={"callbacks": [callback]})
    history.append(AIMessage(response.text))

    usage = response.usage_metadata
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    print(
        f"ход {number}: вход {usage['input_tokens']} "
        f"(из кеша {cached}), выход {usage['output_tokens']}"
    )

print()
print("ИТОГ ПО ВСЕМ ВЫЗОВАМ")

for model_name, usage in callback.usage_metadata.items():
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    fresh = usage["input_tokens"] - cached
    money = (
        fresh * PRICE_INPUT_MISS
        + cached * PRICE_INPUT_HIT
        + usage["output_tokens"] * PRICE_OUTPUT
    ) / 1_000_000

    print(f"  модель: {model_name}")
    print(f"  вход:   {usage['input_tokens']} токенов, из них из кеша {cached}")
    print(f"  выход:  {usage['output_tokens']} токенов")
    print(f"  всего:  {usage['total_tokens']} токенов")
    print(f"  деньги: ${money:.6f}")
