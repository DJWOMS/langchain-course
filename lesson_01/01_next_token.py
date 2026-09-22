"""Урок 1, пример 1. Модель выбирает следующий токен из распределения.

Просим модель продолжить фразу и запрашиваем логарифмы вероятностей: для каждой
позиции провайдер возвращает выбранный токен и несколько ближайших конкурентов
с их вероятностями. Это видно глазами, а не на слово.

Запуск: python 01_next_token.py

Основания:
1) параметр logprobs и чтение response_metadata["logprobs"],
   oss/langchain/models.mdx, раздел "Log probabilities"
2) смысл logprobs и top_logprobs (до 20 кандидатов на позицию),
   https://api-docs.deepseek.com/api/create-chat-completion, обращение 31.08.2026
3) bind() как способ дописать параметры к вызову модели,
   oss/langchain/models.mdx, раздел "Log probabilities"

Провайдер может не отдавать logprobs: в документации LangChain это оговорено
словами "certain models can be configured to return". Скрипт это переживает
и говорит об этом вслух.
"""

import math

from course_model import build_model

# temperature=0 здесь не ради воспроизводимости, а чтобы модель брала самого
# вероятного кандидата и картинка распределения читалась однозначно.
model = build_model(temperature=0).bind(logprobs=True, top_logprobs=5)

response = model.invoke("Продолжите фразу тремя словами: столица Франции, это")

print("ОТВЕТ:", response.text)
print()

logprobs = response.response_metadata.get("logprobs")

if not logprobs:
    print("Провайдер не вернул logprobs.")
    print("Это не ошибка примера: поле необязательное, и часть шлюзов его режет.")
    print("Механика от этого не меняется, но увидеть её на своём ключе не выйдет.")
    raise SystemExit(0)

print("Первые пять позиций и кандидаты на каждую:")
for position in logprobs["content"][:5]:
    print(f"\nвыбран: {position['token']!r}")
    for candidate in position.get("top_logprobs", []):
        # Провайдер отдаёт натуральный логарифм вероятности, exp возвращает
        # обратно долю от единицы.
        probability = math.exp(candidate["logprob"])
        print(f"    {candidate['token']!r:<20} {probability:.4f}")
