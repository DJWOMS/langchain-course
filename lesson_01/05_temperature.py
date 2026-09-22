"""Урок 1, пример 5. Температура: четыре прогона и контрольный замер.

Четыре запроса на один и тот же вопрос: два при temperature=0 и два при
temperature=1.5. В конце контрольный замер: короткий ответ, длинный счётчик.

Что смотреть в выводе:
1) отличаются ли ответы внутри пары с одинаковой температурой
2) отличается ли пара 1.5 от пары 0
3) насколько счётчик выходных токенов больше видимого текста

Попытка выключить режим рассуждения вынесена в соседний файл
06_thinking_off.py: это отдельный вопрос и отдельный набор параметров.

Запуск: python 05_temperature.py

Основания:
1) параметр temperature, oss/langchain/models.mdx, раздел "Parameters"
2) usage_metadata и output_token_details, oss/langchain/messages.mdx,
   раздел "AI message", подраздел "Token usage"
3) диапазон и значение по умолчанию для temperature (0..2, по умолчанию 1),
   https://api-docs.deepseek.com/api/create-chat-completion, обращение 31.08.2026
4) режим рассуждения включён по умолчанию, а temperature и top_p в нём не
   поддерживаются и молча не действуют,
   https://api-docs.deepseek.com/guides/thinking_mode, обращение 31.08.2026
"""

from course_model import build_model

PROMPT = "Придумайте название для кофейни рядом с университетом. Ответьте только названием."


def show(title, model):
    response = model.invoke(PROMPT)
    # Поле output_token_details заполняет провайдер. Если он не разделяет
    # выход на рассуждение и ответ, словарь придёт пустым, и это тоже факт.
    details = response.usage_metadata.get("output_token_details") or {}
    reasoning = details.get("reasoning", "провайдер не разделил")
    print(f"{title:<24} {response.text.strip()!r}")
    print(f"{'':<24} выход {response.usage_metadata['output_tokens']} токенов, "
          f"из них на рассуждение: {reasoning}")
    return response.text.strip()


cold = build_model(temperature=0)
hot = build_model(temperature=1.5)

print("ТЕМПЕРАТУРА 0")
cold_1 = show("прогон 1", cold)
cold_2 = show("прогон 2", cold)

print()
print("ТЕМПЕРАТУРА 1.5")
hot_1 = show("прогон 1", hot)
hot_2 = show("прогон 2", hot)

print()
print(f"пара при 0 совпала:   {cold_1 == cold_2}")
print(f"пара при 1.5 совпала: {hot_1 == hot_2}")

print()
print("КОНТРОЛЬ: ОДНОСЛОВНЫЙ ОТВЕТ И СЧЁТЧИК ВЫХОДА")
control = cold.invoke("Столица Франции? Ответьте одним словом.")
print(f"  видимый текст:   {control.text.strip()!r}")
print(f"  счётчик выхода:  {control.usage_metadata['output_tokens']} токенов")
