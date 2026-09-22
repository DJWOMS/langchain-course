"""Урок 1, пример 6. Попытка отключить режим рассуждения.

Два рычага, оба на одном и том же вопросе:
1) полное выключение режима через собственное поле провайдера
2) снижение усилия через стандартный параметр reasoning_effort

Смотреть надо на счётчик выходных токенов: если рычаг сработал, он заметно
падает. Оба вызова обёрнуты в обработку отказа, потому что поле принадлежит
провайдеру, а между вами и провайдером может стоять шлюз.

Запуск: python 06_thinking_off.py

Основания:
1) extra_body как способ передать провайдеру его собственные поля,
   oss/python/integrations/chat/perplexity.mdx, раздел "Using perplexity-specific
   parameters through ChatPerplexity"
2) стандартный параметр reasoning_effort, задаётся при создании модели или на
   вызов, oss/langchain/models.mdx, раздел "Reasoning"
3) провайдер требует передавать thinking именно внутри extra_body, а
   reasoning_effort обычным параметром, и отображает уровни low, high, max,
   https://api-docs.deepseek.com/guides/thinking_mode, обращение 31.08.2026
"""

from course_model import build_model

PROMPT = "Придумайте название для кофейни рядом с университетом. Ответьте только названием."

model = build_model(temperature=0)

print("БЕЗ РЫЧАГОВ, ДЛЯ СРАВНЕНИЯ")
base = model.invoke(PROMPT)
print(f"  {base.text.strip()!r}, выход {base.usage_metadata['output_tokens']} токенов")

print()
print("РЫЧАГ 1: ВЫКЛЮЧИТЬ РЕЖИМ РАССУЖДЕНИЯ")
try:
    off = model.invoke(PROMPT, extra_body={"thinking": {"type": "disabled"}})
    print(f"  {off.text.strip()!r}, выход {off.usage_metadata['output_tokens']} токенов")
except Exception as error:  # noqa: BLE001
    print(f"  отказ: {type(error).__name__}")
    print(f"  текст: {error}")

print()
print("РЫЧАГ 2: СНИЗИТЬ УСИЛИЕ НА РАССУЖДЕНИЕ")
try:
    low = model.invoke(PROMPT, reasoning_effort="low")
    print(f"  {low.text.strip()!r}, выход {low.usage_metadata['output_tokens']} токенов")
except Exception as error:  # noqa: BLE001
    print(f"  отказ: {type(error).__name__}")
    print(f"  текст: {error}")
