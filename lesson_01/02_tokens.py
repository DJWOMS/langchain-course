"""Урок 1, пример 2. Символы, приблизительная оценка и настоящий счёт токенов.

Для каждой строки печатаются три числа:
1) длина в символах, её вы видите в редакторе
2) оценка count_tokens_approximately, локальная и бесплатная
3) input_tokens из usage_metadata, то, за что списали деньги

Первая строка списка, это одна точка. Она показывает накладную плату: даже
пустой запрос стоит несколько токенов, потому что вокруг текста добавляется
служебная обёртка сообщения.

Запуск: python 02_tokens.py

Основания:
1) usage_metadata на ответе модели, oss/langchain/messages.mdx,
   раздел "AI message", подраздел "Token usage"
2) count_tokens_approximately и его импорт,
   oss/langgraph/add-memory.mdx, раздел "Manage short-term memory"
3) параметр max_tokens ограничивает длину ответа,
   oss/langchain/models.mdx, раздел "Parameters"
4) соотношение символов и токенов зависит от модели и от языка,
   https://api-docs.deepseek.com/quick_start/token_usage, обращение 31.08.2026
"""

from langchain.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from course_model import build_model

SAMPLES = [
    ".",
    "cat",
    "кот",
    "The quick brown fox jumps over the lazy dog",
    "Быстрая бурая лиса прыгает через ленивого пса",
    "1234567890",
    "достопримечательность",
]

# max_tokens режет ответ: платить за длинную генерацию здесь не за что,
# нас интересует только вход.
model = build_model(temperature=0, max_tokens=16)

print(f"{'текст':<48}{'симв.':>7}{'оценка':>8}{'реально':>9}")
print("-" * 72)

for text in SAMPLES:
    response = model.invoke(text)
    approximate = count_tokens_approximately([HumanMessage(text)])
    real = response.usage_metadata["input_tokens"]
    shown = text if len(text) <= 45 else text[:42] + "..."
    print(f"{shown:<48}{len(text):>7}{approximate:>8}{real:>9}")
