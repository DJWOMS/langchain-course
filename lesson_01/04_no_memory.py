"""Урок 1, пример 4. Модель не помнит предыдущий вызов.

Два независимых вызова против одного вызова со списком сообщений. Разница видна
и в ответе, и в счётчике входных токенов.

Запуск: python 04_no_memory.py

Основания:
1) взаимодействия с моделью не хранят состояние, историю наращивает
   вызывающая сторона, oss/langchain/messages.mdx, раздел "Use with chat models"
2) типы сообщений и их импорт, oss/langchain/messages.mdx, раздел "Message types"
3) сервер провайдера не записывает контекст запросов и всю историю нужно
   присылать заново, https://api-docs.deepseek.com/guides/multi_round_chat,
   обращение 31.08.2026
"""

from langchain.messages import AIMessage, HumanMessage

from course_model import build_model

model = build_model(temperature=0)

print("ДВА ОТДЕЛЬНЫХ ВЫЗОВА")

first = model.invoke("Меня зовут Михаил. Запомните это.")
print("  вопрос 1: Меня зовут Михаил. Запомните это.")
print(f"  ответ 1:  {first.text}")
print(f"  вход:     {first.usage_metadata['input_tokens']} токенов")

second = model.invoke("Как меня зовут? Ответьте одним словом.")
print("  вопрос 2: Как меня зовут? Ответьте одним словом.")
print(f"  ответ 2:  {second.text}")
print(f"  вход:     {second.usage_metadata['input_tokens']} токенов")

print()
print("ТОТ ЖЕ ВТОРОЙ ВОПРОС, НО СО СПИСКОМ СООБЩЕНИЙ")

history = [
    HumanMessage("Меня зовут Михаил. Запомните это."),
    AIMessage(first.text),
    HumanMessage("Как меня зовут? Ответьте одним словом."),
]

third = model.invoke(history)
print(f"  ответ:    {third.text}")
print(f"  вход:     {third.usage_metadata['input_tokens']} токенов")
