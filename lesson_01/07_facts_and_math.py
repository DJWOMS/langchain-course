"""Урок 1, пример 7. Арифметика и выдуманные факты.

Две проверки:
1) произведение двух двенадцатизначных чисел. Правильный ответ считает Python,
   и сравнение делает он же, а не ваш глаз
2) вопрос про функцию, которой в LangChain нет. Смотрим, скажет ли модель
   "не знаю" или напишет уверенное описание

Запуск: python 07_facts_and_math.py

Основания:
1) модели угадывают при неуверенности, потому что обучение и оценка поощряют
   догадку сильнее, чем признание незнания. A. T. Kalai, O. Nachum, S. Vempala,
   E. Zhang, "Why Language Models Hallucinate", arXiv:2509.04664, 04.09.2025,
   https://arxiv.org/abs/2509.04664, обращение 31.08.2026
2) разрешить модели сказать "я не знаю", это первый приём против выдумок,
   https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations,
   обращение 31.08.2026
3) usage_metadata на ответе, oss/langchain/messages.mdx, раздел "AI message"
"""

import re

from course_model import build_model

A = 972_345_618_407
B = 851_209_366_773

model = build_model(temperature=0)

print("ПРОВЕРКА ПЕРВАЯ: АРИФМЕТИКА")
question = f"Посчитайте {A} * {B}. В ответе дайте только число, без пояснений."
answer = model.invoke(question)

truth = A * B
digits = re.sub(r"\D", "", answer.text)

print(f"  вопрос:        {A} * {B}")
print(f"  ответ модели:  {answer.text.strip()}")
print(f"  ответ Python:  {truth}")
print(f"  совпало:       {digits == str(truth)}")

print()
print("ПРОВЕРКА ВТОРАЯ: НЕСУЩЕСТВУЮЩАЯ ФУНКЦИЯ")
fake = model.invoke(
    "Опишите параметры функции create_agent_pool из LangChain "
    "и приведите пример вызова."
)
print(f"  {fake.text.strip()}")

print()
print("ТОТ ЖЕ ВОПРОС, НО С РАЗРЕШЕНИЕМ НЕ ЗНАТЬ")
honest = model.invoke(
    "Опишите параметры функции create_agent_pool из LangChain "
    "и приведите пример вызова. Если такой функции нет или вы не уверены, "
    "ответьте ровно одной фразой: не знаю."
)
print(f"  {honest.text.strip()}")
