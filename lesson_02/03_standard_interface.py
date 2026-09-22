"""Пример 4 урока 2: три метода одного объекта модели.
"""

from course_model import build_model

model = build_model(temperature=0)

# 1. invoke: список сообщений с ролями вместо одной строки.
conversation = [
    {"role": "system", "content": "Вы переводите с русского на французский. Отвечайте только переводом."},
    {"role": "user", "content": "Переведите: я люблю программировать."},
    {"role": "assistant", "content": "J'adore programmer."},
    {"role": "user", "content": "Переведите: я люблю собирать приложения."},
]

print("INVOKE:", model.invoke(conversation).text)
print()

# 2. batch: три независимых запроса уходят параллельно, порядок сохраняется.
questions = [
    "Одним предложением: зачем нужен HTTP?",
    "Одним предложением: зачем нужен DNS?",
    "Одним предложением: зачем нужен TLS?",
]

print("BATCH:")
for question, answer in zip(questions, model.batch(questions)):
    print(f"{question}")
    print(f"{answer.text}")
print()

# 3. stream: тот же объект отдаёт ответ кусками по мере генерации.
print("STREAM:")
for chunk in model.stream("Одним предложением: зачем нужен фреймворк?"):
    print(chunk.text, end="|", flush=True)
print()
