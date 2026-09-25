"""Пример 1 урока 3: четыре роли и три способа их записать.

Запуск: python 01_roles.py
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from course_model import build_model

model = build_model(temperature=0)

# 1. Диалог объектами сообщений. Роль задаётся классом, а не полем.
dialogue = [
    SystemMessage("Вы отвечаете одним предложением, без вступлений."),
    HumanMessage("Что такое очередь задач?"),
    AIMessage("Очередь задач, это список работ, которые исполнители разбирают по одной."),
    HumanMessage("А чем она отличается от стека?"),
]

for message in dialogue:
    print(f"{type(message).__name__:14} {message.content!r}")

print()
print("ОТВЕТ НА ОБЪЕКТАХ:", model.invoke(dialogue).text)
print()

# 2. Тот же диалог словарями. Роль задаётся полем role.
as_dicts = [
    {"role": "system", "content": "Вы отвечаете одним предложением, без вступлений."},
    {"role": "user", "content": "Что такое очередь задач?"},
    {"role": "assistant", "content": "Очередь задач, это список работ, которые исполнители разбирают по одной."},
    {"role": "user", "content": "А чем она отличается от стека?"},
]

print("ОТВЕТ НА СЛОВАРЯХ:", model.invoke(as_dicts).text)
print()

# 3. Строка это сокращение для списка из одного HumanMessage.
print("ОТВЕТ НА СТРОКУ:", model.invoke("Что такое очередь задач? Одно предложение.").text)
print()

# 4. Необязательные поля сообщения: имя автора и идентификатор.
named = HumanMessage(content="Здравствуйте!", name="alice", id="msg_123")

print("NAME:", named.name)
print("ID:  ", named.id)
print("ТИП ОТВЕТА МОДЕЛИ:", type(model.invoke([named])).__name__)
