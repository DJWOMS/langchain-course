"""Пример 2 урока 2: тот же запрос через LangChain.

Основания:
1) init_chat_model и invoke: oss/langchain/models.mdx, разделы
   "Initialize a model" и "Invoke"
2) атрибуты сообщения text, usage_metadata, response_metadata:
   oss/langchain/messages.mdx, раздел "AI message", врезка Attributes
3) состав usage_metadata: oss/langchain/messages.mdx, раздел "Token usage"
"""

from course_model import build_model

model = build_model(temperature=0)

response = model.invoke("Ответьте одним словом: столица Франции")

print("ТИП ОТВЕТА:", type(response).__name__)
print("ТЕКСТ:", response.text)
print("КЛЮЧИ response_metadata:", sorted(response.response_metadata))
print("РАСХОД:", response.usage_metadata)
