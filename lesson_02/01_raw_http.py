"""Пример 1 урока 2: вызов модели без фреймворка, голым HTTP.

Тот же запрос, что в примере 2, но собранный руками. Нужен, чтобы увидеть,
что именно уходит на сервер провайдера и что приходит обратно.

Основания:
1) формат запроса и ответа, справочник API DeepSeek, "Create Chat Completion":
   POST на /chat/completions, заголовки Content-Type и Authorization: Bearer,
   обязательные поля тела model и messages, поля ответа choices и usage
   https://api-docs.deepseek.com/api/create-chat-completion
2) совместимость формата с OpenAI, https://api-docs.deepseek.com/
3) адрес, совместимый с OpenAI Chat Completions API, как штатный способ
   подключения: oss/langchain/models.mdx, раздел "Base URL and proxy settings"
"""

import json
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("MODEL_BASE_URL")

if not base_url:
    print("MODEL_BASE_URL пуст.")
    print("Этот пример показывает голый вызов адреса, совместимого с OpenAI.")
    print("Впишите адрес своего провайдера в .env и запустите снова.")
    raise SystemExit(0)

url = base_url.rstrip("/") + "/chat/completions"

payload = {
    "model": os.environ["MODEL_NAME"],
    "messages": [
        {"role": "user", "content": "Ответьте одним словом: столица Франции"}
    ],
    "temperature": 0,
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    },
    method="POST",
)

with urllib.request.urlopen(request, timeout=60) as connection:
    body = json.loads(connection.read().decode("utf-8"))

choice = body["choices"][0]

print("КЛЮЧИ ОТВЕТА:", sorted(body))
print("ТЕКСТ:", choice["message"]["content"])
print("ПРИЧИНА ОСТАНОВКИ:", choice.get("finish_reason"))
print("РАСХОД:", body.get("usage"))
