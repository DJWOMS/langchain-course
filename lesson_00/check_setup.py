"""Проверка окружения курса. Урок 0.

Скрипт печатает версии установленных пакетов и делает один короткий запрос
к модели. Если он отработал до конца, окружение курса собрано.

Запуск: python check_setup.py

Отступление от документации, сделанное осознанно:
oss/versioning.mdx, раздел "Check your version", предлагает читать версии
через langchain_core.__version__ и langgraph.__version__. Для langgraph это
больше не работает: пакет собран как пространство имён, атрибута __version__
у него нет, и строка роняет скрипт с AttributeError. Версии читаются штатным
importlib.metadata.version() из стандартной библиотеки, он спрашивает номер
у менеджера пакетов и работает для всех трёх пакетов одинаково.
"""

import os
from importlib.metadata import version

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Читает файл .env из текущей папки и кладёт значения в переменные окружения.
load_dotenv()

for package in ("langchain", "langchain-core", "langgraph"):
    print(f"{package}: {version(package)}")

# MODEL_NAME и MODEL_BASE_URL, это переменные курса, а не переменные LangChain.
# Фреймворк про них ничего не знает, мы сами передаём их значения в вызов.
model_name = os.environ["MODEL_NAME"]
base_url = os.getenv("MODEL_BASE_URL")

if base_url:
    # Путь для любого адреса, совместимого с OpenAI Chat Completions API.
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )
else:
    # Путь напрямую к провайдеру. Ключ берётся из переменной окружения,
    # имя которой задаёт сам провайдер (например, OPENAI_API_KEY).
    model = init_chat_model(model_name, temperature=0)

response = model.invoke("Ответьте одним словом: готово")

print(response.text)
print(response.usage_metadata)
