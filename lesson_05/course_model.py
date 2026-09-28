"""Общая сборка модели для примеров урока 5.

Тот же модуль, что course_model.py урока 4, без изменений: у build_model есть
необязательный первый аргумент с именем модели, а рядом лежит gateway_kwargs()
для примеров, которые собирают модель сами.

Файл .env берётся тот же, что в уроке 0. Положите его рядом с этой папкой или
выше по дереву: load_dotenv() ищет файл начиная с папки этого модуля и поднимается
вверх.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def gateway_kwargs():
    """Возвращает аргументы доступа к провайдеру: имя, адрес, ключ.

    Нужны примерам, которые зовут init_chat_model сами. У настраиваемой модели
    имени модели при создании нет, поэтому build_model ей не подходит.
    """
    base_url = os.getenv("MODEL_BASE_URL")

    if base_url:
        return {
            "model_provider": "openai",
            "base_url": base_url,
            "api_key": os.environ["OPENAI_API_KEY"],
        }

    return {}


def build_model(model_name=None, **kwargs):
    """Собирает модель курса.

    model_name без значения означает модель из переменной MODEL_NAME. Явное имя
    нужно примерам, где моделей в приложении больше одной.

    Все именованные аргументы уходят в init_chat_model как есть: temperature,
    max_tokens, timeout, max_retries, rate_limiter, profile и прочее из раздела
    Parameters.
    """
    model_name = model_name or os.environ["MODEL_NAME"]
    access = gateway_kwargs()

    if access:
        # Путь для любого адреса, совместимого с OpenAI Chat Completions API.
        return init_chat_model(model=model_name, **access, **kwargs)

    # Путь напрямую к провайдеру.
    return init_chat_model(model_name, **kwargs)
