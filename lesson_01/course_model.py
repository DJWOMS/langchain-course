"""Общая сборка модели для примеров урока 1.

Ровно та же логика, что в check_setup.py из урока 0, вынесенная в функцию,
чтобы примеры урока не повторяли одни и те же двенадцать строк.

Файл .env берётся тот же, что в уроке 0. Положите его рядом с примерами или
выше по дереву: load_dotenv() ищет файл начиная с папки этого модуля и поднимается
вверх.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def build_model(**kwargs):
    """Собирает модель курса.

    Все именованные аргументы уходят в init_chat_model как есть: temperature,
    max_tokens, model_kwargs и прочее из раздела Parameters.
    """
    model_name = os.environ["MODEL_NAME"]
    base_url = os.getenv("MODEL_BASE_URL")

    if base_url:
        # Путь для любого адреса, совместимого с OpenAI Chat Completions API.
        return init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            **kwargs,
        )

    # Путь напрямую к провайдеру.
    return init_chat_model(model_name, **kwargs)
