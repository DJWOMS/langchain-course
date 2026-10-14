"""Пример 3 урока 15: тот же набор middleware в двух порядках.

Повтор упавшего инструмента и обработка его ошибки. Middleware те же, разница
только в порядке внутри списка middleware.

Инструмент падает всегда и считает свои запуски. По отношению числа запусков
к числу вызовов, которые сделала модель, видно, отработали повторы или нет.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware, ToolRetryMiddleware

from course_model import build_model

attempts = {"count": 0}


def fetch_index(path: str) -> str:
    """Загружает индекс модулей репозитория по пути."""
    attempts["count"] += 1
    raise ConnectionError(f"хранилище не ответило на запрос {path}")


def on_error(exc, request):
    """Превращает отказ хранилища в сообщение для модели."""
    if isinstance(exc, ConnectionError):
        return (
            f"Инструмент {request.tool_call['name']} недоступен "
            f"({type(exc).__name__}). Повторять вызов не нужно, сообщите об "
            f"этом пользователю."
        )
    return None


def retry_layer():
    """Повтор с короткой паузой, чтобы пример отработал за секунды."""
    return ToolRetryMiddleware(
        max_retries=2,
        initial_delay=0.2,
        backoff_factor=1.0,
        jitter=False,
        on_failure="error",
    )


SYSTEM_PROMPT = (
    "Вы помощник по разбору легаси. Отвечайте по-русски одной фразой. "
    "Если инструмент сообщил об ошибке, не повторяйте вызов."
)

QUESTION = "Загрузите индекс модулей по пути /src и скажите, что получилось."

model = build_model(temperature=0, max_tokens=256)


def run(label, middleware):
    """Собирает агента с заданным порядком middleware, отправляет вопрос и печатает строку отчёта."""
    attempts["count"] = 0
    agent = create_agent(
        model=model,
        tools=[fetch_index],
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    calls = sum(1 for message in result["messages"] if message.type == "tool")
    per_call = attempts["count"] / calls if calls else 0
    print(f"{label:<24}{calls:>8}{attempts['count']:>10}{per_call:>12.1f}")
    return result


print(f"{'порядок middleware':<24}{'вызовов':>8}{'запусков':>10}{'на вызов':>12}")

first = run("повтор, обработка", [retry_layer(), ToolErrorMiddleware(on_error)])
second = run("обработка, повтор", [ToolErrorMiddleware(on_error), retry_layer()])

print()
print("ответ при порядке 'повтор, обработка':", first["messages"][-1].text)
print("ответ при порядке 'обработка, повтор':", second["messages"][-1].text)
