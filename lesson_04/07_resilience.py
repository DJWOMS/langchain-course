"""Пример 7 урока 4: таймаут, повторы и типы отказов.
"""

import time

from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

from course_model import build_model

EXCEPTIONS = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    ContextOverflowError,
)

print("ЧТО ФРЕЙМВОРК СЧИТАЕТ ПРИГОДНЫМ ДЛЯ ПОВТОРА")

for exception_type in EXCEPTIONS:
    mark = "повторяем" if exception_type.is_retryable else "не повторяем"
    print(f"  {exception_type.__name__:<28} {mark}")

print()

# Таймаут в одну десятитысячную секунды не успеет никто. Повторы выключены,
# чтобы измерить одну попытку.
print("ТАЙМАУТ БЕЗ ПОВТОРОВ")
impatient = build_model(temperature=0, timeout=0.0001, max_retries=0)
started = time.perf_counter()

try:
    impatient.invoke("Здравствуйте")
except ModelTimeoutError as error:
    print(f"  ModelTimeoutError за {time.perf_counter() - started:.2f} с")
    print(f"  is_retryable: {error.is_retryable}")
except Exception as error:
    print(f"  пришёл другой тип: {type(error).__name__}: {str(error)[:150]}")
else:
    print("  ответ успел прийти, что для такого таймаута неожиданно")

print()

# Тот же безнадёжный запрос, но с тремя повторами. Смотрите на время.
print("ТОТ ЖЕ ТАЙМАУТ С ТРЕМЯ ПОВТОРАМИ")
patient = build_model(temperature=0, timeout=0.0001, max_retries=3)
started = time.perf_counter()

try:
    patient.invoke("Здравствуйте")
except Exception as error:
    print(f"  {type(error).__name__} за {time.perf_counter() - started:.2f} с")

print()
print("НАСТРОЙКИ, КОТОРЫЕ ВИДНО НА ОБЪЕКТЕ")
default_model = build_model(temperature=0)
print(f"  max_retries по умолчанию: {default_model.max_retries}")
print(f"  timeout по умолчанию:     {default_model.request_timeout}")
print("  None означает, что число берёт клиент провайдера, а не LangChain")
