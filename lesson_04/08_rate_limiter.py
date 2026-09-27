"""Пример 8 урока 4: ограничитель частоты запросов.
"""

import time

from langchain.rate_limiters import InMemoryRateLimiter

from course_model import build_model

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # один запрос раз в две секунды
    check_every_n_seconds=0.1,  # как часто просыпаться и смотреть, можно ли
    max_bucket_size=1,  # сколько запросов разрешено выпустить пачкой
)

model = build_model(temperature=0, max_tokens=16, rate_limiter=rate_limiter)

print("ТРИ ЗАПРОСА ПОДРЯД, РАЗРЕШЕНО 0.5 ЗАПРОСА В СЕКУНДУ")
started = time.perf_counter()
previous = started

for number in range(1, 4):
    model.invoke("Ответьте одним словом: да или нет?")
    now = time.perf_counter()
    print(
        f"  запрос {number}: {now - previous:.1f} с от предыдущего, "
        f"{now - started:.1f} с от начала"
    )
    previous = now

print()
print("ОГРАНИЧИТЕЛЬ СЧИТАЕТ ЗАПРОСЫ, А НЕ ТОКЕНЫ")
print(f"  requests_per_second:  {rate_limiter.requests_per_second}")
print(f"  max_bucket_size:      {rate_limiter.max_bucket_size}")
print("  длина запроса на паузу не влияет, лимит провайдера по токенам он не знает")
