"""Пример 1 урока 7: поток у самой модели и куда девается расход токенов.

Метод stream() возвращает итератор chunks. Chunks складываются оператором + в одно
сообщение, и только у собранного сообщения есть цельный текст. Первые chunks
приходят без текста: модель курса сначала рассуждает, а ChatOpenAI текст
рассуждения не извлекает. Расход токенов в потоке приходит не всегда: у Chat
Completions его надо попросить отдельно.
"""

from course_model import build_model

QUESTION = "Назовите три причины использовать очередь задач, по одному предложению на причину."


def run(title: str, **extra) -> None:
    """Прогоняет один поток и печатает первые chunks с текстом, счётчики и сборку."""
    print(title)
    model = build_model(temperature=0, max_tokens=1024, **extra)

    full = None
    count = 0
    empty = 0
    shown = 0
    with_usage = 0

    for chunk in model.stream(QUESTION):
        count += 1
        if not chunk.text:
            empty += 1
        elif shown < 3:
            shown += 1
            print(f"  chunk {count}: {type(chunk).__name__}, text={chunk.text!r}")
        if chunk.usage_metadata:
            with_usage += 1
        full = chunk if full is None else full + chunk

    print(f"  всего chunks: {count}, из них без текста: {empty}")
    print(f"  chunks с расходом токенов: {with_usage}")

    if full is None:
        # Защитная ветка: провайдер закрыл поток, не прислав ни одного chunk.
        print("  провайдер не прислал ни одного chunk")
        return

    print(f"  тип собранного: {type(full).__name__}")
    print(f"  длина текста: {len(full.text)} знаков")
    print(f"  расход у собранного: {full.usage_metadata}")


run("ПОТОК ПО УМОЛЧАНИЮ")
print()
run("ПОТОК С stream_usage=True", stream_usage=True)
