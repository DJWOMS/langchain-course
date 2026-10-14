"""Пример 4 урока 14: что инструмент отдаёт модели и что он кладёт в состояние.

Два агента решают одну задачу по одному и тому же отчёту. Первый получает отчёт
целиком текстом инструмента, и этот текст уходит в каждый следующий запрос.
Второй получает выжимку, а полный текст уходит в состояние, откуда его достаёт
второй инструмент по ключевому слову.

Печатается число вызовов модели и сумма входных токенов за весь запуск.
"""

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from course_model import build_model

REPORT = (
    "Отчёт по складу Казань за неделю.\n"
    "Приёмка: принято 412 паллет, из них 38 по электронной очереди с опозданием.\n"
    "Отгрузка: отгружено 389 паллет, средняя загрузка машины 84 процента.\n"
    "Брак: выявлено 17 мест, из них бой 9, подмочка 5, пересорт 3.\n"
    "Брак по перевозчикам: 11 мест у перевозчика М-1, 6 мест у перевозчика М-4.\n"
    "Возвраты: оформлено 23 возврата, средний срок обработки 1,8 дня.\n"
    "Инвентаризация: пересчитаны зоны A и B, расхождение 0,4 процента.\n"
    "Персонал: средняя численность смены 14 человек, две смены неполные.\n"
    "Техника: один погрузчик в ремонте с вторника, замена не выделена.\n"
    "Температурная зона: два срыва режима, оба короче пятнадцати минут.\n"
    "Документы: в учётную систему с опозданием попали 6 актов приёмки.\n"
    "Планы: на следующую неделю заявлено 430 паллет приёмки и 400 отгрузки.\n"
)

QUESTION = "Откройте отчёт по складу Казань и скажите, что в нём написано про брак."

PROMPT = (
    "Вы помощник руководителя склада. Отвечайте по-русски, одной-двумя фразами. "
    "Данные берите инструментами, по памяти не отвечайте."
)


class DeskState(AgentState):
    """Состояние агента: к messages добавлено место для загруженных документов."""

    documents: dict[str, str]


def open_report(name: str) -> str:
    """Возвращает полный текст отчёта по складу указанного города."""
    if name != "Казань":
        return f"Отчёта по складу {name} нет."

    return REPORT


@tool
def load_report(name: str, runtime: ToolRuntime[None, DeskState]) -> Command:
    """Загружает отчёт по складу города в рабочую память и отдаёт его состав."""
    if name != "Казань":
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Отчёта по складу {name} нет.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    lines = REPORT.splitlines()
    topics = ", ".join(line.split(":")[0] for line in lines[1:])
    digest = (
        f"Отчёт по складу {name} загружен: {len(lines)} строк, "
        f"{len(REPORT)} знаков. Разделы: {topics}. Чтобы прочитать нужный "
        f"раздел, вызовите find_in_report с ключевым словом."
    )

    return Command(
        update={
            "documents": {**runtime.state.get("documents", {}), name: REPORT},
            "messages": [
                ToolMessage(content=digest, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def find_in_report(
    name: str, keyword: str, runtime: ToolRuntime[None, DeskState]
) -> str:
    """Возвращает строки загруженного отчёта, в которых встречается слово."""
    text = runtime.state.get("documents", {}).get(name)

    if text is None:
        return f"Отчёт по складу {name} ещё не загружен."

    found = [line for line in text.splitlines() if keyword.lower() in line.lower()]

    if not found:
        return f"Слово '{keyword}' в отчёте по складу {name} не встречается."

    return "\n".join(found)


def spent(messages) -> int:
    """Складывает входные токены по всем ответам модели в истории запуска."""
    total = 0

    for message in messages:
        # Расход возвращает не каждый провайдер, поэтому пустой словарь как запасной путь.
        usage = getattr(message, "usage_metadata", None) or {}
        total += usage.get("input_tokens", 0)

    return total


def model_calls(messages) -> int:
    """Считает ответы модели в истории запуска."""
    return sum(1 for message in messages if message.type == "ai")


direct = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[open_report],
    system_prompt=PROMPT,
)

filtered = create_agent(
    model=build_model(temperature=0, max_tokens=384),
    tools=[load_report, find_in_report],
    system_prompt=PROMPT,
    state_schema=DeskState,
)

print(f"{'агент':<22}  {'вызовов':>7}  {'вход всего':>10}")

for label, agent in (("отчёт целиком", direct), ("выжимка и состояние", filtered)):
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    messages = result["messages"]
    answer = messages[-1].text.replace("\n", " ")
    print(f"{label:<22}  {model_calls(messages):>7}  {spent(messages):>10}")
    print(f"{'':<22}  ОТВЕТ: {answer}")

print()
print("В СОСТОЯНИИ ВТОРОГО АГЕНТА ОСТАЛОСЬ:")
for name, text in result.get("documents", {}).items():
    print(f"  {name}: {len(text)} знаков")
