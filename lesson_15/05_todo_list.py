"""Пример 5 урока 15: список задач, который агент ведёт сам.

TodoListMiddleware добавляет агенту инструмент write_todos и кусок системного
промпта про то, как им пользоваться. Поле todos хранится в состоянии агента.

Задача требует пройти по всем модулям репозитория и для каждого узнать три
вещи по отдельности, а потом свести это в один вывод. Шаги внутри модуля
зависят друг от друга: владельца и тесты спрашивают про уже найденный модуль.
Модулей несколько, поэтому такую задачу удобнее вести по списку.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from course_model import build_model

MODULES = {
    "billing": {"lines": 4120, "owner": "Северов", "tests": 12},
    "auth": {"lines": 890, "owner": "Липатова", "tests": 41},
    "reports": {"lines": 7305, "owner": "владелец не назначен", "tests": 0},
}


def list_modules() -> str:
    """Возвращает список модулей репозитория."""
    return ", ".join(sorted(MODULES))


def module_size(name: str) -> str:
    """Возвращает число строк кода в модуле."""
    if name not in MODULES:
        return f"Модуля {name} в репозитории нет."
    return f"{name}: {MODULES[name]['lines']} строк."


def module_owner(name: str) -> str:
    """Возвращает владельца модуля и число тестов на него."""
    if name not in MODULES:
        return f"Модуля {name} в репозитории нет."
    info = MODULES[name]
    return f"{name}: владелец {info['owner']}, тестов {info['tests']}."


SYSTEM_PROMPT = (
    "Вы разбираете чужой репозиторий. Отвечайте по-русски. "
    "Работу над многошаговой задачей начинайте с плана."
)

QUESTION = (
    "Проверьте все модули репозитория по очереди: для каждого узнайте размер, "
    "владельца и число тестов. Отдельно выпишите модули без владельца и "
    "модули, где тестов меньше десяти. В конце скажите, какой модуль чинить "
    "первым и почему."
)

agent = create_agent(
    model=build_model(temperature=0, max_tokens=768),
    tools=[list_modules, module_size, module_owner],
    system_prompt=SYSTEM_PROMPT,
    middleware=[TodoListMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

tool_calls = [
    call["name"]
    for message in result["messages"]
    if message.type == "ai"
    for call in message.tool_calls
]
print("вызовы инструментов:", ", ".join(tool_calls) or "их не было")

print()
print("список задач в конце запуска:")
for item in result.get("todos") or []:
    print(f"  [{item['status']:<11}] {item['content']}")

print()
print("ответ:", result["messages"][-1].text.replace("\n", " "))
