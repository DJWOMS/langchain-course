"""Пример 1 урока 15: карта точек встраивания у встроенных middleware.

Модель здесь не вызывается. Пример спрашивает у самих классов, какие хуки они
переопределяют, и печатает это таблицей.

Приём взят из сборки агента: фабрика решает, подключать middleware к точке или нет,
сравнением метода класса с методом базового AgentMiddleware. Асинхронный
вариант хука считается тем же хуком, поэтому проверяются оба имени.
"""

from pathlib import Path

from langchain.agents.middleware import (
    AgentMiddleware,
    ContextEditingMiddleware,
    FilesystemFileSearchMiddleware,
    HumanInTheLoopMiddleware,
    LLMToolEmulator,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)

# Шесть точек встраивания, по колонке таблицы на каждую.
HOOKS = (
    "before_agent",
    "before_model",
    "wrap_model_call",
    "wrap_tool_call",
    "after_model",
    "after_agent",
)

SHORT = {
    "before_agent": "b_ag",
    "before_model": "b_md",
    "wrap_model_call": "w_md",
    "wrap_tool_call": "w_tl",
    "after_model": "a_md",
    "after_agent": "a_ag",
}

LAYERS = (
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    LLMToolEmulator,
    FilesystemFileSearchMiddleware,
    PIIMiddleware,
    HumanInTheLoopMiddleware,
)


def implements(layer, hook):
    """Переопределяет ли middleware этот хук, в синхронном или асинхронном виде."""
    async_hook = "a" + hook
    sync_own = getattr(layer, hook) is not getattr(AgentMiddleware, hook)
    async_own = getattr(layer, async_hook) is not getattr(AgentMiddleware, async_hook)
    return sync_own or async_own


header = "".join(f"{SHORT[hook]:>6}" for hook in HOOKS)
print(f"{'middleware':<32}{header}")

for layer in LAYERS:
    marks = "".join(f"{'X' if implements(layer, hook) else '.':>6}" for hook in HOOKS)
    print(f"{layer.__name__:<32}{marks}")

print()
print("узел графа появляется только у before_* и after_*,")
print("хуки wrap_* оборачивают вызов внутри уже существующих узлов")

# Вторая работа middleware: принести агенту свои инструменты. Это свойство экземпляра,
# а не класса, поэтому здесь middleware создаются.
print()
print("инструменты, которые middleware добавляет агенту:")

WITH_TOOLS = (
    TodoListMiddleware(),
    FilesystemFileSearchMiddleware(root_path=str(Path(__file__).parent)),
    ToolRetryMiddleware(),
)

for layer in WITH_TOOLS:
    names = [tool.name for tool in layer.tools] or ["нет"]
    print(f"  {layer.name:<32}{', '.join(names)}")
