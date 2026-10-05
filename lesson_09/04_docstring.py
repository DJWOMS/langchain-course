"""Пример 4. Docstring это часть промпта: четыре варианта одного инструмента.

Модель здесь не вызывается. Пример печатает то, что уйдёт в модель вместе с запросом:
описание инструмента и схему его аргументов.
"""

from langchain.tools import tool


@tool
def search_orders_v1(query: str, limit: int = 5) -> str:
    """Поиск заказов."""
    return f"{limit} заказов по запросу {query}"


@tool
def search_orders_v2(query: str, limit: int = 5) -> str:
    """Найти заказы покупателя по строке поиска.

    Args:
        query: номер заказа, имя покупателя или название товара
        limit: сколько заказов вернуть, по умолчанию 5
    """
    return f"{limit} заказов по запросу {query}"


@tool(parse_docstring=True)
def search_orders_v3(query: str, limit: int = 5) -> str:
    """Найти заказы покупателя по строке поиска.

    Args:
        query: номер заказа, имя покупателя или название товара
        limit: сколько заказов вернуть, по умолчанию 5
    """
    return f"{limit} заказов по запросу {query}"


@tool(
    "order_search",
    description="Искать заказы. Вызывайте, только когда в вопросе есть номер заказа.",
)
def search_orders_v4(query: str, limit: int = 5) -> str:
    """Этот текст в модель не уйдёт: description его перекрывает."""
    return f"{limit} заказов по запросу {query}"


def show(title, tool_object):
    print("=" * 70)
    print(title)
    print("имя:", tool_object.name)
    print("описание:", repr(tool_object.description))
    schema = tool_object.tool_call_schema.model_json_schema()
    print("описания аргументов:")
    for name, field in schema["properties"].items():
        print(f"  {name}: {field.get('description', '<нет>')}")
    print()


show("1. Короткий docstring", search_orders_v1)
show("2. Docstring с блоком Args, parse_docstring не задан", search_orders_v2)
show("3. То же самое с parse_docstring=True", search_orders_v3)
show("4. Своё имя и своё описание в декораторе", search_orders_v4)
