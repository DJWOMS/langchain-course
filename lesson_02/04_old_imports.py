"""Пример 4 урока 2: какие импорты из старых статей живы на текущей сборке.

Скрипт ничего не вызывает по сети и денег не тратит. Он берёт список строк
импорта и проверяет каждую: есть ли такой модуль и есть ли в нём такое имя.

Откуда взяты строки:
1) состав пространства имён langchain: oss/python/migrate/langchain-v1.mdx,
   раздел "Namespace", и oss/python/releases/langchain-v1.mdx, раздел
   "Simplified package"
2) что уехало в langchain-classic (цепочки, ретриверы, индексация, hub,
   реэкспорты langchain-community): те же две страницы, раздел
   "langchain-classic"
3) перенос агента из langgraph.prebuilt в langchain.agents:
   oss/python/migrate/langchain-v1.mdx, раздел "Import path"
4) ChatPromptTemplate и StrOutputParser в связке через оператор |:
   oss/python/integrations/chat/amazon_nova.mdx, раздел "Chaining"
5) импорт из langchain_community: oss/python/integrations/document_loaders,
   строка встречается на девяти страницах интеграций
6) langchain.rate_limiters: oss/langchain/models.mdx, раздел "Rate limiting",
   строка "from langchain.rate_limiters import InMemoryRateLimiter". В таблицу
   пространства имён в migrate/langchain-v1.mdx этот адрес не попал
7) langchain.agents.middleware: oss/langchain/middleware/overview.mdx и ещё
   шестнадцать страниц раздела LangChain. В ту же таблицу адрес не попал
"""

import importlib

# (модуль, имя внутри модуля или None, как строка импорта выглядит в статье)
V0_IMPORTS = [
    ("langchain.chains", "LLMChain", "from langchain.chains import LLMChain"),
    ("langchain", "hub", "from langchain import hub"),
    ("langchain.retrievers", "MultiQueryRetriever",
     "from langchain.retrievers import MultiQueryRetriever"),
    ("langchain.indexes", None, "import langchain.indexes"),
    ("langgraph.prebuilt", "create_react_agent",
     "from langgraph.prebuilt import create_react_agent"),
    ("langchain_core.prompts", "ChatPromptTemplate",
     "from langchain_core.prompts import ChatPromptTemplate"),
    ("langchain_core.output_parsers", "StrOutputParser",
     "from langchain_core.output_parsers import StrOutputParser"),
    ("langchain_classic.chains", "LLMChain",
     "from langchain_classic.chains import LLMChain"),
    ("langchain_community.document_loaders", "TextLoader",
     "from langchain_community.document_loaders import TextLoader"),
]

V1_IMPORTS = [
    ("langchain.agents", "create_agent",
     "from langchain.agents import create_agent"),
    ("langchain.chat_models", "init_chat_model",
     "from langchain.chat_models import init_chat_model"),
    ("langchain.messages", "HumanMessage",
     "from langchain.messages import HumanMessage"),
    ("langchain.tools", "tool", "from langchain.tools import tool"),
    ("langchain.embeddings", "init_embeddings",
     "from langchain.embeddings import init_embeddings"),
    ("langchain.rate_limiters", "InMemoryRateLimiter",
     "from langchain.rate_limiters import InMemoryRateLimiter"),
    ("langchain.agents.middleware", None, "import langchain.agents.middleware"),
]


def check(module_name, attribute):
    """Возвращает пару: получилось или нет, и причина отказа."""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"

    if attribute is None:
        return True, ""

    if not hasattr(module, attribute):
        return False, f"AttributeError: в модуле {module_name} нет имени {attribute}"

    return True, ""


def report(title, imports):
    print(title)
    for module_name, attribute, line in imports:
        works, reason = check(module_name, attribute)
        status = "ЕСТЬ" if works else "НЕТ"
        print(f"{status:<5} {line}")
        if reason:
            print(f"{reason}")
    print()


report("СТРОКИ ИЗ СТАТЕЙ ПРО ВЕРСИЮ 0.x", V0_IMPORTS)
report("СТРОКИ ТЕКУЩЕЙ ВЕРСИИ", V1_IMPORTS)
