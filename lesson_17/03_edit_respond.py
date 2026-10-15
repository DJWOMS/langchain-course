"""Пример 3 урока 17: правка аргументов и ответ вместо инструмента.

Решение edit меняет аргументы вызова до выполнения: человек исправляет
адрес получателя, и письмо уходит по исправленному адресу.

Решение respond подставляет ответ человека вместо работы инструмента.
Тело функции ask_user при этом не выполняется, и список EXECUTED это
показывает.

Решение отправляется только после сверки с тем, что пришло в прерывании.
Агент может попросить не то действие, которого ждёт обработчик, а политика
разрешает не все решения подряд. Обе проверки делает функция resume.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from course_model import build_model

EXECUTED = []


def send_email(to: str, subject: str, body: str) -> str:
    """Отправляет письмо по адресу электронной почты."""
    EXECUTED.append(f"send_email(to={to}, subject={subject})")
    return f"письмо отправлено на {to}"


def ask_user(question: str) -> str:
    """Задаёт уточняющий вопрос заказчику и возвращает его ответ."""
    EXECUTED.append(f"ask_user({question})")
    return "ЭТА СТРОКА ЗНАЧИТ, ЧТО ТЕЛО ИНСТРУМЕНТА ВЫПОЛНИЛОСЬ"


SYSTEM_PROMPT = (
    "Вы помощник менеджера типографии. Если данных не хватает, спрашивайте "
    "инструментом ask_user и не придумывайте их сами. "
    "Отвечайте по-русски, одной фразой."
)

agent = create_agent(
    model=build_model(temperature=0),
    tools=[send_email, ask_user],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "ask_user": {"allowed_decisions": ["respond"]},
            },
        ),
    ],
    checkpointer=InMemorySaver(),
)


def pause_on(thread_id, task):
    """Доводит запуск до паузы и возвращает конфигурацию, запрошенное действие и его политику."""
    EXECUTED.clear()
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config,
        version="v2",
    )
    request = result.interrupts[0].value
    return config, request["action_requests"][0], request["review_configs"][0]


def resume(config, action, review, expected_tool, decision):
    """Отправляет решение, когда действие и тип решения совпали с ожидаемыми.

    Имя действия и список разрешённых решений приходят в самом прерывании.
    По ним и сверяются: агент может запросить другое действие, а решение,
    которого политика не разрешает, поднимет ValueError уже внутри middleware.
    """
    if action["name"] != expected_tool:
        print(f"  ждали действие {expected_tool}, пришло {action['name']}, решение не отправлено")
        return None

    if decision["type"] not in review["allowed_decisions"]:
        print(f"  решение {decision['type']} политикой не разрешено: {review['allowed_decisions']}")
        return None

    return agent.invoke(Command(resume={"decisions": [decision]}), config, version="v2")


def last_text(result):
    """Последнее сообщение запуска одной строкой."""
    return " ".join(result.value["messages"][-1].text.split())


print("РЕШЕНИЕ EDIT: человек правит аргументы до выполнения")

config, action, review = pause_on(
    "letter-1",
    "Отправьте на ivanov@example.com письмо с темой Оплата и текстом "
    "Счёт за август оплачен, спасибо. Все данные для письма есть в задании, "
    "уточнять нечего.",
)
print("  агент предложил:", action["name"], action["args"])
print("  политика вызова:", review["action_name"], review["allowed_decisions"])

edited_args = {**action["args"], "to": "petrov@example.com"}
print("  человек правит: ", edited_args)

result = resume(
    config,
    action,
    review,
    "send_email",
    {"type": "edit", "edited_action": {"name": "send_email", "args": edited_args}},
)

print("  выполнено:      ", EXECUTED)

if result is not None:
    print("  ответ агента:   ", last_text(result))

print()

print("РЕШЕНИЕ RESPOND: человек отвечает вместо инструмента")

config, action, review = pause_on(
    "letter-2",
    "Уточните у заказчика цвет обложки и назовите его в ответе.",
)
print("  агент спрашивает:", action["name"], action["args"])
print("  политика вызова: ", review["action_name"], review["allowed_decisions"])

result = resume(
    config,
    action,
    review,
    "ask_user",
    {"type": "respond", "message": "Тёмно-синий."},
)

print("  выполнено:       ", EXECUTED)

if result is not None:
    tool_messages = [m for m in result.value["messages"] if m.type == "tool"]
    print("  ответ человека дошёл как результат инструмента:", tool_messages[-1].content)
    print("  статус этого сообщения:", tool_messages[-1].status)
    print("  ответ агента:    ", last_text(result))
