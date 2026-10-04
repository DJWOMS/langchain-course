"""Свой преобразователь потока для примеров 6 и 7 урока 8.

Преобразователь смотрит канал tools и собирает две проекции:

1) tool_activity, именованный канал. Каждая запись одновременно уходит в общий
   поток событий отдельным событием custom:tool_activity

2) tool_totals, безымянный канал. Туда в finalize() кладётся один словарь со
   счётчиком событий за прогон

Имя инструмента приходит только в событии tool-started, поэтому оно
запоминается по tool_call_id и подставляется в остальные три события.
"""

from collections import Counter
from typing import Any, TypedDict

from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer


class Activity(TypedDict):
    """Одна запись проекции tool_activity."""

    tool: str
    event: str


class ToolActivityTransformer(StreamTransformer):
    """Проекция "что происходит с инструментами" поверх канала tools."""

    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.activity: StreamChannel[Activity] = StreamChannel("tool_activity")
        self.totals: StreamChannel[dict[str, int]] = StreamChannel()
        self.names: dict[str, str] = {}
        self.counts: Counter[str] = Counter()

    def init(self) -> dict[str, Any]:
        """Ключи этого словаря станут ключами stream.extensions."""
        return {"tool_activity": self.activity, "tool_totals": self.totals}

    def process(self, event: ProtocolEvent) -> bool:
        """Смотрит каждое событие прогона и отбирает свои по имени канала."""
        if event["method"] != "tools":
            return True

        data = event["params"]["data"]
        if not isinstance(data, dict):
            return True

        kind = data.get("event")
        call_id = data.get("tool_call_id")
        if kind is None or call_id is None:
            return True

        if kind == "tool-started":
            self.names[call_id] = data.get("tool_name", "")

        self.counts[kind] += 1
        self.activity.push({"tool": self.names.get(call_id, ""), "event": kind})

        # True оставляет событие в общем потоке. False его гасит.
        return True

    def finalize(self) -> None:
        """Вызывается один раз в конце прогона, до закрытия каналов."""
        self.totals.push(dict(self.counts))
