from dataclasses import dataclass
from typing import Union

from core.problem import Core, Channel, Memory, OperationNode


@dataclass(frozen=True, eq=True)
class EventFreeCore:
    core: Core


@dataclass(frozen=True, eq=True)
class EventFreeChannel:
    channel: Channel


@dataclass(frozen=True, eq=True)
class EventValueAvailable:
    mem: Memory
    value: OperationNode


@dataclass(frozen=True, eq=True)
class EventReadReleased:
    mem: Memory
    value: OperationNode


Event = Union[EventFreeCore, EventFreeChannel, EventValueAvailable, EventReadReleased]

# TODO add dropped value event (or for deduplication purposes: "available mem size increased"
#    (careful: make sure before and after get merged properly)
