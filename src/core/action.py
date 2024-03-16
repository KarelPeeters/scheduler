from dataclasses import dataclass
from typing import Union

from core.problem import OperationNode, OperationAllocation, Channel, Memory


@dataclass(frozen=True, eq=True)
class ActionWait:
    time_start: float
    time_end: float

    def __str__(self):
        return f"ActionWait(time={self.time_start}..{self.time_end})"


@dataclass(frozen=True, eq=True)
class ActionCore:
    time_start: float

    node: OperationNode
    alloc: OperationAllocation

    @property
    def time_end(self):
        return self.time_start + self.alloc.time

    def __str__(self):
        return f"ActionCore(time={self.time_start}..{self.time_end}, node={self.node.id}, alloc={self.alloc.id}, core={self.alloc.core.id})"


@dataclass(frozen=True, eq=True)
class ActionChannel:
    time_start: float

    channel: Channel
    source: Memory
    dest: Memory
    value: OperationNode

    @property
    def total_latency(self):
        return self.channel.latency + self.value.size_bits * self.channel.time_per_bit

    @property
    def time_end(self):
        return self.time_start + self.total_latency

    def __str__(self):
        return f"ActionChannel(time={self.time_start}..{self.time_end}, channel={self.channel.id}, value={self.value.id}, source={self.source.id}, dest={self.dest.id})"


RealAction = Union[ActionCore, ActionChannel]
Action = Union[ActionWait, ActionCore, ActionChannel]
