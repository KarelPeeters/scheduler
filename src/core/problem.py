from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

from graphviz import Digraph


# General design notes:
# * avoid users having to subclass at any cost
# * ids are optional and only there for debugging, they don't have semantics


@dataclass
class Memory:
    id: Optional[str]
    size: Optional[int]


@dataclass
class Channel:
    id: Optional[str]

    memory_a: Memory
    memory_b: Memory
    dir_a_to_b: bool
    dir_b_to_a: bool

    latency: float
    time_per_bit: float
    energy_per_bit: float


@dataclass
class Core:
    id: Optional[str]
    connected_memories: List[Memory]


@dataclass
class Hardware:
    id: Optional[str]
    cores: List[Core]
    memories: List[Memory]
    channels: List[Channel]

    def assert_valid(self):
        for core in self.cores:
            for mem in core.connected_memories:
                assert mem in self.memories
        for chan in self.channels:
            assert chan.memory_a in self.memories
            assert chan.memory_b in self.memories

    def to_graphviz(self) -> Digraph:
        dot = Digraph()

        for i, mem in enumerate(self.memories):
            dot.node(f"mem-{i}", mem.id, shape="box", color="blue")

        for i, core in enumerate(self.cores):
            dot.node(f"core-{i}", core.id, color="green")
            for j, mem in enumerate(core.connected_memories):
                head = f"core-{i}"
                tail = f"mem-{self.memories.index(mem)}"
                dot.edge(tail, head, headlabel=str(j), dir="none")

        for i, chan in enumerate(self.channels):
            head = f"mem-{self.memories.index(chan.memory_a)}"
            tail = f"mem-{self.memories.index(chan.memory_b)}"

            dirs = {(False, False): "none", (False, True): "back", (True, False): "forward", (True, True): "both"}
            dir = dirs[(chan.dir_a_to_b, chan.dir_b_to_a)]
            dot.edge(tail, head, label=chan.id, dir=dir)

        return dot


# TODO dedicated slice, concat, transpose, pad ... operations that don't require computation,
#   but only specific memory transfers
# TODO dedicated input node
# TODO change to a mutable graph representation that can also deal with accumulation
@dataclass
class OperationNode:
    id: Optional[str]
    size_bits: int
    # no inputs means this is an input node
    inputs: List['OperationNode']


@dataclass
class OperationGraph:
    id: Optional[str]
    # inputs are nodes with no
    nodes: List[OperationNode]
    outputs: List[OperationNode]


@dataclass
class OperationAllocation:
    core: Core

    input_memories: List[int]
    output_memory: int

    time: float
    energy: float


@dataclass
class Problem:
    id: Optional[str]
    hardware: Hardware
    operationGraph: OperationGraph

    possible_allocations: Dict[OperationNode, Set[OperationAllocation]]

    initial_placement: Dict[OperationNode, Tuple[Memory, Optional[int]]]
    final_placement: Dict[OperationNode, Tuple[Memory, Optional[int]]]
