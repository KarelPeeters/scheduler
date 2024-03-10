from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

from graphviz import Digraph


# General design notes:
# * avoid users having to subclass at any cost
# * ids are optional and only there for debugging, they don't have semantics


def constructor(cls):
    return dataclass(
        init=True,
        eq=False,
        order=False,
        unsafe_hash=False,
        frozen=True,
        match_args=False,
        kw_only=False
    )(cls)


@constructor
class Memory:
    id: Optional[str]
    size_bits: Optional[int]


@constructor
class Channel:
    id: Optional[str]

    memory_a: Memory
    memory_b: Memory
    dir_a_to_b: bool
    dir_b_to_a: bool

    latency: float
    time_per_bit: float
    energy_per_bit: float


@constructor
class Core:
    id: Optional[str]
    connected_memories: List[Memory]


def dot_table(title: str, rows: List[Tuple[str, str]]) -> str:
    table = f"<TABLE BORDER=\"0\" COLUMNS=\"*\" ROWS=\"*\"><TR><TD colspan=\"2\"><B>{title}</B></TD></TR>"
    for key, value in rows:
        table += f"<TR><TD>{key}</TD><TD>{value}</TD></TR>"
    table += "</TABLE>"
    return table


def dot_html(content: str) -> str:
    return f"<{content}>"


@constructor
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
        if self.id is not None:
            dot.attr(label=f"<<B>{self.id}</B>>")
            dot.attr(labelloc="t")

        for i, mem in enumerate(self.memories):
            size = "inf" if mem.size_bits is None else str(mem.size_bits)
            label = dot_html(dot_table("Memory", [("id", mem.id), ("size_bits", size)]))
            dot.node(f"mem-{i}", label, shape="box",
                     color="blue")

        for i, core in enumerate(self.cores):
            label = dot_html(dot_table("Core", [("id", core.id)]))
            dot.node(f"core-{i}", label, color="green")
            for j, mem in enumerate(core.connected_memories):
                head = f"core-{i}"
                tail = f"mem-{self.memories.index(mem)}"
                dot.edge(tail, head, headlabel=str(j), dir="none")

        for i, chan in enumerate(self.channels):
            rows = [
                ("id", chan.id),
                ("latency", str(chan.latency)),
                ("time_per_bit", str(chan.time_per_bit)),
                ("energy_per_bit", str(chan.energy_per_bit))
            ]
            label = dot_html(dot_table("Channel", rows))
            mid = f"chan-{i}"
            dot.node(mid, label, shape="box", style="rounded", color="orange")

            head = f"mem-{self.memories.index(chan.memory_a)}"
            tail = f"mem-{self.memories.index(chan.memory_b)}"

            dirs = {(False, False): "none", (False, True): "back", (True, False): "forward", (True, True): "both"}
            dir = dirs[(chan.dir_a_to_b, chan.dir_b_to_a)]

            dot.edge(tail, mid, dir=dir)
            dot.edge(mid, head, dir=dir)

        return dot


# TODO dedicated slice, concat, transpose, pad ... operations that don't require computation,
#   but only specific memory transfers
# TODO dedicated input node
# TODO change to a mutable graph representation that can also deal with accumulation
@constructor
class OperationNode:
    id: Optional[str]
    size_bits: int
    # no inputs means this is an input node
    inputs: List['OperationNode']


# TODO split operations and values to support multiple outputs, ...
@constructor
class OperationGraph:
    id: Optional[str]
    nodes: List[OperationNode]

    inputs: List[OperationNode]
    outputs: List[OperationNode]

    def assert_valid(self):
        for node in self.nodes:
            for input in node.inputs:
                assert input in self.nodes
        assert len(set(self.inputs)) == len(self.inputs)
        assert len(set(self.outputs)) == len(self.outputs)
        for input in self.inputs:
            assert input in self.nodes
        for output in self.outputs:
            assert output in self.nodes

    def to_graphviz(self):
        dot = Digraph()

        if self.id is not None:
            dot.attr(label=f"<<B>{self.id}</B>>")
            dot.attr(labelloc="t")

        for i, node in enumerate(self.nodes):
            rows = [
                ("id", node.id),
                ("size_bits", str(node.size_bits)),
            ]

            input = [i for i, n in enumerate(self.inputs) if n == node]
            output = [i for i, n in enumerate(self.outputs) if n == node]
            if input:
                rows.append(("input", str(input)))
            if output:
                rows.append(("output", str(output)))

            label = dot_html(dot_table("Node", rows))
            dot.node(f"node-{i}", label, shape="box", color="blue")
            for j, input in enumerate(node.inputs):
                head = f"node-{self.nodes.index(input)}"
                tail = f"node-{i}"
                dot.edge(head, tail, headlabel=str(j))

        return dot


@constructor
class OperationAllocation:
    core: Core

    input_memories: List[int]
    output_memory: int

    time: float
    energy: float


@constructor
class Problem:
    id: Optional[str]
    hardware: Hardware
    operationGraph: OperationGraph

    possible_allocations: Dict[OperationNode, Set[OperationAllocation]]

    initial_placement: Dict[OperationNode, Tuple[Memory, Optional[int]]]
    final_placement: Dict[OperationNode, Tuple[Memory, Optional[int]]]
