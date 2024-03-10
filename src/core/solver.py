from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, DefaultDict

from core.frontier import ParetoFrontier
from core.problem import Problem, OperationNode, Memory


# TODO add latency and energy bounds. Some examples:
#  * energy: run every remaining operation on the lowest energy core (including energy for the transfers for this path)
#  * latency: run every remaining on the fastest possible core (+concurrency), ignoring any transfer latencies?

# TODO what to do about dropping old values?
#  * drop values that will never be used again immediately
#  * only allow dropping values that will still be used if they're duplicated somewhere else

# TODO try some move ordering, should speed things up by a lot

# TODO pareto: should scheduling be started or done? both?
# TODO pareto: do we also want to include peak memory in each memory block here?

# TODO make the pareto-front transposition aware (between hardware and graph symmetries)

def schedule(problem: Problem):
    state = RecurseState(problem)
    recurse(state)


@dataclass
class Claim:
    value: OperationNode
    mem: Memory

    read: bool
    write: bool


@dataclass(eq=False)
class CoreRunning:
    time_done: float
    node: OperationNode
    claims: List[Claim]

    def __str__(self):
        return f"CoreRunning(time_done={self.time_done}, node={self.node.id}, claims={self.claims})"


@dataclass(eq=False)
class ChannelRunning:
    time_done: float
    value: OperationNode
    claims: List[Claim]

    def __str__(self):
        return f"ChannelRunning(time_done={self.time_done}, value={self.value.id}, claims={self.claims})"


@dataclass
class ClaimCounter:
    reads: int
    writes: int


class RecurseState:
    problem: Problem
    frontier: ParetoFrontier

    curr_time: float
    curr_energy: float

    unstarted_nodes = List[OperationNode]
    value_remaining_unstarted_uses: Dict[OperationNode, int]

    core_state: List[Optional[CoreRunning]]
    channel_state: List[Optional[ChannelRunning]]
    memory_contents: Dict[Memory, Set[OperationNode]]
    active_claims: DefaultDict[Tuple[OperationNode, Memory], ClaimCounter]

    def __init__(self, problem: Problem):
        # basics
        self.problem = problem
        hw = problem.hardware
        graph = problem.graph

        # frontier
        real_nodes = [n for n in graph.nodes if n not in graph.inputs]

        # The pareto key consists of:
        # * scheduling progress: [done] per real node (higher is better)
        # * core availability: [-next_free_time] per core and channel (lower is better)
        # * costs: (-time), (-energy) (lower is better)
        # TODO include "value in memory" availability as value
        # TODO add a more complicated "dominated proper", eg. we can obvious trade memory availability for time+energy

        self.frontier = ParetoFrontier(len(real_nodes) + len(hw.cores) + len(hw.channels) + 2)

        # current
        self.curr_time = 0.0
        self.curr_energy = 0.0

        # unstarted nodes and values that have remaining uses
        self.unstarted_nodes = [n for n in graph.nodes if n not in graph.outputs]

        self.value_remaining_unstarted_uses = {n: 0 for n in graph.nodes}
        for node in graph.nodes:
            for input in node.inputs:
                self.value_remaining_unstarted_uses[input] += 1
        for output in graph.outputs:
            self.value_remaining_unstarted_uses[output] = 1

        # states
        self.core_state = [None] * len(hw.cores)
        self.channel_state = [None] * len(hw.channels)

        # memory contents and claims
        self.memory_contents = {m: set() for m in hw.memories}
        for node, memory in problem.placement_inputs.items():
            self.memory_contents[memory].add(node)
        self.active_claims = defaultdict(lambda: ClaimCounter(0, 0))

    def print(self):
        hw = self.problem.hardware

        print("RecurseState(")
        print(f"  curr_time: {self.curr_time}, curr_energy: {self.curr_energy}")
        print(f"  unstarted_nodes: {[n.id for n in self.unstarted_nodes]}")
        print(
            f"  value_remaining_unstarted_uses: {({n.id: c for n, c in self.value_remaining_unstarted_uses.items()})}")

        print(f"  core_state: [")
        for i in range(len(hw.cores)):
            print(f"    core {hw.cores[i].id}: {self.core_state[i]}")
        print(f"  ]")

        print(f"  channel_state: [")
        for i in range(len(hw.channels)):
            print(f"    channel {hw.channels[i].id}: {self.channel_state[i]}")
        print(f"  ]")

        print(f"  memory_content: [")
        for i in range(len(hw.memories)):
            print(f"    memory {hw.memories[i].id}: {({n.id for n in self.memory_contents[hw.memories[i]]})}")
        print(f"  ]")

        print("  active_claims: {")
        for (node, memory), counter in self.active_claims.items():
            print(f"    ({node.id}, {memory.id}): {counter}")
        print(f"  }}")
        print(")")

    def claim_all_if_possible(self, claims: List[Claim]):
        # TODO define output here? that's a bit sketchy
        # TODO
        pass

    def release(self, claim: Claim):
        # TODO
        pass


# TODO find a working solution first to avoid copying stuff around forever without making any progress?
# TODO have a separate frontier for only energy and latency for fully finished schedules?

# TODO repeatedly check frontier between different action attempts? will we ever get new hits?

def recurse(state: RecurseState):
    # Always:
    # * drop all values that will never be used again from memories
    # * check if the problem is done, report result
    # Possible actions:
    # * start an operation (if the inputs are already available in the connected memories!):
    #   * on any of the currently idle cores
    #   * wait for any of the currently running cores, then start it on that one
    # * start a transfer:
    #   * on any of the currently idle channels
    #   * wait for any of the currently running channels to become idle, then start it on that one
    # * drop a value

    state.print()
    hardware = state.problem.hardware
    graph = state.problem.graph

    # TODO
    pass
