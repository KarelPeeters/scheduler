import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, DefaultDict, Union

from core.frontier import ParetoFrontier
from core.problem import Problem, OperationNode, Memory, Channel, Core


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
    # The pareto key consists of:
    # * scheduling progress: [done] per real node (higher is better)
    # * core availability: [-next_free_time] per core and channel (lower is better)
    # * costs: (-time), (-energy) (lower is better)
    # TODO include "value in memory" availability as value
    # TODO add a more complicated "dominated proper", eg. we can obvious trade memory availability for time+energy

    real_nodes = [n for n in problem.graph.nodes if n not in problem.graph.inputs]
    frontier = ParetoFrontier(len(real_nodes) + len(problem.hardware.cores) + len(problem.hardware.channels) + 2)

    state = RecurseState.initial(problem)
    recurse(problem, frontier, state)


# TODO remove?
@dataclass(frozen=True, eq=False)
class Claim:
    value: OperationNode
    mem: Memory

    read: bool
    write: bool


@dataclass(eq=False)
class ClaimCounter:
    reads: int
    writes: int

    def clone(self):
        return ClaimCounter(self.reads, self.writes)


@dataclass(frozen=True, eq=False)
class WaitAction:
    time_until: float


@dataclass(frozen=True, eq=False)
class CoreAction:
    # TODO
    pass


@dataclass(frozen=True, eq=False)
class ChannelAction:
    channel: Channel
    source: Memory
    dest: Memory
    value: OperationNode


Action = Union[ChannelAction, CoreAction, WaitAction]


@dataclass(frozen=True, eq=False)
class CoreRunning:
    time_done: float
    node: OperationNode
    claims: List[Claim]

    def __str__(self):
        return f"CoreRunning(time_done={self.time_done}, node={self.node.id}, claims={self.claims})"


@dataclass(frozen=True, eq=False)
class ChannelRunning:
    time_done: float
    action: ChannelAction

    def __str__(self):
        return f"ChannelRunning(time_done={self.time_done}, action={self.action})"


@dataclass(eq=False)
class RecurseState:
    curr_time: float
    curr_energy: float

    unstarted_nodes: List[OperationNode]
    value_remaining_unstarted_uses: Dict[OperationNode, int]

    core_state: Dict[Core, Optional[CoreRunning]]
    channel_state: Dict[Channel, Optional[ChannelRunning]]
    memory_contents: Dict[Memory, Dict[OperationNode, bool]]  # mem -> value -> ready
    active_claims: DefaultDict[Tuple[OperationNode, Memory], ClaimCounter]

    @staticmethod
    def initial(problem: Problem):
        # basics
        hw = problem.hardware
        graph = problem.graph

        # current
        curr_time = 0.0
        curr_energy = 0.0

        # unstarted nodes and values that have remaining uses
        unstarted_nodes = [n for n in graph.nodes if n not in graph.inputs]

        value_remaining_unstarted_uses = {n: 0 for n in graph.nodes}
        for node in graph.nodes:
            for input in node.inputs:
                value_remaining_unstarted_uses[input] += 1
        for output in graph.outputs:
            value_remaining_unstarted_uses[output] = 1

        # memory contents
        memory_contents = {m: {} for m in hw.memories}
        for node, memory in problem.placement_inputs.items():
            memory_contents[memory][node] = True

        return RecurseState(
            curr_time=0.0,
            curr_energy=0.0,

            unstarted_nodes=unstarted_nodes,
            value_remaining_unstarted_uses=value_remaining_unstarted_uses,

            core_state={c: None for c in hw.cores},
            channel_state={c: None for c in hw.channels},
            memory_contents=memory_contents,
            active_claims=(defaultdict(lambda: ClaimCounter(0, 0))),
        )

    def clone(self):
        """ deep clone """
        active_claims = defaultdict(lambda: ClaimCounter(0, 0))
        for key, counter in self.active_claims.items():
            active_claims[key] = counter.clone()

        return RecurseState(
            curr_time=self.curr_time,
            curr_energy=self.curr_energy,

            unstarted_nodes=self.unstarted_nodes.copy(),
            value_remaining_unstarted_uses=self.value_remaining_unstarted_uses.copy(),

            core_state=self.core_state.copy(),
            channel_state=self.channel_state.copy(),
            memory_contents={m: c.copy() for m, c in self.memory_contents.items()},
            active_claims=active_claims,
        )

    def print(self, problem: Problem):
        hw = problem.hardware

        print("RecurseState(")
        print(f"  curr_time: {self.curr_time}, curr_energy: {self.curr_energy}")
        print(f"  unstarted_nodes: {[n.id for n in self.unstarted_nodes]}")
        print(
            f"  value_remaining_unstarted_uses: {({n.id: c for n, c in self.value_remaining_unstarted_uses.items()})}")

        print(f"  core_state: [")
        for c in hw.cores:
            print(f"    core {c.id}: {self.core_state[c]}")
        print(f"  ]")

        print(f"  channel_state: [")
        for c in hw.channels:
            print(f"    channel {c.id}: {self.channel_state[c]}")
        print(f"  ]")

        print(f"  memory_content: [")
        for m in hw.memories:
            print(f"    memory {m.id}: {({n.id: b for n, b in self.memory_contents[m].items()})}")
        print(f"  ]")

        print("  active_claims: {")
        for (node, memory), counter in self.active_claims.items():
            print(f"    ({node.id}, {memory.id}): {counter}")
        print(f"  }}")
        print(")")

    def clone_do_action(self, action: Action):
        next_state = self.clone()
        next_state.do_action(action)
        return next_state

    def do_action(self, action: Action):
        print(f"Running action: {action}")

        if isinstance(action, WaitAction):
            self.do_wait_action(action)
        elif isinstance(action, CoreAction):
            self.do_core_action(action)
        elif isinstance(action, ChannelAction):
            self.do_channel_action(action)
        else:
            assert False, f"unknown action type: {action}"

    def do_wait_action(self, action: WaitAction):
        assert action.time_until > self.curr_time
        self.curr_time = action.time_until

        for core, state in self.core_state.items():
            if state is None or state.time_done > self.curr_time:
                continue
            assert False, "TODO"  # TODO

        for channel, state in self.channel_state.items():
            if state is None or state.time_done > self.curr_time:
                continue

            assert self.memory_contents[state.action.dest][state.action.value] is False
            self.memory_contents[state.action.dest][state.action.value] = True

            self.active_claims[(state.action.value, state.action.source)].reads -= 1
            self.active_claims[(state.action.value, state.action.dest)].writes -= 1

            self.channel_state[channel] = None

    def do_core_action(self, action: CoreAction):
        assert False, "TODO"  # TODO

    def do_channel_action(self, action: ChannelAction):
        assert action.value not in self.memory_contents[action.dest]
        self.memory_contents[action.dest][action.value] = False

        self.active_claims[(action.value, action.source)].reads += 1
        self.active_claims[(action.value, action.dest)].writes += 1

        time_transfer = action.channel.latency + action.value.size_bits * action.channel.time_per_bit

        assert self.channel_state[action.channel] is None
        self.channel_state[action.channel] = ChannelRunning(
            time_done=self.curr_time + time_transfer,
            action=action,
        )

    def undo_action(self, _: Action):
        # TODO implement this and stop using clone for a speedup
        assert False, "TODO"


# TODO find a working solution first to avoid copying stuff around forever without making any progress?
# TODO have a separate frontier for only energy and latency for fully finished schedules?

# TODO repeatedly check frontier between different action attempts? will we ever get new hits?
# TODO find some way to avoid permutations for multiple operations starting at the same time?
#   or will pareto handle this for us?

# TODO optimization: only consider taking actions after waiting that were possible because of the waiting after this?
#   or will pareto handle that for us?

def recurse(problem: Problem, frontier: ParetoFrontier, state: RecurseState):
    # Always:
    # * check if the problem is done, report result
    # * drop all values that will never be used again from memories
    # Possible actions:
    # * start an operation (if the inputs are already available in the connected memories!):
    #   * on any of the currently idle cores
    #   * wait for any of the currently running cores, then start it on that one
    # * start a transfer:
    #   * on any of the currently idle channels
    #   * wait for any of the currently running channels to become idle, then start it on that one
    # * drop a value

    state.print(problem)
    hardware = problem.hardware
    graph = problem.graph

    # TODO check problem done, report back
    # TODO drop fully dead values

    # wait for the next node to finish
    first_done_time = min(
        min((r.time_done for r in state.core_state.values() if r is not None), default=math.inf),
        min((r.time_done for r in state.channel_state.values() if r is not None), default=math.inf),
    )
    if first_done_time < math.inf:
        recurse(problem, frontier, state.clone_do_action(WaitAction(first_done_time)))

    # TODO start operations

    # start transfers
    for channel in hardware.channels:
        if state.channel_state[channel] is not None:
            continue

        if channel.dir_a_to_b:
            recurse_channel_actions(problem, frontier, state, channel, channel.memory_a, channel.memory_b)
        if channel.dir_b_to_a:
            recurse_channel_actions(problem, frontier, state, channel, channel.memory_b, channel.memory_a)


def recurse_channel_actions(
        problem: Problem, frontier: ParetoFrontier, state: RecurseState,
        channel: Channel, source: Memory, dest: Memory
):
    assert state.channel_state[channel] is None
    for value, done in state.memory_contents[source].items():
        # check if the source value is done
        if not done:
            continue

        # check if the value is already in the target memory
        if value in state.memory_contents[dest]:
            continue

        # check if the value can fit in the target memory
        target_size_before = sum(v.size_bits for v in state.memory_contents[dest])
        if value.size_bits + target_size_before > dest.size_bits:
            continue

        # run action
        action = ChannelAction(channel, source=source, dest=dest, value=value)
        next_state = state.clone_do_action(action)
        recurse(problem, frontier, next_state)
