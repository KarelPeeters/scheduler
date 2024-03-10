import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union, DefaultDict

from core.problem import Problem, OperationNode, Memory, Channel, Core, OperationAllocation


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

class SimpleFrontier:
    def __init__(self):
        self.min_time = math.inf
        self.min_time_actions = None

        self.min_energy = math.inf
        self.min_energy_actions = None

    def add_solution(self, time: float, energy: float, actions: List):
        either = False

        if time < self.min_time:
            either = True
            self.min_time = time
            self.min_time_actions = actions
            print(f"New best time solution: ({time}, {energy})")

        if energy < self.min_energy:
            either = True
            self.min_energy = energy
            self.min_energy_actions = actions
            print(f"New best energy solution: ({time}, {energy})")

        if either:
            for action in actions:
                print(f"  {action}")

    def is_dominated(self, time: float, energy: float):
        return time >= self.min_time and energy >= self.min_energy


def schedule(problem: Problem):
    # The pareto key consists of:
    # * scheduling progress: [done] per real node (higher is better)
    # * core availability: [-next_free_time] per core and channel (lower is better)
    # * costs: (-time), (-energy) (lower is better)
    # TODO include "value in memory" availability as value
    # TODO add a more complicated "dominated proper", eg. we can obvious trade memory availability for time+energy

    # real_nodes = [n for n in problem.graph.nodes if n not in problem.graph.inputs]
    # frontier = ParetoFrontier(len(real_nodes) + len(problem.hardware.cores) + len(problem.hardware.channels) + 2)

    frontier = SimpleFrontier()

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
class ActionWait:
    time_until: float


@dataclass(frozen=True, eq=False)
class ActionCore:
    node: OperationNode
    alloc: OperationAllocation

    def __str__(self):
        return f"ActionCore(node={self.node.id}, alloc={self.alloc.id}, core={self.alloc.core.id})"


@dataclass(frozen=True, eq=False)
class ActionChannel:
    channel: Channel
    source: Memory
    dest: Memory
    value: OperationNode

    def __str__(self):
        return f"ActionChannel(channel={self.channel.id}, value={self.value.id}, source={self.source.id}, dest={self.dest.id})"


Action = Union[ActionWait, ActionCore, ActionChannel]


@dataclass(frozen=True, eq=False)
class CoreRunning:
    time_done: float
    action: ActionCore

    def __str__(self):
        return f"CoreRunning(time_done={self.time_done}, action={self.action})"


@dataclass(frozen=True, eq=False)
class ChannelRunning:
    time_done: float
    action: ActionChannel

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
    active_reads: DefaultDict[Tuple[OperationNode, Memory], int]

    actions_taken: List[Action]

    @staticmethod
    def initial(problem: Problem):
        # basics
        hw = problem.hardware
        graph = problem.graph

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
            active_reads=defaultdict(lambda: 0),
            actions_taken=[],
        )

    def clone(self):
        """ deep clone """
        return RecurseState(
            curr_time=self.curr_time,
            curr_energy=self.curr_energy,

            unstarted_nodes=self.unstarted_nodes.copy(),
            value_remaining_unstarted_uses=self.value_remaining_unstarted_uses.copy(),

            core_state=self.core_state.copy(),
            channel_state=self.channel_state.copy(),
            memory_contents={m: c.copy() for m, c in self.memory_contents.items()},
            active_reads=self.active_reads.copy(),

            actions_taken=self.actions_taken.copy(),
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
        for (node, memory), counter in self.active_reads.items():
            print(f"    ({node.id}, {memory.id}): {counter}")
        print(f"  }}")
        print(")")

    def is_done(self, problem: Problem):
        for node, mem in problem.placement_outputs.items():
            if node not in self.memory_contents[mem] or self.memory_contents[mem][node] is not True:
                return False

        assert all(s is None for s in self.core_state.values())
        return True

    def mem_space_used(self, dest):
        return sum(v.size_bits for v in self.memory_contents[dest])

    def clone_do_action(self, action: Action):
        next_state = self.clone()
        next_state.do_action(action)
        return next_state

    def do_action(self, action: Action):
        # print(f"Running action: {action}")

        if isinstance(action, ActionWait):
            self.do_wait_action(action)
        elif isinstance(action, ActionCore):
            self.do_core_action(action)
        elif isinstance(action, ActionChannel):
            self.do_channel_action(action)
        else:
            assert False, f"unknown action type: {action}"

        self.actions_taken.append(action)

    def do_wait_action(self, action: ActionWait):
        assert action.time_until > self.curr_time
        self.curr_time = action.time_until

        for core, state in self.core_state.items():
            if state is None or state.time_done > self.curr_time:
                continue

            node = state.action.node
            mem_out = state.action.alloc.output_memory

            assert self.memory_contents[mem_out][node] is False
            self.memory_contents[mem_out][node] = True

            for mem_input in state.action.alloc.input_memories:
                self.active_reads[(node, mem_input)] -= 1
                self.core_state[core] = None

        for channel, state in self.channel_state.items():
            if state is None or state.time_done > self.curr_time:
                continue

            assert self.memory_contents[state.action.dest][state.action.value] is False
            self.memory_contents[state.action.dest][state.action.value] = True
            self.active_reads[(state.action.value, state.action.source)] -= 1
            self.channel_state[channel] = None

    def do_core_action(self, action: ActionCore):
        node = action.node
        alloc = action.alloc
        core = alloc.core

        self.unstarted_nodes.remove(node)

        # add claims
        assert node not in self.memory_contents[alloc.output_memory]
        self.memory_contents[alloc.output_memory][node] = False

        for index_input, mem_input in enumerate(alloc.input_memories):
            input = node.inputs[index_input]
            self.active_reads[(input, mem_input)] += 1

        self.curr_energy += alloc.energy

        assert self.core_state[core] is None
        self.core_state[core] = CoreRunning(
            time_done=self.curr_time + alloc.time,
            action=action
        )

    def do_channel_action(self, action: ActionChannel):
        assert action.value not in self.memory_contents[action.dest]
        self.memory_contents[action.dest][action.value] = False
        self.active_reads[(action.value, action.source)] += 1

        time_transfer = action.channel.latency + action.value.size_bits * action.channel.time_per_bit
        energy_transfer = action.value.size_bits * action.channel.energy_per_bit
        self.curr_energy += energy_transfer

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

def recurse(problem: Problem, frontier: SimpleFrontier, state: RecurseState):
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

    # state.print(problem)
    # print("Current time/energy:", state.curr_time, state.curr_energy)

    hardware = problem.hardware
    graph = problem.graph

    if frontier.is_dominated(state.curr_time, state.curr_energy):
        return

    if state.is_done(problem):
        # TODO cancel all still-running channel transfers and subtract their energy again?
        frontier.add_solution(state.curr_time, state.curr_energy, state.actions_taken)

    # TODO check problem done, report back
    # TODO drop fully dead values

    # TODO action: drop value from core
    #    add a bunch of conditions to this to ensure we don't keep looping

    # wait for the next node to finish
    first_done_time = min(
        min((r.time_done for r in state.core_state.values() if r is not None), default=math.inf),
        min((r.time_done for r in state.channel_state.values() if r is not None), default=math.inf),
    )
    if first_done_time < math.inf:
        next_state = state.clone_do_action(ActionWait(first_done_time))
        recurse(problem, frontier, next_state)

    # start core operations
    for node in state.unstarted_nodes:
        for alloc in problem.possible_allocations[node]:
            # check if core is free
            core = alloc.core
            if state.core_state[core] is not None:
                continue

            # check if the output memory has space
            if node.size_bits + state.mem_space_used(alloc.output_memory) > alloc.output_memory.size_bits:
                continue

            # check if inputs are available in the right memories
            inputs_available = True
            for index_input, mem_input in enumerate(alloc.input_memories):
                input = node.inputs[index_input]
                if input not in state.memory_contents[mem_input] or not state.memory_contents[mem_input][input]:
                    inputs_available = False
                    break
            if not inputs_available:
                continue

            next_state = state.clone_do_action(ActionCore(node=node, alloc=alloc))
            recurse(problem, frontier, next_state)

    # start transfers
    for channel in hardware.channels:
        if state.channel_state[channel] is not None:
            continue

        if channel.dir_a_to_b:
            recurse_channel_actions(problem, frontier, state, channel, channel.memory_a, channel.memory_b)
        if channel.dir_b_to_a:
            recurse_channel_actions(problem, frontier, state, channel, channel.memory_b, channel.memory_a)


def recurse_channel_actions(
        problem: Problem, frontier: SimpleFrontier, state: RecurseState,
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
        if dest.size_bits is not None:
            if value.size_bits + state.mem_space_used(dest) > dest.size_bits:
                continue

        # run action
        action = ActionChannel(channel, source=source, dest=dest, value=value)
        next_state = state.clone_do_action(action)
        recurse(problem, frontier, next_state)
