import itertools
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, DefaultDict, Set

from matplotlib import pyplot as plt

from core.action import ActionWait, ActionCore, ActionChannel, Action
from core.frontier import ParetoFrontier, tuple_dominates, render_2d_frontier
from core.problem import Problem, OperationNode, Memory, Channel, Core
from core.schedule import Schedule
from core.util import zip_eq


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

# TODO general effects/available system,
#   that only allows actions that are newly available ready after waiting to be queued

@dataclass(eq=False)
class Frontiers:
    done: ParetoFrontier[Tuple[float, float]]
    partial: ParetoFrontier['RecurseState']


def schedule(problem: Problem):
    start = time.perf_counter()

    state = RecurseState.initial(problem)

    frontiers = Frontiers(
        done=ParetoFrontier[Tuple[float, float]](dominates=tuple_dominates),
        partial=ParetoFrontier(dominates=lambda curr, other: curr.dominates(other)),
    )

    initial_events = Events()
    for core in problem.hardware.cores:
        initial_events.add_free_core(core)
    for channel in problem.hardware.channels:
        initial_events.add_free_channel(channel)
    for value, mem in problem.placement_inputs.items():
        initial_events.add_value_available_in(value, mem)

    recurse(problem, frontiers, state, events=initial_events, skipped_actions=[])

    end = time.perf_counter()
    print(f"Visited {next_plot_index} states in {end - start}s")


@dataclass(eq=False)
class RecurseState:
    problem: Problem

    curr_time: float
    curr_energy: float
    minimum_time: float

    unstarted_nodes: List[OperationNode]
    value_remaining_unstarted_uses: Dict[OperationNode, int]

    core_state: Dict[Core, Optional[ActionCore]]
    channel_state: Dict[Channel, Optional[ActionChannel]]

    # mem -> value -> time_available (None if already available)
    memory_contents: Dict[Memory, Dict[OperationNode, Optional[float]]]
    # (node, mem) -> read
    # TODO add times to this too?
    active_reads: DefaultDict[Tuple[OperationNode, Memory], int]

    actions_taken: List[Action]

    def assert_valid(self):
        # type checks
        assert isinstance(self.problem, Problem)
        assert isinstance(self.curr_time, float)
        assert isinstance(self.curr_energy, float)
        assert isinstance(self.minimum_time, float)

        assert isinstance(self.unstarted_nodes, list)
        for node in self.unstarted_nodes:
            assert isinstance(node, OperationNode)

        assert isinstance(self.value_remaining_unstarted_uses, dict)
        for node, uses in self.value_remaining_unstarted_uses.items():
            assert isinstance(node, OperationNode)
            assert isinstance(uses, int)

        expected_active_reads = {}

        assert isinstance(self.core_state, dict)
        for core, action in self.core_state.items():
            assert isinstance(core, Core)
            assert action is None or isinstance(action, ActionCore)
            if action is not None:
                for index_input, mem_input in enumerate(action.alloc.input_memories):
                    key = (action.node.inputs[index_input], mem_input)
                    expected_active_reads.setdefault(key, 0)
                    expected_active_reads[key] += 1
                assert self.memory_contents[action.alloc.output_memory][action.node] == action.time_end

        assert isinstance(self.channel_state, dict)
        for channel, action in self.channel_state.items():
            assert isinstance(channel, Channel)
            assert action is None or isinstance(action, ActionChannel)

            if action is not None:
                expected_active_reads.setdefault((action.value, action.mem_source), 0)
                expected_active_reads[(action.value, action.mem_source)] += 1
                assert self.memory_contents[action.mem_dest][action.value] == action.time_end

        assert isinstance(self.memory_contents, dict)
        for memory, nodes_dict in self.memory_contents.items():
            assert isinstance(memory, Memory)
            for node, time_available in nodes_dict.items():
                assert isinstance(node, OperationNode)
                assert time_available is None or isinstance(time_available, float)

        assert isinstance(self.active_reads, defaultdict)
        for (node, memory), counter in self.active_reads.items():
            assert isinstance(node, OperationNode)
            assert isinstance(memory, Memory)
            assert isinstance(counter, int)

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
            value_remaining_unstarted_uses[output] += 1

        # memory contents
        memory_contents = {m: {} for m in hw.memories}
        for node, memory in problem.placement_inputs.items():
            memory_contents[memory][node] = None

        return RecurseState(
            problem=problem,

            curr_time=0.0,
            curr_energy=0.0,
            minimum_time=0.0,

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
            problem=self.problem,

            curr_time=self.curr_time,
            curr_energy=self.curr_energy,
            minimum_time=self.minimum_time,

            unstarted_nodes=self.unstarted_nodes.copy(),
            value_remaining_unstarted_uses=self.value_remaining_unstarted_uses.copy(),

            core_state=self.core_state.copy(),
            channel_state=self.channel_state.copy(),
            memory_contents={m: c.copy() for m, c in self.memory_contents.items()},
            active_reads=self.active_reads.copy(),

            actions_taken=self.actions_taken.copy(),
        )

    def value_dom_key_min(self, value: OperationNode, memory: Memory):
        if self.value_remaining_unstarted_uses[value] == 0:
            # dead, best possible
            return 0, 0
        if value in self.memory_contents[memory]:
            # scheduled, better if done (or done earlier)
            time_ready = self.memory_contents[memory][value]
            if time_ready is None:
                return 1, 0
            else:
                return 2, -time_ready
        # not even scheduled, worst
        return 3, 0

    def core_dom_key_min(self, core: Core):
        if self.core_state[core] is None:
            return 0, 0
        return 1, -self.core_state[core].time_end

    def channel_dom_key_min(self, channel: Channel):
        if self.channel_state[channel] is None:
            return 0, 0
        return 1, -self.channel_state[channel].time_end

    def dominates(self, other: "RecurseState"):
        """
        Returns whether this state dominates `other` by being
        strictly better in any way and better or equal in every way.
        """

        # TODO better way to think about this:
        #   return true iff we can reach "other" from "self" by doing some useless actions:
        #     waiting, burning energy, dropping values, ...

        # TODO do we even need to assert this?
        # assert isinstance(self.actions_taken[-1], ActionWait)
        # assert isinstance(other.actions_taken[-1], ActionWait)

        # better in any way
        compare_better = False
        compare_worse = False

        def check_minimize(self_value, other_value):
            nonlocal compare_better, compare_worse
            if self_value > other_value:
                compare_worse = True
                return True
            if self_value < other_value:
                compare_better = True

        # energy and time
        if check_minimize(self.curr_energy, other.curr_energy):
            return False
        if check_minimize(self.minimum_time, other.minimum_time):
            return False

        # value in memory availability
        for mem in self.memory_contents:
            for value in self.memory_contents[mem]:
                if check_minimize(self.value_dom_key_min(value, mem), other.value_dom_key_min(value, mem)):
                    return False
        # TODO is it useful to skip duplicates in this second loop?
        for mem in other.memory_contents:
            for value in other.memory_contents[mem]:
                if check_minimize(self.value_dom_key_min(value, mem), other.value_dom_key_min(value, mem)):
                    return False

        # core and channel free-ness
        for core in self.problem.hardware.cores:
            if check_minimize(self.core_dom_key_min(core), other.core_dom_key_min(core)):
                return False
        for channel in self.problem.hardware.channels:
            if check_minimize(self.channel_dom_key_min(channel), other.channel_dom_key_min(channel)):
                return False

        # TODO add value unlock times?

        # for values (in memories): dead (no remaining usages) is better than scheduled (with lower timing done being better) is better than unscheduled

        # TODO go through all properties
        # TODO different active reads and writes should break up dominance too!
        #       (since values being locked in different places can have implications)

        # TODO for computations: started < running (by time) < done
        #   same for memory transfers! (except it's a separate value per core/value)
        #     what about a value being copied, dropped and then recopied? -> should be fine, dominance is not scalar

        # at this point we know we're not worse in any way, so we dominate if we're better if any way
        assert not compare_worse
        return compare_better

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

    def is_done(self):
        for node, mem in self.problem.placement_outputs.items():
            if node not in self.memory_contents[mem] or self.memory_contents[mem][node] is not None:
                return False

        assert all(s is None for s in self.core_state.values())
        return True

    def mem_space_used(self, dest):
        return sum(v.size_bits for v in self.memory_contents[dest])

    def clone_do_action(self, action: Action):
        next_state = self.clone()
        next_state.do_action_full(action)
        return next_state

    def clone_do_wait(self, action: ActionWait):
        next_state = self.clone()
        events = next_state.do_wait_action_full(action)
        # TODO make events part of the state?
        return next_state, events

    def active_read_add(self, mem: Memory, value: OperationNode):
        assert isinstance(mem, Memory) and isinstance(value, OperationNode)
        key = (value, mem)
        self.active_reads.setdefault(key, 0)
        self.active_reads[key] += 1

    def active_read_drop(self, mem: Memory, value: OperationNode, events: 'Events'):
        assert isinstance(mem, Memory) and isinstance(value, OperationNode)
        key = (value, mem)
        assert key in self.active_reads

        self.active_reads[key] -= 1

        count = self.active_reads[key]
        assert count >= 0

        if count == 0:
            self.active_reads.pop(key)
            events.add_release_read(value, mem)

    def do_action_full(self, action: Action):
        assert action.time_start == self.curr_time

        # print(f"Running action: {action}")
        self.minimum_time = max(self.minimum_time, action.time_end)
        self.curr_energy += action.energy

        if isinstance(action, ActionCore):
            self.do_core_action_part(action)
        elif isinstance(action, ActionChannel):
            self.do_channel_action_part(action)
        else:
            assert False, f"unexpected action type: {action}"

        self.actions_taken.append(action)

    def do_wait_action_full(self, action: Action) -> 'Events':
        assert isinstance(action, ActionWait)

        assert action.time_start == self.curr_time
        assert action.time_end > self.curr_time
        self.curr_time = action.time_end
        self.minimum_time = max(self.minimum_time, action.time_end)
        self.curr_energy += action.energy

        events = Events()

        for core, core_action in self.core_state.items():
            if core_action is None or core_action.time_end > self.curr_time:
                continue

            node = core_action.node
            mem_out = core_action.alloc.output_memory

            assert self.memory_contents[mem_out][node] is not None
            self.memory_contents[mem_out][node] = None
            events.add_value_available_in(node, mem_out)

            for value_input, mem_input in zip_eq(node.inputs, core_action.alloc.input_memories):
                # TODO probably a bug
                self.value_remaining_unstarted_uses[value_input] -= 1
                self.active_read_drop(mem_input, value_input, events)

            self.core_state[core] = None
            events.add_free_core(core)

        for channel, channel_action in self.channel_state.items():
            if channel_action is None or channel_action.time_end > self.curr_time:
                continue

            assert self.memory_contents[channel_action.mem_dest][channel_action.value] is not None
            self.memory_contents[channel_action.mem_dest][channel_action.value] = None
            events.add_value_available_in(channel_action.value, channel_action.mem_dest)

            self.active_read_drop(channel_action.mem_source, channel_action.value, events)

            self.channel_state[channel] = None
            events.add_free_channel(channel)

        self.actions_taken.append(action)
        return events

    def do_core_action_part(self, action: ActionCore):
        node = action.node
        alloc = action.alloc
        core = alloc.core

        self.unstarted_nodes.remove(node)

        assert node not in self.memory_contents[alloc.output_memory]
        self.memory_contents[alloc.output_memory][node] = action.time_end

        # add claims
        for index_input, mem_input in enumerate(alloc.input_memories):
            input = node.inputs[index_input]
            self.active_reads[(input, mem_input)] += 1

        assert self.core_state[core] is None
        self.core_state[core] = action

    def do_channel_action_part(self, action: ActionChannel):
        assert action.value not in self.memory_contents[action.mem_dest]
        self.memory_contents[action.mem_dest][action.value] = action.time_end
        self.active_reads[(action.value, action.mem_source)] += 1

        assert self.channel_state[action.channel] is None
        self.channel_state[action.channel] = action

    def undo_action(self, _: Action):
        # TODO implement this and stop using clone for a speedup
        assert False, "TODO"


# TODO maybe merge this into state?
class Events:
    def __init__(self):
        self.trigger_free_core: Set[Core] = set()
        self.trigger_free_channel: Set[Channel] = set()
        self.trigger_value_available: Set[Tuple[OperationNode, Memory]] = set()
        self.trigger_read_released: Set[Tuple[OperationNode, Memory]] = set()
        self.trigger_space_increased: Dict[Memory, Tuple[int, int]] = {}

        self.triggered = None
        self.state: Optional[RecurseState] = None

    def clone(self) -> 'Events':
        clone = Events()
        clone.trigger_free_core = self.trigger_free_core.copy()
        clone.trigger_free_channel = self.trigger_free_channel.copy()
        clone.trigger_value_available = self.trigger_value_available.copy()
        clone.trigger_read_released = self.trigger_read_released.copy()
        clone.trigger_space_increased = self.trigger_space_increased.copy()
        return clone

    def add_free_core(self, core: Core):
        self.trigger_free_core.add(core)

    def add_free_channel(self, channel: Channel):
        self.trigger_free_channel.add(channel)

    def add_value_available_in(self, value: OperationNode, mem: Memory):
        self.trigger_value_available.add((value, mem))

    def add_release_read(self, value: OperationNode, mem: Memory):
        self.trigger_read_released.add((value, mem))

    def add_increase_memory_space(self, mem: Memory, bits_used_before: int, bits_used_after: int):
        assert mem not in self.trigger_space_increased, "Can only increase memory space once per memory."
        assert bits_used_after < bits_used_before

        self.trigger_space_increased[mem] = (bits_used_before, bits_used_after)

    def record_triggers(self, state: RecurseState):
        self.triggered = False
        self.state = state

    def was_triggered(self) -> bool:
        assert self.triggered is not None
        triggered = self.triggered
        self.triggered = None
        self.state = None
        return triggered

    def _mark_trigger(self, trigger: bool):
        self.triggered = self.triggered or trigger

    def check_free_core(self, core: Core) -> bool:
        self._mark_trigger(core in self.trigger_free_core)
        return self.state.core_state[core] is None

    def check_free_channel(self, channel: Channel) -> bool:
        self._mark_trigger(channel in self.trigger_free_channel)
        return self.state.channel_state[channel] is None

    def check_value_available(self, value: OperationNode, mem: Memory) -> bool:
        self._mark_trigger((value, mem) in self.trigger_value_available)
        return value in self.state.memory_contents[mem] and self.state.memory_contents[mem][value] is None

    def check_read_released(self, value: OperationNode, mem: Memory) -> bool:
        # TODO does this function have convenient semantics for value dropping?
        self._mark_trigger((value, mem) in self.trigger_read_released)
        return self.state.active_reads[(value, mem)] == 0

    def check_space_available(self, mem: Memory, bits: int) -> bool:
        # inf memory is always available and never triggers
        if mem.size_bits is None:
            return True

        bits_used_now = self.state.mem_space_used(mem)
        fits = bits_used_now + bits <= mem.size_bits

        if mem in self.trigger_space_increased:
            bits_used_before, _ = self.trigger_space_increased[mem]
            self._mark_trigger(fits and mem.size_bits < bits_used_before + bits)

        return fits


# TODO find a working solution first to avoid copying stuff around forever without making any progress?
# TODO have a separate frontier for only energy and latency for fully finished schedules?

# TODO repeatedly check frontier between different action attempts? will we ever get new hits?
# TODO find some way to avoid permutations for multiple operations starting at the same time?
#   or will pareto handle this for us?

# TODO optimization: only consider taking actions after waiting that were possible because of the waiting after this?
#   or will pareto handle that for us?
# TODO optimization: after copying a value to a buffer, force _something_ to use it at some point
#   equivalently refuse to drop it until it's used at least once
#   (and then after ever change to drop it that was not taken, force using it again)


# prev = time.perf_counter()


# TODO debug why we never visit non-redundant copy solution?
def recurse(
        problem: Problem, frontiers: Frontiers, state: RecurseState,
        events: Events, skipped_actions: List[Action]
):
    # global prev
    # now = time.perf_counter()
    # if now - prev >= 0.1 or True:
    #     prev = now
    #     f = None
    # print("Current state:", file=f)
    # for action in state.actions_taken:
    #     print(f"  {action}", file=f)

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

    state.assert_valid()

    # defensive copy
    # TODO replace skipped_actions with action comparison to the last skipped action?
    # TODO or a bigger reorg: have a separate recursion function for action starting
    # TODO is this events clone really necessary?
    skipped_actions = list(skipped_actions)
    events = events.clone()

    # TODO remove this once the real one works properly
    #   (and is used even for non-wait states)

    # TODO put this bounding in a better place
    min_additional_energy = 0
    for node in state.unstarted_nodes:
        min_additional_energy += min(alloc.energy for alloc in problem.possible_allocations[node])

    min_additional_time = max(
        (min(alloc.time for alloc in problem.possible_allocations[node]) for node in state.unstarted_nodes),
        default=0
    )
    min_time = max(state.minimum_time, state.curr_time + min_additional_time)

    if not frontiers.done.would_add((min_time, state.curr_energy + min_additional_energy)):
        return

    # TODO double check the pareto logic, why can we only do this after a wait?
    # TODO re-enable this
    # if len(state.actions_taken) and isinstance(state.actions_taken[-1], ActionWait):
    if not frontiers.partial.add(state):
        return

    if state.is_done():
        # TODO cancel all still-running channel transfers and subtract their energy again?
        #   or let the rest of the solver figure out the better solution
        is_better = frontiers.done.add((state.curr_time, state.curr_energy))
        # frontiers.complete.add((state.curr_time, state.curr_energy), state.actions_taken)

        log_state(state, is_better=is_better)
        if is_better:
            render_2d_frontier(frontiers.done, ("time", "energy"))

        return

    log_state(state, is_better=False)

    # if next_plot_index >= 10_000:
    #     return

    if next_plot_index % 1000 == 0:
        print(f"Frontier sizes: done={len(frontiers.done)}, partial={len(frontiers.partial)}")

    # drop dead values from all memories
    for mem, nodes_dict in state.memory_contents.items():
        dead_values = set()
        for node, time_ready in nodes_dict.items():
            # TODO dead values should never even be started, add an assert for this
            if time_ready is None and state.value_remaining_unstarted_uses[node] == 0:
                dead_values.add(node)
        used_before = state.mem_space_used(mem)
        for v in dead_values:
            nodes_dict.pop(v, None)
        used_after = state.mem_space_used(mem)
        if used_after != used_before:
            events.add_increase_memory_space(mem, used_before, used_after)

    # TODO action: drop value from core
    #    add a bunch of conditions to this to ensure we don't get stuck looping forever
    #      * don't copy value to core if it was "recently" (ie. without any intermediate interaction with (or write to(?)) the memory) dropped

    # wait for the next node to finish
    first_done_time = min(
        min((r.time_end for r in state.core_state.values() if r is not None), default=math.inf),
        min((r.time_end for r in state.channel_state.values() if r is not None), default=math.inf),
    )
    if first_done_time < math.inf:
        action_wait = ActionWait(state.curr_time, first_done_time)
        next_state, next_events = state.clone_do_wait(action_wait)
        # after a wait all actions are allowed again
        # TODO actually no, restrict this even more
        recurse(problem, frontiers, next_state, next_events, skipped_actions=[])

    # start core operations
    for node in state.unstarted_nodes:
        for alloc in problem.possible_allocations[node]:
            events.record_triggers(state)

            # check if core is free
            if not events.check_free_core(alloc.core):
                continue

            # check if the output memory has space
            if not events.check_space_available(alloc.output_memory, node.size_bits):
                continue

            # check if inputs are available in the right memories
            if not all(events.check_value_available(v, m) for v, m in zip_eq(node.inputs, alloc.input_memories)):
                continue

            # final trigger and skip checks
            if not events.was_triggered():
                continue
            # TODO move to the front?
            action_core = ActionCore(time_start=state.curr_time, node=node, alloc=alloc)
            if action_core in skipped_actions:
                continue

            # potentially run action
            next_state = state.clone_do_action(action_core)
            recurse(problem, frontiers, next_state, events, skipped_actions)
            skipped_actions.append(action_core)

    # start transfers
    for channel in problem.hardware.channels:
        events.record_triggers(state)

        # check that channel is free
        if not events.check_free_channel(channel):
            continue

        if channel.dir_a_to_b:
            recurse_channel_actions(
                problem, frontiers, state, events, skipped_actions,
                channel, channel.memory_a, channel.memory_b
            )
        if channel.dir_b_to_a:
            recurse_channel_actions(
                problem, frontiers, state, events, skipped_actions,
                channel, channel.memory_b, channel.memory_a
            )


def recurse_channel_actions(
        problem: Problem, frontiers: Frontiers, state: RecurseState, events: Events, skipped_actions: List[Action],
        channel: Channel, source: Memory, dest: Memory,
):
    assert state.channel_state[channel] is None
    for value in state.memory_contents[source]:
        events.record_triggers(state)

        # check if the value is still alive, no point in copying dead values around
        if state.value_remaining_unstarted_uses[value] == 0:
            continue
        # check if the value is already (going to be in) the target memory
        # TODO what if we're faster/cheaper? if everything is right another attempt will catch that
        # TODO should this count as an event too? "the value has recently been dropped"
        if value in state.memory_contents[dest]:
            continue

        # check if the source value is done
        if not events.check_value_available(value, source):
            continue

        # check if the value can fit in the target memory
        if not events.check_space_available(dest, value.size_bits):
            continue

        # final trigger and skip checks
        if not events.was_triggered():
            continue
        # TODO move to the front?
        action_channel = ActionChannel(
            time_start=state.curr_time,
            channel=channel, mem_source=source, mem_dest=dest,
            value=value
        )
        if action_channel in skipped_actions:
            continue

        # potentially run action
        next_state = state.clone_do_action(action_channel)
        recurse(problem, frontiers, next_state, events, skipped_actions)
        skipped_actions.append(action_channel)


next_plot_index = 0


def log_state(state: RecurseState, is_better: bool):
    is_done = state.is_done()

    global next_plot_index
    index = next_plot_index
    next_plot_index += 1

    if index == 0:
        shutil.rmtree("../../ignored/schedules", ignore_errors=True)
        os.makedirs("../../ignored/schedules/done", exist_ok=False)
        os.makedirs("../../ignored/schedules/all", exist_ok=False)
        os.makedirs("../../ignored/schedules/better", exist_ok=False)

    include = index % 1000 == 0
    if include:
        print(f"Visited {index} states")

    if include or is_done or is_better:
        result = Schedule(problem=state.problem, actions=state.actions_taken, curr_time=state.curr_time)
        fig, ax = plt.subplots()
        result.plot_schedule_actions(ax)

        fig.savefig(f"../../ignored/schedules/all/schedule_{index}.png")
        if is_done:
            fig.savefig(f"../../ignored/schedules/done/schedule_{index}.png")
        if is_better:
            fig.savefig(f"../../ignored/schedules/better/schedule_{index}.png")
        plt.close(fig)
