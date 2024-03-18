import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, DefaultDict

from matplotlib import pyplot as plt

from core.action import ActionWait, ActionCore, ActionChannel, Action
from core.frontier import ParetoFrontier, tuple_dominates, render_2d_frontier
from core.problem import Problem, OperationNode, Memory, Channel, Core
from core.schedule import Schedule


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

    recurse(problem, frontiers, state, skipped_actions=[])

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
            value_remaining_unstarted_uses[output] += 1

        # memory contents
        memory_contents = {m: {} for m in hw.memories}
        for node, memory in problem.placement_inputs.items():
            memory_contents[memory][node] = True

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
            # scheduled, better if done
            return 1, not self.memory_contents[memory]
        # not even scheduled, worst
        return 2, 0

    def dominates(self, other: "RecurseState"):
        """
        Returns whether this state dominates `other` by being
        strictly better in any way and better or equal in every way.
        """

        # TODO do we even need to assert this?
        assert isinstance(self.actions_taken[-1], ActionWait)
        assert isinstance(other.actions_taken[-1], ActionWait)

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

        if check_minimize(self.curr_energy, other.curr_energy):
            return False
        if check_minimize(self.minimum_time, other.minimum_time):
            return False

        # TODO early exit eg. if both are dead
        # TODO is just checking all these values already enough for dominance?
        for value in self.problem.graph.nodes:
            for mem in self.problem.hardware.memories:
                if check_minimize(self.value_dom_key_min(value, mem), other.value_dom_key_min(value, mem)):
                    return False

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
            if node not in self.memory_contents[mem] or self.memory_contents[mem][node] is not True:
                return False

        assert all(s is None for s in self.core_state.values())
        return True

    # def to_pareto_key(self) -> Tuple:
    #     # TODO add a more complicated "dominated" option, eg. we can obvious trade memory availability for time+energy
    #
    #     # The pareto key (for which higher is better) consists of:
    #     # * costs: (-time), (-energy) (lower is better)
    #     result = [-self.minimum_time, -self.curr_energy]
    #
    #     # TODO add this again
    #     # # * scheduling progress: [started] per core and node (higher is better)
    #     # # TODO should progress by done or started? does it matter?
    #     for node in self.problem.graph.nodes:
    #         if node in self.problem.graph.inputs:
    #             continue
    #         for alloc in self.problem.possible_allocations[node]:
    #             started = False
    #             for action in self.actions_taken:
    #                 if isinstance(action, ActionCore) and action.node == node and action.alloc == alloc:
    #                     started = True
    #                     break
    #             result.append(started)
    #
    #     # * worker availability: [-next_free_time] per core and channel (lower is better)
    #     # TODO should time until ready be relative or absolute?
    #     for state in self.core_state.values():
    #         time_left = 0 if state is None else state.time_end - self.curr_time
    #         result.append(-time_left)
    #     for state in self.channel_state.values():
    #         time_left = 0 if state is None else state.time_end - self.curr_time
    #         result.append(-time_left)
    #
    #     # * data availability [is_available] per mem and value (higher is better)
    #     # TODO is this true once dropping gets implemented?
    #     for mem in self.problem.hardware.memories:
    #         for value in self.problem.graph.nodes:
    #             is_available = value in self.memory_contents[mem] and self.memory_contents[mem][value]
    #             result.append(is_available)
    #
    #     return tuple(result)

    def mem_space_used(self, dest):
        return sum(v.size_bits for v in self.memory_contents[dest])

    def clone_do_action(self, action: Action):
        next_state = self.clone()
        next_state.do_action(action)
        return next_state

    def do_action(self, action: Action):
        # print(f"Running action: {action}")
        self.minimum_time = max(self.minimum_time, action.time_end)
        self.curr_energy += action.energy

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
        assert action.time_end > self.curr_time
        self.curr_time = action.time_end

        for core, action in self.core_state.items():
            if action is None or action.time_end > self.curr_time:
                continue

            node = action.node
            mem_out = action.alloc.output_memory

            assert self.memory_contents[mem_out][node] is False
            self.memory_contents[mem_out][node] = True

            for node_input in node.inputs:
                self.value_remaining_unstarted_uses[node_input] -= 1
            for mem_input in action.alloc.input_memories:
                self.active_reads[(node, mem_input)] -= 1
                self.core_state[core] = None

        for channel, action in self.channel_state.items():
            if action is None or action.time_end > self.curr_time:
                continue

            assert self.memory_contents[action.dest][action.value] is False
            self.memory_contents[action.dest][action.value] = True
            self.active_reads[(action.value, action.source)] -= 1
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

        assert self.core_state[core] is None
        self.core_state[core] = action

    def do_channel_action(self, action: ActionChannel):
        assert action.value not in self.memory_contents[action.dest]
        self.memory_contents[action.dest][action.value] = False
        self.active_reads[(action.value, action.source)] += 1

        assert self.channel_state[action.channel] is None
        self.channel_state[action.channel] = action

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
# TODO optimization: after copying a value to a buffer, force _something_ to use it at some point
#   equivalently refuse to drop it until it's used at least once
#   (and then after ever change to drop it that was not taken, force using it again)


# prev = time.perf_counter()


# TODO debug why we never visit non-redundant copy solution?
def recurse(problem: Problem, frontiers: Frontiers, state: RecurseState, skipped_actions: List[Action]):
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

    hardware = problem.hardware
    graph = problem.graph

    # defensive copy
    skipped_actions = list(skipped_actions)

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
    if len(state.actions_taken) and isinstance(state.actions_taken[-1], ActionWait):
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

    # drop dead values from all memories
    for mem, nodes_dict in state.memory_contents.items():
        dead_values = set()
        for node, done in nodes_dict.items():
            if done is True and state.value_remaining_unstarted_uses[node] == 0:
                dead_values.add(node)
        for v in dead_values:
            nodes_dict.pop(v, None)

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
        next_state = state.clone_do_action(action_wait)
        # after a wait all actions are allowed again
        # TODO actually no, restrict this even more
        recurse(problem, frontiers, next_state, skipped_actions=[])

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

            action_core = ActionCore(time_start=state.curr_time, node=node, alloc=alloc)
            if action_core in skipped_actions:
                continue

            next_state = state.clone_do_action(action_core)
            recurse(problem, frontiers, next_state, skipped_actions)
            skipped_actions.append(action_core)

    # start transfers
    for channel in hardware.channels:
        if state.channel_state[channel] is not None:
            continue

        if channel.dir_a_to_b:
            recurse_channel_actions(problem, frontiers, state, skipped_actions, channel, channel.memory_a,
                                    channel.memory_b)
        if channel.dir_b_to_a:
            recurse_channel_actions(problem, frontiers, state, skipped_actions, channel, channel.memory_b,
                                    channel.memory_a)


def recurse_channel_actions(
        problem: Problem, frontiers: Frontiers, state: RecurseState, skipped_actions: List[Action],
        channel: Channel, source: Memory, dest: Memory
):
    assert state.channel_state[channel] is None
    for value, done in state.memory_contents[source].items():
        # check if the source value is done
        if not done:
            continue

        # check if the value is still alive
        if state.value_remaining_unstarted_uses[value] == 0:
            continue

        # check if the value is already in the target memory
        if value in state.memory_contents[dest]:
            continue

        # check if the value can fit in the target memory
        if dest.size_bits is not None:
            if value.size_bits + state.mem_space_used(dest) > dest.size_bits:
                continue

        # run action
        action_channel = ActionChannel(time_start=state.curr_time, channel=channel, source=source, dest=dest,
                                       value=value)
        if action_channel in skipped_actions:
            continue
        next_state = state.clone_do_action(action_channel)
        recurse(problem, frontiers, next_state, skipped_actions)
        skipped_actions.append(action_channel)


next_plot_index = 0


def log_state(state: RecurseState, is_better: bool):
    global next_plot_index
    index = next_plot_index
    next_plot_index += 1

    return

    if index == 0:
        shutil.rmtree("../ignored/schedules", ignore_errors=True)
        os.makedirs("../ignored/schedules/done", exist_ok=False)
        os.makedirs("../ignored/schedules/all", exist_ok=False)
        os.makedirs("../ignored/schedules/better", exist_ok=False)

    result = Schedule(problem=state.problem, actions=state.actions_taken, curr_time=state.curr_time)
    fig, ax = plt.subplots()
    result.plot_schedule_actions(ax)

    fig.savefig(f"../ignored/schedules/all/schedule_{index}.png")
    if state.is_done():
        fig.savefig(f"../ignored/schedules/done/schedule_{index}.png")
    if is_better:
        fig.savefig(f"../ignored/schedules/better/schedule_{index}.png")
    plt.close(fig)
