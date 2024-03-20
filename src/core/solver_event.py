from typing import Dict, Optional, Set, List, DefaultDict, Tuple

from core.action import ActionCore, ActionChannel, Action, ActionWait, RealAction
from core.event import EventFreeCore, EventFreeChannel, EventValueAvailable, EventReadReleased, Event
from core.problem import Problem, Core, Channel, OperationNode, Memory
from core.util import zip_eq


def solve(problem: Problem):
    pass


class State:
    problem: Problem

    # list of actions taken so far
    history_pairs: List[(Set[RealAction], ActionWait)]
    history_last_actions: Set[RealAction]

    # simple float stats
    curr_time: float
    curr_energy: float

    # currently running actions
    state_core: Dict[Core, Optional[ActionCore]]
    state_channel: Dict[Channel, Optional[ActionChannel]]

    # derived information
    #   node -> (non-started) usages remaining
    value_remaining_unstarted_uses: Dict[OperationNode, int]
    #   mem -> node -> time_available (None if already available)
    memory_contents: Dict[Memory, Dict[OperationNode, Optional[float]]]  # TODO rename to refer to time
    #   (mem, node) -> read
    active_reads: DefaultDict[Tuple[Memory, OperationNode], int]
    #   nodes that haven't started yet
    unstarted_nodes: List[OperationNode]

    def init(self, problem: Problem):
        pass

    def assert_valid(self):
        # TODO add checks that make sure derived values match the actual schedule
        raise NotImplementedError("TODO")

    def clone(self) -> 'State':
        raise NotImplementedError("TODO")

    def dominates(self, other) -> bool:
        raise NotImplementedError("TODO")

    def do_action_wait(self, action: ActionWait, events: Set[Event]):
        # update time
        assert action.time_end > self.curr_time
        self.curr_time = action.time_end

        # record action
        assert self.history_last_actions
        self.history_pairs.append((self.history_last_actions, action))
        self.history_last_actions = set()

        # TODO unify cores and channels some more?
        # update core actions
        for core, action in self.state_core.items():
            if action is None or action.time_end > self.curr_time:
                continue
            node = action.node
            mem_out = action.alloc.output_memory

            # update values
            self.value_mark_available_now(mem=mem_out, value=node, events=events)
            for mem_in, value_in in zip_eq(action.alloc.input_memories, action.node.inputs):
                self.value_release_read(mem=mem_in, value=value_in, events=events)

            # clear state
            self.state_core[core] = None
            events.add(EventFreeCore(core=core))

        # update channel actions
        for channel, action in self.state_channel.items():
            if action is None or action.time_end > self.curr_time:
                continue

            # update values
            self.value_mark_available_now(mem=action.mem_dest, value=action.value, events=events)
            self.value_release_read(mem=action.mem_source, value=action.value, events=events)

            # clear state
            self.state_channel[channel] = None
            events.add(EventFreeChannel(channel=channel))

    def do_action_core(self, action: ActionCore):
        # record action
        assert action not in self.history_last_actions
        self.history_last_actions.add(action)

        self.unstarted_nodes.remove(action.node)

        # update values
        self.value_mark_available_future(mem=action.alloc.output_memory, value=action.node, time=action.time_end)
        for mem_in, value_in in zip_eq(action.alloc.input_memories, action.node.inputs):
            self.value_claim_read(mem=mem_in, value=value_in)

        # set state
        assert self.state_core[action.core] is None
        self.state_core[action.core] = action

        # TODO include:
        # for node_input in node.inputs:
        #     self.value_remaining_unstarted_uses[node_input] -= 1
        raise NotImplementedError("TODO")

    def do_action_channel(self, action: ActionChannel):
        # record action
        assert action not in self.history_last_actions
        self.history_last_actions.add(action)

        raise NotImplementedError("TODO")

    def value_mark_available_future(self, mem: Memory, value: OperationNode, time: float):
        assert value not in self.memory_contents[mem]
        self.memory_contents[mem][value] = time

    def value_mark_available_now(self, mem: Memory, value: OperationNode, events: Set[Event]):
        assert self.memory_contents[mem][value] == self.curr_time
        self.memory_contents[mem][value] = None
        events.add(EventValueAvailable(mem=mem, value=value))

    def value_claim_read(self, mem: Memory, value: OperationNode):
        # check that value is really available
        assert self.memory_contents[mem][value] is None

        # increment reads
        key = (mem, value)
        self.active_reads[key] += 1

        # decrement unstarted counter (and maybe drop value from all non-locked memories)
        self.value_remaining_unstarted_uses[value] -= 1
        if self.value_remaining_unstarted_uses[value] == 0:
            # TODO this is super awkward, since this causes a backwards dependency
            #   (eg. earlier compute nodes that didn't have enough space could have started!)
            #    can we just fully solve the "iterate over compatible sets of actions" problem?
            #      careful: it's not just a max selection, selecting additional items might make more items possible!
            #      -> not really, as long as we include dropping in the set this is already covered
            raise NotImplementedError("TODO")

    def value_release_read(self, mem: Memory, value: OperationNode, events: Set[Event]):
        # check that value is still available
        assert self.memory_contents[mem][value] is None

        # decrement reads
        key = (mem, value)
        self.active_reads[key] -= 1
        assert self.active_reads[key] >= 0
        if self.active_reads[key] == 0:
            events.add(EventReadReleased(mem=mem, value=value))


# TODO ideally we would try _not_ taking actions initially (especially channel copies),
#   is there an easy way to write that?
def recurse(state: State, events: Set[Event], max_action_skipped: Optional[Action]):
    pass
