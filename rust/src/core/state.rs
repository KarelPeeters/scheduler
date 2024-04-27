use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};

use itertools::{enumerate, zip_eq};

use crate::core::frontier::{DomBuilder, DomDir, Dominance};
use crate::core::new_frontier::SparseVec;
use crate::core::problem::{Allocation, Channel, CostTarget, Group, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionCore, ActionDrop, ActionWait, TimeRange};
use crate::core::wrapper::{Energy, Time};
use crate::dom_early_check;

#[derive(Clone)]
pub struct State {
    // minimal state
    pub actions_taken: Vec<Action>,

    // memoized information
    pub curr_time: Time,
    pub curr_energy: Energy,
    pub minimum_time: Time,

    pub state_group: Vec<Option<GroupClaim>>,
    pub state_memory_node: Vec<HashMap<Node, ValueState>>,
    pub value_live_count: Vec<usize>,

    pub unstarted_nodes: HashSet<Node>,
    pub value_remaining_unstarted_uses: Vec<u32>,

    // triggers
    pub trigger_everything: bool,
    pub trigger_group_free: Vec<bool>,
    pub trigger_value_mem_available: Vec<Vec<bool>>,
    // TODO set to false again for dead values? 
    pub trigger_value_mem_unlocked_or_read: Vec<Vec<bool>>,
    // TODO don't track this for inf-sized memories?
    pub trigger_mem_usage_decreased: Vec<Option<(u64, u64)>>,
    pub trigger_value_live_count_increased: Vec<bool>,

    // filtering (attempt -> most recent time range)
    pub tried_allocs: HashMap<Allocation, TimeRange>,
    pub tried_transfers: HashMap<(Channel, Node), TimeRange>,
}

// TODO rename
// TODO store only minimal information here
#[derive(Debug, Copy, Clone)]
pub enum GroupClaim {
    Core(ActionCore),
    Channel(ActionChannel),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Cost {
    pub time: Time,
    pub energy: Energy,
}

#[derive(Clone, Copy)]
pub struct Trigger<'s> {
    state: &'s State,
    triggered: bool,
    valid: bool,
}

// TODO add dead and unavailable here?
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ValueState {
    AvailableNow {
        // the number of currently active readers
        read_lock_count: u64,
        // the number of reads that happened in total,
        read_count: u64,

        // the initial time since when the value has been continuously available
        since: Time,
    },
    AvailableAtTime(Time),
}

impl State {
    pub fn new(problem: &Problem) -> Self {
        // aliases
        let graph = &problem.graph;
        let hardware = &problem.hardware;

        let node_count = graph.nodes().len();
        let mem_count = hardware.memories().len();
        let group_count = hardware.groups().len();

        // precompute
        let unstarted_nodes = graph.nodes().filter(|n| !graph.inputs.contains(n)).collect();

        let mut value_remaining_unstarted_uses = vec![0; node_count];
        for node in graph.nodes() {
            for x in &graph.node_info[node.0].inputs {
                value_remaining_unstarted_uses[x.0] += 1;
            }
        }
        for x in &graph.outputs {
            value_remaining_unstarted_uses[x.0] += 1;
        }

        let mut state_memory_node = vec![HashMap::new(); mem_count];
        let mut value_live_count = vec![0; node_count];

        for (&value, &mem) in zip_eq(&graph.inputs, &problem.input_placements) {
            let state = ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: Time(0) };
            let prev = state_memory_node[mem.0].insert(value, state);
            assert!(prev.is_none());
            value_live_count[value.0] += 1;
        }

        // construct
        State {
            actions_taken: vec![],
            curr_time: Time(0),
            curr_energy: Energy(0),
            minimum_time: Time(0),
            state_group: vec![None; group_count],
            state_memory_node,
            value_live_count,
            unstarted_nodes,
            value_remaining_unstarted_uses,
            trigger_everything: true,
            trigger_group_free: vec![false; group_count],
            trigger_value_mem_available: vec![vec![false; mem_count]; node_count],
            trigger_value_mem_unlocked_or_read: vec![vec![false; mem_count]; node_count],
            trigger_mem_usage_decreased: vec![None; mem_count],
            trigger_value_live_count_increased: vec![false; node_count],
            tried_allocs: HashMap::new(),
            tried_transfers: HashMap::new(),
        }
    }

    pub fn assert_valid(&self, problem: &Problem) {
        // aliases
        let graph = &problem.graph;
        let hardware = &problem.hardware;

        // let node_count = graph.nodes().len();
        let mem_count = hardware.memories().len();
        // let channel_count = hardware.channels().len();
        // let core_count = hardware.cores().len();

        // checks
        let mut expected_state_memory_read_lock_count = vec![HashMap::new(); mem_count];
        let mut expected_value_live_count = vec![0; graph.nodes().len()];

        for state in &self.state_group {
            match state {
                None => {},
                Some(GroupClaim::Core(state)) => {
                    let alloc = &problem.allocation_info[state.alloc.0];
                    for (&input, &mem) in zip_eq(&graph.node_info[alloc.node.0].inputs, &alloc.input_memories) {
                        *expected_state_memory_read_lock_count[mem.0].entry(input).or_insert(0) += 1;
                    }
                    assert_eq!(self.state_memory_node[alloc.output_memory.0].get(&alloc.node).unwrap(), &ValueState::AvailableAtTime(state.time.end));
                }
                Some(GroupClaim::Channel(state)) => {
                    let channel_info = &problem.hardware.channel_info[state.channel.0];
                    *expected_state_memory_read_lock_count[channel_info.mem_source.0].entry(state.value).or_insert(0) += 1;
                    assert_eq!(self.state_memory_node[channel_info.mem_dest.0].get(&state.value).unwrap(), &ValueState::AvailableAtTime(state.time.end));
                }
            }
        }

        for mem in hardware.memories() {
            let expected = &expected_state_memory_read_lock_count[mem.0];
            let actual = &self.state_memory_node[mem.0];

            for (node, &node_actual) in actual {
                expected_value_live_count[node.0] += 1;

                match node_actual {
                    ValueState::AvailableNow { read_lock_count, read_count: _, since: _ } => {
                        assert_eq!(expected.get(node).copied().unwrap_or(0), read_lock_count)
                    }
                    ValueState::AvailableAtTime(_) => assert!(!expected.contains_key(node)),
                }
            }

            for (node, &expected_read_lock_count) in expected {
                match *actual.get(node).unwrap() {
                    ValueState::AvailableNow { read_count: _, read_lock_count, since: _ } => {
                        assert_eq!(read_lock_count, expected_read_lock_count);
                    }
                    ValueState::AvailableAtTime(_) => panic!(),
                }
            }
        }

        assert_eq!(expected_value_live_count, self.value_live_count);
    }

    pub fn estimate_final_cost_conservative(&self, problem: &Problem) -> Cost {
        // TODO precompute these values for each node
        let min_additional_energy = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.energy)
                .min().unwrap()
        }).sum::<Energy>();

        let min_additional_time = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.time)
                .min().unwrap()
        }).max().unwrap_or(Time(0));

        Cost {
            time: max(self.minimum_time, self.curr_time + min_additional_time),
            energy: self.curr_energy + min_additional_energy,
        }
    }

    // TODO use this function in a bunch more places
    pub fn value_mem_availability(&self, value: Node, mem: Memory) -> Option<ValueState> {
        self.state_memory_node[mem.0].get(&value).copied()
    }

    pub fn value_available_in_mem_now(&self, value: Node, mem: Memory, since_max: Option<Time>) -> bool {
        match self.value_mem_availability(value, mem) {
            Some(ValueState::AvailableNow { read_lock_count: _, read_count: _, since }) => {
                match since_max {
                    None => true,
                    Some(since_max) => since <= since_max,
                }
            },
            Some(ValueState::AvailableAtTime(_)) => false,
            None => false,
        }
    }

    pub fn has_achieved_output_placements(&self, problem: &Problem) -> bool {
        for (output_i, &mem) in enumerate(&problem.output_placements) {
            let output = problem.graph.outputs[output_i];
            if !self.value_available_in_mem_now(output, mem, None) {
                return false;
            }
        }
        true
    }

    pub fn is_idle(&self) -> bool {
        self.state_group.iter().all(|s| s.is_none())
    }

    pub fn is_done(&self, problem: &Problem) -> bool {
        self.is_idle() && self.has_achieved_output_placements(problem)
    }

    pub fn current_cost(&self) -> Cost {
        Cost {
            time: self.curr_time,
            energy: self.curr_energy,
        }
    }

    pub fn mem_space_used(&self, problem: &Problem, mem: Memory) -> u64 {
        // TODO update incrementally
        let used = self.state_memory_node[mem.0].iter()
            .map(|(&value, _)| problem.graph.node_info[value.0].size_bits)
            .sum();

        if let Some(mem_size_bits) = problem.hardware.mem_info[mem.0].size_bits {
            assert!(used <= mem_size_bits);
        }

        used
    }

    pub fn new_trigger(&self) -> Trigger {
        Trigger {
            state: self,
            triggered: self.trigger_everything,
            valid: true,
        }
    }

    pub fn first_done_time(&self) -> Option<Time> {
        self.state_group.iter().filter_map(|a| a.map(|a| a.time().end)).min()
    }

    #[must_use]
    pub fn clone_and_then(&self, mut f: impl FnMut(&mut Self)) -> Self {
        let mut next = self.clone();
        f(&mut next);
        next
    }

    pub fn clear_triggers(&mut self) {
        self.trigger_everything = false;
        self.trigger_group_free.fill(false);
        self.trigger_value_mem_available.iter_mut().for_each(|x| x.fill(false));
        self.trigger_value_mem_unlocked_or_read.iter_mut().for_each(|x| x.fill(false));
        self.trigger_mem_usage_decreased.fill(None);
        self.trigger_value_live_count_increased.fill(false);
    }

    fn add_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();

        match availability {
            ValueState::AvailableNow { read_lock_count, read_count, since: _ } => {
                *read_lock_count += 1;
                *read_count += 1;

                if *read_count == 1 {
                    // this is not really useful (since it's locked anyway) but just for completeness
                    self.trigger_value_mem_unlocked_or_read[value.0][mem.0] = true;
                }
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn release_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();
        match availability {
            ValueState::AvailableNow { read_lock_count, read_count: _, since: _ } => {
                assert!(*read_lock_count > 0);
                *read_lock_count -= 1;
                if *read_lock_count == 0 {
                    self.trigger_value_mem_unlocked_or_read[value.0][mem.0] = true;
                }
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn mark_mem_value_available(&mut self, value: Node, mem: Memory, availability: ValueState) {
        let prev = self.state_memory_node[mem.0].insert(value, availability);

        match availability {
            ValueState::AvailableNow { read_lock_count, read_count, since: _ } => {
                assert!(matches!(prev, Some(ValueState::AvailableAtTime(_))));

                let trigger = &mut self.trigger_value_mem_available[value.0][mem.0];
                assert!(!*trigger);
                *trigger = true;

                if read_lock_count == 0 || read_count > 0 {
                    self.trigger_value_mem_unlocked_or_read[value.0][mem.0] = true;
                }
            }
            ValueState::AvailableAtTime(_) => {
                assert_eq!(prev, None, "Value {value:?} in mem {mem:?} already has availability, trying to insert {availability:?}");

                self.value_live_count[value.0] += 1;
                self.trigger_value_live_count_increased[value.0];
            }
        }
    }

    pub fn drop_value(&mut self, problem: &Problem, mem: Memory, value: Node) {
        // get value before removal
        let mem_space_used_before = self.mem_space_used(problem, mem);
        let value_size_bits = problem.graph.node_info[value.0].size_bits;

        let mem_size_bits = problem.hardware.mem_info[mem.0].size_bits;
        if let Some(mem_size_bits) = mem_size_bits {
            assert!(mem_space_used_before <= mem_size_bits);
        }

        // remove
        let prev = self.state_memory_node[mem.0].remove(&value);
        assert!(matches!(prev, Some(ValueState::AvailableNow { read_lock_count: 0, read_count: _, since: _ })));
        assert!(prev.is_some());

        let live_count = &mut self.value_live_count[value.0];
        assert!(*live_count > 0);
        *live_count -= 1;

        // update memory usage trigger
        let slot = &mut self.trigger_mem_usage_decreased[mem.0];
        if let Some((_, slot_after)) = slot {
            // further decrease memory used
            *slot_after = min(*slot_after, mem_space_used_before - value_size_bits);
        } else {
            // mark first memory decrease
            *slot = Some((mem_space_used_before, mem_space_used_before - value_size_bits));
        }

        // record action
        self.actions_taken.push(Action::Drop(ActionDrop {
            time: self.curr_time,
            value,
            mem,
        }));
    }

    fn claim_group(&mut self, group: Group, claim: GroupClaim) {
        let state = &mut self.state_group[group.0];
        assert!(state.is_none());
        *state = Some(claim);
    }

    fn release_group(&mut self, group: Group) {
        let state = &mut self.state_group[group.0];
        assert!(state.is_some());
        *state = None;

        let trigger = &mut self.trigger_group_free[group.0];
        assert!(!*trigger);
        *trigger = true;
    }

    pub fn do_action_wait(&mut self, problem: &Problem, time_end: Time) -> Result<(), ()> {
        // metadata
        assert!(time_end >= self.curr_time);
        assert!(time_end <= self.minimum_time);
        self.curr_time = time_end;
        self.actions_taken.push(Action::Wait(ActionWait { time: TimeRange { start: self.curr_time, end: time_end}}));

        // TODO avoid cloning these in the first place!
        // TODO do these before or after completing the operations? probably before, since them mem usage is highest
        self.clear_triggers();
        self.maybe_clear_tried(problem);

        let mut fail = false;

        // complete operations
        for group in problem.hardware.groups() {
            if let Some(action) = self.state_group[group.0] {
                if action.time().end > time_end {
                    // not done yet
                    continue
                }

                match action {
                    GroupClaim::Core(action) => {
                        let alloc_info = &problem.allocation_info[action.alloc.0];
                        for (&input_value, &input_mem) in zip_eq(&problem.graph.node_info[alloc_info.node.0].inputs, &alloc_info.input_memories) {
                            self.release_mem_value_read_lock(input_value, input_mem);
                        }
                        self.mark_mem_value_available(alloc_info.node, alloc_info.output_memory, ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: time_end });

                        // symmetry breaking
                        if !fail {
                            for &other_action in &self.actions_taken {
                                if let Action::Core(other_action) = other_action {
                                    if action.alloc.0 < other_action.alloc.0 && self.could_swap_core_actions(problem, action, other_action) {
                                        // println!("swapping");
                                        fail = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    GroupClaim::Channel(action) => {
                        let channel_info = &problem.hardware.channel_info[action.channel.0];
                        self.release_mem_value_read_lock(action.value, channel_info.mem_source);
                        self.mark_mem_value_available(action.value, channel_info.mem_dest, ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: time_end });
                    }
                }

                self.release_group(group);
            }
        }

        if fail {
            Err(())
        } else {
            Ok(())
        }
    }

    fn could_swap_core_actions(&self, problem: &Problem, curr_action: ActionCore, other_action: ActionCore) -> bool {
        let curr_info = &problem.allocation_info[curr_action.alloc.0];
        let other_info = &problem.allocation_info[other_action.alloc.0];

        // things to check:
        // * inputs of the earlier node are still available (tricky, they've probably been deleted because they're dead)
        // * the outputs of the earlier node have still not been used

        // TODO only allow exact time sizes for now, other arrangements are more questionable
        //  (eg. what if we create an earlier gap)
        // check that operands are still available (for the entire duration of the current action)
        //  TODO expand to "there was enough memory space during the entire period to keep the operands" to deal with dead operands

        // TODO less arbitrary symmetry check (we want invariance according to graph order)?

        let other_has_not_been_used = matches!(
            self.value_mem_availability(other_info.node, other_info.output_memory),
            Some(ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: _ })
        );

        let other_has_operands_now = problem.graph.node_info[other_info.node.0].inputs.iter().enumerate().all(|(i, &other_input)| {
            self.value_available_in_mem_now(other_input, other_info.input_memories[i], Some(self.curr_time - other_info.time))
        });

        let curr_has_operands_earlier = problem.graph.node_info[other_info.node.0].inputs.iter().enumerate().all(|(i, &curr_input)| {
            // TODO this is too strict, we don't need them to still be available now, just for the previous time range
            self.value_available_in_mem_now(curr_input, curr_info.input_memories[i], Some(other_action.time.end - curr_info.time))
        });

        let could_swap = other_has_operands_now
            && other_has_not_been_used
            && curr_has_operands_earlier
            && other_info.time == curr_info.time;

        could_swap
    }

    pub fn drop_dead_values(&mut self, problem: &Problem) -> Result<(), ()> {
        for mem in problem.hardware.memories() {
            // don't drop dead values in inf sized memories
            // * it's useless anyway
            // * this makes value liveness checking for past time ranges easier
            if problem.hardware.mem_info[mem.0].size_bits.is_none() {
                continue;
            }

            let used_before = self.mem_space_used(problem, mem);

            let mem_content = &mut self.state_memory_node[mem.0];
            let mut exit = false;

            mem_content.retain(|&value, &mut availability| {
                if let ValueState::AvailableNow { read_lock_count, read_count, since: _ } = availability {
                    let dead = self.value_remaining_unstarted_uses[value.0] == 0;

                    if read_lock_count > 0 || !dead {
                        // not (really) dead, keep
                        return true;
                    }

                    if read_count == 0 {
                        // dead but never read, prune this state
                        exit = true;
                        return true;
                    }

                    // dead and read at some point, drop
                    // TODO go back in time and drop at end of last usage? mostly for plotting clarity
                    self.actions_taken.push(Action::Drop(ActionDrop {
                        time: self.curr_time,
                        value,
                        mem,
                    }));

                    let live_count = &mut self.value_live_count[value.0];
                    assert!(*live_count > 0);
                    *live_count -= 1;

                    false
                } else {
                    true
                }
            });

            if exit {
                return Err(());
            }

            let used_after = self.mem_space_used(problem, mem);
            if used_before != used_after {
                assert!(used_after < used_before);

                let slot = &mut self.trigger_mem_usage_decreased[mem.0];
                let new = (used_before, used_after);

                match slot {
                    None => {
                        *slot = Some(new);
                    }
                    Some((_slot_before, slot_after)) => {
                        *slot_after = min(*slot_after, used_after);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn do_action_core(&mut self, problem: &Problem, alloc: Allocation) -> TimeRange {
        let alloc_info = &problem.allocation_info[alloc.0];
        let node_info = &problem.graph.node_info[alloc_info.node.0];

        // metadata
        let time_delta = alloc_info.time;
        let energy_delta = alloc_info.energy;

        let time_end = self.curr_time + time_delta;
        let time_range = TimeRange { start: self.curr_time, end: time_end };
        let action = ActionCore {
            time: time_range,
            alloc,
        };

        // real changes
        assert!(self.unstarted_nodes.remove(&alloc_info.node));
        for (&input_value, &input_mem) in zip_eq(&node_info.inputs, &alloc_info.input_memories) {
            let uses = &mut self.value_remaining_unstarted_uses[input_value.0];
            assert!(*uses >= 1);
            *uses -= 1;

            self.add_mem_value_read_lock(input_value, input_mem);
        }
        self.mark_mem_value_available(alloc_info.node, alloc_info.output_memory, ValueState::AvailableAtTime(time_end));
        self.claim_group(alloc_info.group, GroupClaim::Core(action));

        // common
        self.start_action_common(Action::Core(action), time_end, energy_delta);
        time_range
    }

    pub fn do_action_channel(&mut self, problem: &Problem, channel: Channel, value: Node) -> TimeRange {
        let channel_info = &problem.hardware.channel_info[channel.0];

        assert!(self.value_mem_availability(value, channel_info.mem_dest).is_none());

        // metadata
        let size_bits = problem.graph.node_info[value.0].size_bits;
        let time_delta = channel_info.cost.time_to_transfer(size_bits);
        let energy_delta = channel_info.cost.energy_to_transfer(size_bits);

        let time_end = self.curr_time + time_delta;
        let time_range = TimeRange { start: self.curr_time, end: time_end };
        let action = ActionChannel {
            time: time_range,
            channel,
            value,
        };

        // real changes
        self.add_mem_value_read_lock(value, channel_info.mem_source);
        self.mark_mem_value_available(value, channel_info.mem_dest, ValueState::AvailableAtTime(time_end));
        self.claim_group(channel_info.group, GroupClaim::Channel(action));

        // common
        self.start_action_common(Action::Channel(action), time_end, energy_delta);
        time_range
    }

    fn start_action_common(&mut self, action: Action, time_end: Time, energy_delta: Energy) {
        self.actions_taken.push(action);
        self.minimum_time = max(self.minimum_time, time_end);
        self.curr_energy += energy_delta;
    }

    fn maybe_clear_tried(&mut self, problem: &Problem) {
        // only drop iff any of
        //   * the action is not longer relevant and will no longer be tried anyway (eg. because the involved values are dead)
        //   * we decided to do something else with that group in that time slot (not necessarily at the same start, just overlapping)
        //   * the output value would not have fit in memory at some point between then and now
        let mut tried_transfers = std::mem::take(&mut self.tried_transfers);
        tried_transfers.retain(|&(channel, node), &mut time| {
            // dead
            if self.value_remaining_unstarted_uses[node.0] == 0 {
                return false;
            }

            // overlap
            let channel_info = &problem.hardware.channel_info[channel.0];
            if let Some(claim) = self.state_group[channel_info.group.0] {
                if claim.time().overlaps(time) {
                    return false;
                }
            }

            // fit
            {
                let mut dummy = self.new_trigger();
                if !dummy.check_mem_space_available(problem, channel_info.mem_dest, problem.graph.node_info[node.0].size_bits) {
                    return false;
                }
            }

            // keep
            true
        });
        assert!(self.tried_transfers.is_empty());
        self.tried_transfers = tried_transfers;

        let mut tried_allocs = std::mem::take(&mut self.tried_allocs);
        tried_allocs.retain(|&alloc, &mut time| {
            let alloc_info = &problem.allocation_info[alloc.0];

            // dead
            if !self.unstarted_nodes.contains(&alloc_info.node) {
                return false;
            }

            // overlap
            if let Some(claim) = self.state_group[alloc_info.group.0] {
                if claim.time().overlaps(time) {
                    return false;
                }
            }

            // fit
            {
                let mut dummy = self.new_trigger();
                if !dummy.check_mem_space_available(problem, alloc_info.output_memory, problem.graph.node_info[alloc_info.node.0].size_bits) {
                    return false;
                }
            }

            // keep
            true
        });
        assert!(self.tried_allocs.is_empty());
        self.tried_allocs = tried_allocs;
    }

    fn value_mem_dom_key_min(&self, value: Node, mem: Memory) -> i64 {
        if self.value_remaining_unstarted_uses[value.0] == 0 {
            // dead, best possible case
            return i64::MIN;
        }

        match self.value_mem_availability(value, mem) {
            // available now
            // TODO use current time here?
            Some(ValueState::AvailableNow { .. }) => 0,
            // available later
            // TODO subtract current time here?
            Some(ValueState::AvailableAtTime(time)) => (time - self.curr_time).0,
            // not even scheduled, worst case
            // TODO is that really true? what if we decide to schedule a state afterwards?
            None => i64::MAX,
        }
    }

    /// A state is better than another state if the second state can be reached from the first one
    /// by only taking useless or harmful actions. The exhaustive list of these actions is:
    /// * burn an arbitrary positive amount of energy
    /// * waste an arbitrary positive amount of time
    ///     * either right now (ie. increase curr_time)
    ///     * or as a future promise (ie. increase min_time)
    ///     * or as part of operations (ie. keep a core operation running for a bit longer then necessary)
    /// * delete values from memories
    /// * apply a problem automorphism (harmless)
    pub fn dom_key_min(&self, problem: &Problem, target: CostTarget) -> (SparseVec, usize) {
        let mut key = SparseVec::new();
        
        // TODO clean this up, maybe just have SparseVec.push that keeps an internal index?
        //   then we can't construct it efficiently any more through, and wastes a bit of memory
        let mut next_index = 0;
        let mut next_index = move || {
            let i = next_index;
            next_index += 1;
            i
        };

        // basics

        // value chosen to use approximately half of the bits
        // TODO using ordered tuples would be a lot better here
        const M: i64 = i32::MAX as i64;

        let minimum_time_left = self.minimum_time - self.curr_time;

        match target {
            CostTarget::Full => {
                key.push(next_index(), self.curr_time.0);
                key.push(next_index(), self.curr_energy.0);
                key.push(next_index(), minimum_time_left.0);
            }
            CostTarget::Time => {
                key.push(next_index(), self.curr_time.0 * M + self.curr_energy.0);
                key.push(next_index(), minimum_time_left.0 * M + self.curr_energy.0);
            }
            CostTarget::Energy => {
                key.push(next_index(), self.curr_energy.0 * M + self.curr_time.0);
                key.push(next_index(), self.curr_energy.0 * M + minimum_time_left.0);
            }
        }

        // group availability
        for group in problem.hardware.groups() {
            let v = match self.state_group[group.0] {
                // TODO go back to using current time here? that fails with actions that take zero time
                None => i64::MIN,
                Some(action) => (action.time().end - self.curr_time).0,
            };
            key.push(next_index(), v);
        }

        // value availability
        // TODO we should be able to fully skip this during comparisons if the value states don't match
        for mem in problem.hardware.memories() {
            for value in problem.graph.nodes() {
                key.push(next_index(), self.value_mem_dom_key_min(value, mem));
            }
        }

        // memory space (less used is better for memories with limited size)
        for mem in problem.hardware.memories() {
            if problem.hardware.mem_info[mem.0].size_bits.is_some() {
                key.push(next_index(), self.mem_space_used(problem, mem) as i64);
            }
        }

        (key, next_index())
    }
}

impl Trigger<'_> {
    #[must_use]
    fn result(&mut self, valid: bool, trigger: bool) -> bool {
        // we can't assert valid if triggered, here, the condition might have become false already
        self.triggered |= trigger;
        valid
    }

    #[must_use]
    pub fn check_group_free(&mut self, group: Group) -> bool {
        self.result(
            self.state.state_group[group.0].is_none(),
            self.state.trigger_group_free[group.0],
        )
    }

    #[must_use]
    pub fn check_mem_space_available(&mut self, problem: &Problem, mem: Memory, size_bits: u64) -> bool {
        let mem_size = problem.hardware.mem_info[mem.0].size_bits;

        match mem_size {
            None => {
                // inf memory always fits but never triggers
                self.result(true, false)
            }
            Some(mem_size) => {
                // fits if fits
                //  note: this has to be separate from the trigger calculation, we might have made even more space since then
                //  TODO is that note actually true?
                let used = self.state.mem_space_used(problem, mem);
                let fits = used + size_bits <= mem_size;

                // trigger if fit changed
                let triggered = if let Some((before, after)) = self.state.trigger_mem_usage_decreased[mem.0] {
                    let fit_before = before + size_bits <= mem_size;
                    let fit_after = after + size_bits <= mem_size;
                    fit_after && !fit_before
                } else {
                    false
                };

                self.result(fits, triggered)
            }
        }
    }

    #[must_use]
    pub fn check_mem_value_available(&mut self, mem: Memory, value: Node) -> bool {
        self.result(
            self.state.value_available_in_mem_now(value, mem, None),
            self.state.trigger_value_mem_available[value.0][mem.0],
        )
    }

    #[must_use]
    pub fn check_mem_value_no_availability(&mut self, mem: Memory, value: Node) -> bool {
        // TODO use drop trigger here? generally think about how this should interact with triggering
        self.result(
            self.state.value_mem_availability(value, mem).is_none(),
            false,
        )
    }

    #[must_use]
    pub fn check_mem_value_unlocked_and_read(&mut self, mem: Memory, value: Node) -> bool {
        let valid = match self.state.value_mem_availability(value, mem) {
            None | Some(ValueState::AvailableAtTime(_)) => false,
            Some(ValueState::AvailableNow { read_lock_count, read_count, since: _ }) => {
                read_lock_count == 0 && read_count > 0
            }
        };
        self.result(
            valid,
            self.state.trigger_value_mem_unlocked_or_read[value.0][mem.0],
        )
    }

    #[must_use]
    pub fn check_not_single_live_instance(&mut self, value: Node) -> bool {
        let live_count = self.state.value_live_count[value.0];
        assert!(live_count > 0);
        self.result(live_count > 1, self.state.trigger_value_live_count_increased[value.0])
    }

    #[must_use]
    pub fn was_triggered(self) -> bool {
        assert!(self.valid);
        self.triggered
    }
}

impl Dominance for Cost {
    type Aux = CostTarget;
    fn dominance(&self, other: &Self, target: &CostTarget) -> DomDir {
        let mut dom = DomBuilder::new(self, other);

        match *target {
            CostTarget::Time => {
                dom.minimize(|s| (s.time, s.energy));
            }
            CostTarget::Energy => {
                dom.minimize(|s| (s.energy, s.time));
            }
            CostTarget::Full => {
                dom.minimize(|s| s.time);
                dom.minimize(|s| s.energy);
            }
        }

        dom_early_check!(dom);
        dom.finish()
    }
}

impl GroupClaim {
    fn time(&self) -> TimeRange {
        match self {
            GroupClaim::Core(a) => a.time,
            GroupClaim::Channel(a) => a.time,
        }
    }
}
