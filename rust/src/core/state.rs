use std::cmp::max;
use std::collections::{HashMap, HashSet};

use itertools::{enumerate, Itertools, zip_eq};

use crate::core::frontier::{DomBuilder, DomDir, Dominance};
use crate::core::new_frontier::SparseVec;
use crate::core::problem::{Allocation, CostTarget, Group, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionDrop, Timed, TimeRange};
use crate::core::wrapper::{Energy, Time, TypedVec};
use crate::dom_early_check;

#[derive(Clone)]
pub struct State {
    // minimal state
    pub actions_taken: Vec<Timed<Action>>,

    // memoized information
    pub curr_time: Time,
    pub curr_energy: Energy,
    pub minimum_time: Time,

    pub state_group: TypedVec<Group, Option<Timed<GroupClaim>>>,
    // TODO replace hashmap with TypedVec? it's usually sparse though
    pub state_memory_node: TypedVec<Memory, HashMap<Node, ValueState>>,
    pub value_live_count: TypedVec<Node, usize>,

    pub unstarted_nodes: HashSet<Node>,
    pub value_remaining_unstarted_uses: TypedVec<Node, u32>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SkippedDropInfo {
    pub time: Time,
    pub read_count: u64,
}

// TODO rename
// TODO store only minimal information here
#[derive(Debug, Copy, Clone)]
pub enum GroupClaim {
    Core(Allocation),
    Channel(ActionChannel),
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct Cost {
    pub time: Time,
    pub energy: Energy,
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

        // precompute
        let unstarted_nodes = graph.nodes.keys().filter(|n| !graph.inputs.contains(n)).collect();

        let mut value_remaining_unstarted_uses = TypedVec::full_like(0, &graph.nodes);
        for node_info in graph.nodes.values() {
            for &x in &node_info.inputs {
                value_remaining_unstarted_uses[x] += 1;
            }
        }
        for &x in &graph.outputs {
            value_remaining_unstarted_uses[x] += 1;
        }

        let mut state_memory_node = TypedVec::full_like(HashMap::new(), &hardware.memories);
        let mut value_live_count = TypedVec::full_like(0, &graph.nodes);

        for (&value, &mem) in zip_eq(&graph.inputs, &problem.input_placements) {
            let state = ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: Time(0) };
            let prev = state_memory_node[mem].insert(value, state);
            assert!(prev.is_none());
            value_live_count[value] += 1;
        }

        // construct
        State {
            actions_taken: vec![],
            curr_time: Time(0),
            curr_energy: Energy(0),
            minimum_time: Time(0),
            state_group: TypedVec::full_like(None, &hardware.groups),
            state_memory_node,
            value_live_count,
            unstarted_nodes,
            value_remaining_unstarted_uses,
        }
    }

    pub fn assert_valid(&self, problem: &Problem) {
        // aliases
        let graph = &problem.graph;
        let hardware = &problem.hardware;

        // checks
        let mut expected_state_memory_read_lock_count = TypedVec::full_like(HashMap::new(), &hardware.memories);
        let mut expected_value_live_count = TypedVec::full_like(0, &graph.nodes);

        for (_, state) in &self.state_group {
            match *state {
                None => {},
                Some(Timed { time, inner: GroupClaim::Core(alloc) }) => {
                    let alloc = &problem.allocations[alloc];
                    for (&input, &mem) in zip_eq(&graph.nodes[alloc.node].inputs, &alloc.input_memories) {
                        *expected_state_memory_read_lock_count[mem].entry(input).or_insert(0) += 1;
                    }
                    assert_eq!(
                        self.state_memory_node[alloc.output_memory].get(&alloc.node).unwrap(),
                        &ValueState::AvailableAtTime(time.end)
                    );
                }
                Some(Timed { time, inner: GroupClaim::Channel(action) }) => {
                    let channel_info = &problem.hardware.channels[action.channel];
                    *expected_state_memory_read_lock_count[channel_info.mem_source].entry(action.value).or_insert(0) += 1;
                    assert_eq!(
                        self.state_memory_node[channel_info.mem_dest].get(&action.value).unwrap(),
                        &ValueState::AvailableAtTime(time.end)
                    );
                }
            }
        }

        for mem in hardware.memories.keys() {
            let expected = &expected_state_memory_read_lock_count[mem];
            let actual = &self.state_memory_node[mem];

            for (&node, &node_actual) in actual {
                expected_value_live_count[node] += 1;

                match node_actual {
                    ValueState::AvailableNow { read_lock_count, read_count: _, since: _ } => {
                        assert_eq!(expected.get(&node).copied().unwrap_or(0), read_lock_count)
                    }
                    ValueState::AvailableAtTime(_) => assert!(!expected.contains_key(&node)),
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
            problem.allocations.values()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.energy)
                .min().unwrap()
        }).sum::<Energy>();

        let min_additional_times = self.unstarted_nodes.iter().map(|node| {
            problem.allocations.values()
            .filter(|alloc| alloc.node == *node)
            .map(|alloc| alloc.time)
            .min().unwrap()
        }).collect_vec();

        // let mut groups_with_any_allocs = HashSet::new();
        // for alloc_info in problem.allocations.values() {
        //     if self.unstarted_nodes.contains(&alloc_info.node) {
        //         groups_with_any_allocs.insert(alloc_info.group);
        //     }
        // }

        let min_additional_time_single = min_additional_times.iter().copied().min().unwrap_or(Time(0));
        // let min_additional_time_div = min_additional_times.iter().copied().sum::<Time>().ceil_div(max(1, groups_with_any_allocs.len() as i64));

        // TODO why does using a better bound make things slower?
        // let min_additional_time = max(min_additional_time_single, min_additional_time_div);
        let min_additional_time = min_additional_time_single;

        Cost {
            time: max(self.minimum_time, self.curr_time + min_additional_time),
            energy: self.curr_energy + min_additional_energy,
        }
    }

    // TODO use this function in a bunch more places
    pub fn value_mem_availability(&self, value: Node, mem: Memory) -> Option<ValueState> {
        self.state_memory_node[mem].get(&value).copied()
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
        self.state_group.values().all(|s| s.is_none())
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
        let used = self.state_memory_node[mem].iter()
            .map(|(&value, _)| problem.graph.nodes[value].size_bits)
            .sum();

        if let Some(mem_size_bits) = problem.hardware.memories[mem].size_bits {
            assert!(used <= mem_size_bits);
        }

        used
    }

    pub fn first_done_time(&self) -> Option<Time> {
        self.state_group.values().filter_map(|a| a.map(|a| a.time.end)).min()
    }

    pub fn is_dead_value(&self, value: Node) -> bool {
        self.value_remaining_unstarted_uses[value] == 0
    }

    #[must_use]
    pub fn clone_and_then(&self, mut f: impl FnMut(&mut Self)) -> Self {
        let mut next = self.clone();
        f(&mut next);
        next
    }

    fn add_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem].get_mut(&value).unwrap();

        match availability {
            ValueState::AvailableNow { read_lock_count, read_count, since: _ } => {
                *read_lock_count += 1;
                *read_count += 1;
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn release_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem].get_mut(&value).unwrap();
        match availability {
            ValueState::AvailableNow { read_lock_count, read_count: _, since: _ } => {
                assert!(*read_lock_count > 0);
                *read_lock_count -= 1;
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn mark_mem_value_available(&mut self, value: Node, mem: Memory, availability: ValueState) {
        let prev = self.state_memory_node[mem].insert(value, availability);

        match availability {
            ValueState::AvailableNow { .. } => {
                assert!(matches!(prev, Some(ValueState::AvailableAtTime(_))));
            }
            ValueState::AvailableAtTime(_) => {
                assert_eq!(prev, None, "Value {value:?} in mem {mem:?} already has availability, trying to insert {availability:?}");
                self.value_live_count[value] += 1;
            }
        }
    }

    pub fn drop_value(&mut self, problem: &Problem, mem: Memory, value: Node) {
        // get value before removal
        let mem_space_used_before = self.mem_space_used(problem, mem);

        let mem_size_bits = problem.hardware.memories[mem].size_bits;
        if let Some(mem_size_bits) = mem_size_bits {
            assert!(mem_space_used_before <= mem_size_bits);
        }

        // remove
        let prev = self.state_memory_node[mem].remove(&value);
        assert!(matches!(prev, Some(ValueState::AvailableNow { read_lock_count: 0, read_count: _, since: _ })));
        assert!(prev.is_some());

        let live_count = &mut self.value_live_count[value];
        assert!(*live_count > 0);
        *live_count -= 1;

        self.push_action(problem, Action::Drop(ActionDrop { mem, value }));
    }

    fn claim_group(&mut self, group: Group, time: TimeRange, claim: GroupClaim) {
        let state = &mut self.state_group[group];
        assert!(state.is_none());
        *state = Some(Timed { time, inner: claim });
    }

    fn release_group(&mut self, group: Group) {
        let state = &mut self.state_group[group];
        assert!(state.is_some());
        *state = None;
    }

    pub fn could_start_action_now(&self, problem: &Problem, action: Action) -> bool {
        // TODO write all checks with && instead of with if/return
        // TODO split up in sub-functions
        // TODO re-order checks to get this to be as fast as possible

        match action {
            Action::Wait(_) => {
                // wait actions can always run
                true
            },
            Action::Core(alloc) => {
                let alloc_info = &problem.allocations[alloc];
                let node = alloc_info.node;
                let node_info = &problem.graph.nodes[node];

                // each node can only be computed once
                // TODO relax this? maybe recomputing is more efficient!
                if self.is_node_started(node) {
                    return false;
                }

                // TODO add after limiting again?
                //   does the achievement contain enough information that this should be allowed?
                // for &after in &node_info.start_after {
                //     if !self.is_node_started(after) {
                //         return false;
                //     }
                // }

                if !self.check_group_free(alloc_info.group) {
                    return false;
                }

                if !self.check_mem_space_available(problem, alloc_info.output_memory, node_info.size_bits) {
                    return false;
                }

                for (&input_value, &input_mem) in zip_eq(&node_info.inputs, &alloc_info.input_memories) {
                    if !self.value_available_in_mem_now(input_value, input_mem, None) {
                        return false;
                    }
                }

                true
            }
            Action::Channel(action) => {
                let ActionChannel { channel, value } = action;

                let value_info = &problem.graph.nodes[value];
                let channel_info = &problem.hardware.channels[channel];

                // don't bother copying dead values around
                if self.is_dead_value(value) {
                    return false;
                }

                if !self.value_available_in_mem_now(value, channel_info.mem_source, None) {
                    return false;
                }

                if !self.check_group_free(channel_info.group) {
                    return false;
                }

                if self.value_mem_availability(value, channel_info.mem_dest).is_some() {
                    return false;
                }

                if !self.check_mem_space_available(problem, channel_info.mem_dest, value_info.size_bits) {
                    return false;
                }

                true
            }
            Action::Drop(action) => {
                let ActionDrop { mem, value } = action;

                match self.state_memory_node[mem].get(&value) {
                    // can't drop value that's not even available
                    None | Some(ValueState::AvailableAtTime(_)) => {
                        return false;
                    }
                    Some(&ValueState::AvailableNow { read_lock_count, read_count: _, since: _ }) => {
                        // can't drop locked value
                        if read_lock_count != 0 {
                            return false;
                        }

                        // TODO instead of this assert, prune states with dead values that are still being copied or calculated
                        assert!(!self.is_dead_value(value), "dead values should have been dropped already");

                        // don't drop last instance of live value
                        if !self.check_not_single_live_instance(value) {
                            return false;
                        }

                        true
                    }
                }
            }
        }
    }

    fn push_action(&mut self, problem: &Problem, action: Action) -> TimeRange {
        let time = TimeRange {
            start: self.curr_time,
            end: self.curr_time + action.time(problem),
        };

        self.actions_taken.push(Timed { time, inner: action });

        self.minimum_time = max(self.minimum_time, time.end);
        self.curr_energy += action.energy(problem);

        time
    }

    // TODO make other functions non-public?
    pub fn do_action(&mut self, problem: &Problem, action: Action) -> TimeRange {
        match action {
            Action::Wait(delta) => self.do_action_wait(problem, delta + self.curr_time),
            Action::Core(alloc) => self.do_action_core(problem, alloc),
            Action::Channel(action) => self.do_action_channel(problem, action),
            Action::Drop(action) => self.do_action_drop(problem, action),
        }
    }

    // TODO change this to consistently be delta
    pub fn do_action_wait(&mut self, problem: &Problem, time_end: Time) -> TimeRange {
        // metadata
        assert!(time_end >= self.curr_time);
        assert!(time_end <= self.minimum_time);
        let time_range = self.push_action(problem, Action::Wait(time_end - self.curr_time));
        self.curr_time = time_end;

        // complete operations
        for group in problem.hardware.groups.keys() {
            if let Some(claim) = self.state_group[group] {
                if claim.time.end > time_end {
                    // not done yet
                    continue
                }

                // TODO fix available time to be the end of the claim, not the current time
                match claim.inner {
                    GroupClaim::Core(alloc) => {
                        let alloc_info = &problem.allocations[alloc];
                        for (&input_value, &input_mem) in zip_eq(&problem.graph.nodes[alloc_info.node].inputs, &alloc_info.input_memories) {
                            self.release_mem_value_read_lock(input_value, input_mem);
                        }
                        self.mark_mem_value_available(alloc_info.node, alloc_info.output_memory, ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: time_end });
                    }
                    GroupClaim::Channel(action) => {
                        let channel_info = &problem.hardware.channels[action.channel];
                        self.release_mem_value_read_lock(action.value, channel_info.mem_source);
                        self.mark_mem_value_available(action.value, channel_info.mem_dest, ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: time_end });
                    }
                }

                self.release_group(group);
            }
        }

        time_range
    }

    // TODO use curr time or time from the action?
    fn could_swap_core_actions(&self, problem: &Problem, curr_action: Timed<Allocation>, other_action: Timed<Allocation>) -> bool {
        let curr_info = &problem.allocations[curr_action.inner];
        let other_info = &problem.allocations[other_action.inner];
        assert_eq!(curr_info.group, other_info.group);

        // things to check:
        // * inputs of the earlier node are still available (tricky, they've probably been deleted because they're dead)
        // * the outputs of the earlier node have still not been used

        // TODO only allow exact time sizes for now, other arrangements are more questionable
        //  (eg. what if we create an earlier gap)
        // check that operands are still available (for the entire duration of the current action)
        //  TODO expand to "there was enough memory space during the entire period to keep the operands" to deal with dead operands

        // TODO less arbitrary symmetry check (we want invariance according to graph order)?
        // TODO revisit older actions if their outputs still have not been used, maybe we can prune even more

        let other_has_not_been_used = matches!(
            self.value_mem_availability(other_info.node, other_info.output_memory),
            Some(ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: _ })
        );

        // TODO this is too strict, if there was enough memory to keep them alive during the entire time that's enough
        let other_has_operands_now = zip_eq(&problem.graph.nodes[other_info.node].inputs, &other_info.input_memories).all(|(&input_node, &input_mem)| {
            self.value_available_in_mem_now(input_node, input_mem, Some(self.curr_time - other_info.time))
        });

        let curr_has_operands_earlier = zip_eq(&problem.graph.nodes[curr_info.node].inputs, &curr_info.input_memories).all(|(&input_node, &input_mem)| {
            // TODO this is too strict, we don't need them to still be available now, just for the previous time range
            self.value_available_in_mem_now(input_node, input_mem, Some(other_action.time.end - curr_info.time))
        });

        let could_swap = other_has_operands_now
            && other_has_not_been_used
            && curr_has_operands_earlier
            && other_info.time == curr_info.time;

        could_swap
    }

    // TODO also swap across different channels/groups and maybe even across core-channel?
    // TODO use curr time or time from the action?
    fn could_swap_channel_actions(&self, problem: &Problem, curr_action: Timed<ActionChannel>, other_action: Timed<ActionChannel>) -> bool {
        let curr_info = &problem.hardware.channels[curr_action.inner.channel];
        let other_info = &problem.hardware.channels[other_action.inner.channel];
        assert_eq!(curr_info.group, other_info.group);

        let other_has_not_been_used = matches!(
            self.value_mem_availability(other_action.inner.value, other_info.mem_dest),
            Some(ValueState::AvailableNow { read_lock_count: 0, read_count: 0, since: _ })
        );

        // TODO this is too strict, if there was enough memory to keep them alive during the entire time that's enough
        let other_has_operands_now = self.value_available_in_mem_now(other_action.inner.value, other_info.mem_source, Some(self.curr_time - other_action.time.len()));
        let curr_has_operands_earlier = self.value_available_in_mem_now(curr_action.inner.value, curr_info.mem_source, Some(other_action.time.end - curr_action.time.len()));

        let could_swap = other_has_operands_now
            && other_has_not_been_used
            && curr_has_operands_earlier
            && other_action.time.len() == curr_action.time.len();

        could_swap
    }

    pub fn drop_dead_values(&mut self, problem: &Problem) {
        let mut drops = vec![];

        for (mem, info) in &self.state_memory_node {
            for (&value, &availability) in info {
                match availability {
                    ValueState::AvailableNow { read_lock_count, read_count: _, since: _ } => {
                        if read_lock_count == 0 && self.value_remaining_unstarted_uses[value] == 0 {
                            // TODO do back in time and drop after last usage? mostly to get nicer memory plots
                            drops.push(ActionDrop { mem, value });
                        }
                    }
                    ValueState::AvailableAtTime(_) => {},
                }
            };
        }

        for a in drops {
            self.do_action_drop(problem, a);
        }
    }

    fn do_action_drop(&mut self, problem: &Problem, action: ActionDrop) -> TimeRange {
        let time_range = self.push_action(problem, Action::Drop(action));
        let ActionDrop { mem, value } = action;

        let prev = self.state_memory_node[mem].remove(&value);
        assert!(
            matches!(prev, Some(ValueState::AvailableNow { read_lock_count: 0, read_count: _, since: _ })),
            "Can't do drop {:?} with availability {:?}",
            action, prev,
        );

        let live_count = &mut self.value_live_count[value];
        assert!(*live_count > 0);
        *live_count -= 1;

        time_range
    }

    pub fn do_action_core(&mut self, problem: &Problem, alloc: Allocation) -> TimeRange {
        let time_range = self.push_action(problem, Action::Core(alloc));

        let alloc_info = &problem.allocations[alloc];
        let node_info = &problem.graph.nodes[alloc_info.node];
        assert!(self.unstarted_nodes.remove(&alloc_info.node));

        for (&input_value, &input_mem) in zip_eq(&node_info.inputs, &alloc_info.input_memories) {
            let uses = &mut self.value_remaining_unstarted_uses[input_value];
            assert!(*uses >= 1);
            *uses -= 1;

            self.add_mem_value_read_lock(input_value, input_mem);
        }
        self.mark_mem_value_available(alloc_info.node, alloc_info.output_memory, ValueState::AvailableAtTime(time_range.end));
        self.claim_group(alloc_info.group, time_range, GroupClaim::Core(alloc));

        time_range
    }

    pub fn do_action_channel(&mut self, problem: &Problem, action: ActionChannel) -> TimeRange {
        let time_range = self.push_action(problem, Action::Channel(action));

        let ActionChannel { channel, value } = action;
        let channel_info = &problem.hardware.channels[channel];
        assert!(self.value_mem_availability(value, channel_info.mem_dest).is_none());

        self.add_mem_value_read_lock(value, channel_info.mem_source);
        self.mark_mem_value_available(value, channel_info.mem_dest, ValueState::AvailableAtTime(time_range.end));
        self.claim_group(channel_info.group, time_range, GroupClaim::Channel(action));

        time_range
    }

    fn value_mem_dom_key_min(&self, value: Node, mem: Memory) -> i64 {
        // TODO re-enable this after fixing the dropping bug during state reconstruction
        // if self.is_dead_value(value) {
        //     // dead, best possible case
        //     return i64::MIN;
        // }

        match self.value_mem_availability(value, mem) {
            // available now
            // TODO use current time here?
            Some(ValueState::AvailableNow { .. }) => 0,
            // available later
            // TODO subtract current time here?
            Some(ValueState::AvailableAtTime(time)) => {
                assert!(time > self.curr_time);
                (time - self.curr_time).0
            }
            // not even scheduled, worst case
            // TODO is that really true? what if we decide to schedule a state afterwards?
            None => i64::MAX,
        }
    }

    pub fn achievement(&self, problem: &Problem) -> Vec<i64> {
        let mut result = vec![];

        // group availability
        for group in problem.hardware.groups.keys() {
            let v = match self.state_group[group] {
                // TODO go back to using current time here? that fails with actions that take zero time
                None => i64::MIN,
                Some(action) => (action.time.end - self.curr_time).0,
            };
            result.push(v);
        }

        // value availability
        // TODO we should be able to fully skip this during comparisons if the value states don't match
        for mem in problem.hardware.memories.keys() {
            for value in problem.graph.nodes.keys() {
                result.push(self.value_mem_dom_key_min(value, mem));
            }
        }

        result
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
        // TODO switch to making all normal time and energy values i32 so they're guaranteed to fit here? or i128
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
        for group in problem.hardware.groups.keys() {
            let v = match self.state_group[group] {
                // TODO go back to using current time here? that fails with actions that take zero time
                None => i64::MIN,
                Some(action) => (action.time.end - self.curr_time).0,
            };
            key.push(next_index(), v);
        }

        // value availability
        // TODO we should be able to fully skip this during comparisons if the value states don't match
        for mem in problem.hardware.memories.keys() {
            for value in problem.graph.nodes.keys() {
                key.push(next_index(), self.value_mem_dom_key_min(value, mem));
            }
        }

        // memory space (less used is better for memories with limited size)
        for (mem, mem_info) in &problem.hardware.memories {
            if mem_info.size_bits.is_some() {
                key.push(next_index(), self.mem_space_used(problem, mem) as i64);
            }
        }

        (key, next_index())
    }

    // TODO rename and clean up all of the following checks
    #[must_use]
    pub fn check_group_free(&self, group: Group) -> bool {
        self.state_group[group].is_none()
    }

    #[must_use]
    pub fn is_node_started(&self, node: Node) -> bool {
        !self.unstarted_nodes.contains(&node)
    }

    #[must_use]
    pub fn check_mem_space_available(&self, problem: &Problem, mem: Memory, size_bits: u64) -> bool {
        match problem.hardware.memories[mem].size_bits {
            None => true,
            Some(mem_size) => {
                let used = self.mem_space_used(problem, mem);
                used + size_bits <= mem_size
            }
        }
    }

    #[must_use]
    pub fn check_not_single_live_instance(&self, value: Node) -> bool {
        let live_count = self.value_live_count[value];
        assert!(live_count > 0);
        live_count > 1
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

impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, rhs: Self) -> Self::Output {
        Cost { time: self.time + rhs.time, energy: self.energy + rhs.energy }
    }
}

impl std::ops::Sub for Cost {
    type Output = Cost;

    fn sub(self, rhs: Self) -> Self::Output {
        Cost { time: self.time - rhs.time, energy: self.energy - rhs.energy }
    }
}
