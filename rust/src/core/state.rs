use std::collections::{HashMap, HashSet};

use itertools::{chain, enumerate, zip_eq};

use crate::core::frontier::{DomBuilder, DomDir, Dominance};
use crate::core::problem::{Allocation, Channel, Group, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionCore, ActionWait, TimeRange};
use crate::dom_early_check;
use crate::util::mini::{IterFloatExt, max_f64};

#[derive(Clone)]
pub struct State {
    // minimal state
    pub actions_taken: Vec<Action>,

    // memoized information
    pub curr_time: f64,
    pub curr_energy: f64,
    pub minimum_time: f64,

    pub state_group: Vec<Option<GroupClaim>>,
    pub state_memory_node: Vec<HashMap<Node, ValueState>>,

    pub unstarted_nodes: HashSet<Node>,
    pub value_remaining_unstarted_uses: Vec<u32>,

    // triggers
    // TODO add value dropped trigger
    pub trigger_everything: bool,
    pub trigger_group_free: Vec<bool>,
    pub trigger_value_mem_available: Vec<Vec<bool>>,
    pub trigger_value_mem_unlocked: Vec<Vec<bool>>,
    pub trigger_mem_usage_decreased: Vec<(u64, u64)>,

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
    pub time: f64,
    pub energy: f64,
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
    },
    AvailableAtTime(f64),
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
        for (&value, &mem) in zip_eq(&graph.inputs, &problem.input_placements) {
            let prev = state_memory_node[mem.0].insert(value, ValueState::AvailableNow { read_lock_count: 0, read_count: 0 });
            assert!(prev.is_none());
        }

        // construct
        State {
            actions_taken: vec![],
            curr_time: 0.0,
            curr_energy: 0.0,
            minimum_time: 0.0,
            state_group: vec![None; group_count],
            state_memory_node,
            unstarted_nodes,
            value_remaining_unstarted_uses,
            trigger_everything: true,
            trigger_group_free: vec![false; group_count],
            trigger_value_mem_available: vec![vec![false; mem_count]; node_count],
            trigger_value_mem_unlocked: vec![vec![false; mem_count]; node_count],
            trigger_mem_usage_decreased: vec![],
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
                match node_actual {
                    ValueState::AvailableNow { read_lock_count, read_count: _ } => {
                        assert_eq!(expected.get(node).copied().unwrap_or(0), read_lock_count)
                    }
                    ValueState::AvailableAtTime(_) => assert!(!expected.contains_key(node)),
                }
            }

            for (node, &expected_read_lock_count) in expected {
                match *actual.get(node).unwrap() {
                    ValueState::AvailableNow { read_count: _, read_lock_count  } => {
                        assert_eq!(read_lock_count, expected_read_lock_count);
                    }
                    ValueState::AvailableAtTime(_) => panic!(),
                }
            }
        }
    }

    pub fn best_case_cost(&self, problem: &Problem) -> Cost {
        // TODO precompute these values for each node
        let min_additional_energy = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.energy)
                .min_f64().unwrap()
        }).sum::<f64>();

        let min_additional_time = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.time)
                .min_f64().unwrap()
        }).sum::<f64>();

        Cost {
            time: max_f64(self.minimum_time, self.curr_time + min_additional_time),
            energy: self.curr_energy + min_additional_energy,
        }
    }

    // TODO use this function in a bunch more places
    pub fn value_mem_availability(&self, value: Node, mem: Memory) -> Option<ValueState> {
        self.state_memory_node[mem.0].get(&value).copied()
    }

    pub fn value_available_in_mem_now(&self, value: Node, mem: Memory) -> bool {
        match self.value_mem_availability(value, mem) {
            Some(ValueState::AvailableNow { .. }) => true,
            Some(ValueState::AvailableAtTime(_)) => false,
            None => false,
        }
    }

    pub fn is_done(&self, problem: &Problem) -> bool {
        for (output_i, &mem) in enumerate(&problem.output_placements) {
            let output = problem.graph.outputs[output_i];
            if !self.value_available_in_mem_now(output, mem) {
                return false;
            }
        }
        true
    }

    pub fn current_cost(&self) -> Cost {
        Cost {
            time: self.curr_time,
            energy: self.curr_energy,
        }
    }

    pub fn mem_space_used(&self, problem: &Problem, mem: Memory) -> u64 {
        // TODO update incrementally
        self.state_memory_node[mem.0].iter()
            .map(|(&value, _)| problem.graph.node_info[value.0].size_bits)
            .sum()
    }

    pub fn new_trigger(&self) -> Trigger {
        Trigger {
            state: self,
            triggered: self.trigger_everything,
            valid: true,
        }
    }

    pub fn first_done_time(&self) -> Option<f64> {
        self.state_group.iter().filter_map(|a| a.map(|a| a.time().end)).min_f64()
    }

    #[must_use]
    pub fn clone_and_then(&self, mut f: impl FnMut(&mut Self)) -> Self {
        let mut next = self.clone();
        f(&mut next);
        next
    }

    pub fn clear_triggers(&mut self) {
        self.trigger_everything = false;
        self.trigger_group_free.iter_mut().for_each(|x| *x = false);
        self.trigger_value_mem_available.iter_mut().for_each(|x| x.iter_mut().for_each(|x| *x = false));
        self.trigger_value_mem_unlocked.iter_mut().for_each(|x| x.iter_mut().for_each(|x| *x = false));
        self.trigger_mem_usage_decreased.clear();
    }

    fn add_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();

        match availability {
            ValueState::AvailableNow { read_lock_count, read_count } => {
                *read_lock_count += 1;
                *read_count += 1;
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn release_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();
        match availability {
            ValueState::AvailableNow { read_lock_count, read_count: _ } => {
                assert!(*read_lock_count > 0);
                *read_lock_count -= 1;
                if *read_lock_count == 0 {
                    self.trigger_value_mem_unlocked[value.0][mem.0] = true;
                }
            }
            ValueState::AvailableAtTime(_) => panic!(),
        }
    }

    fn mark_mem_value_available(&mut self, value: Node, mem: Memory, availability: ValueState) {
        let prev = self.state_memory_node[mem.0].insert(value, availability);

        match availability {
            ValueState::AvailableNow { .. } => {
                assert!(matches!(prev, Some(ValueState::AvailableAtTime(_))));

                let trigger = &mut self.trigger_value_mem_available[value.0][mem.0];
                assert!(!*trigger);
                *trigger = true;
            }
            ValueState::AvailableAtTime(_) => {
                assert_eq!(prev, None);
            }
        }
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

    pub fn do_action_wait(&mut self, problem: &Problem, time_end: f64) {
        // metadata
        assert!(time_end >= self.curr_time);
        assert!(time_end <= self.minimum_time);
        self.curr_time = time_end;
        self.actions_taken.push(Action::Wait(ActionWait { time: TimeRange { start: self.curr_time, end: time_end}}));

        // TODO avoid cloning these in the first place!
        // TODO do these before or after completing the operations? probably before, since them mem usage is highest
        self.clear_triggers();
        self.maybe_clear_tried(problem);

        // complete core operations
        for group in problem.hardware.groups() {
            if let Some(action) = self.state_group[group.0] {
                if action.time().end > time_end {
                    // not done yet
                    continue
                }

                match action {
                    GroupClaim::Core(action) => {
                        let alloc = &problem.allocation_info[action.alloc.0];
                        for (&input_value, &input_mem) in zip_eq(&problem.graph.node_info[alloc.node.0].inputs, &alloc.input_memories) {
                            self.release_mem_value_read_lock(input_value, input_mem);
                        }
                        self.mark_mem_value_available(alloc.node, alloc.output_memory, ValueState::AvailableNow { read_lock_count: 0, read_count: 0 });
                    }
                    GroupClaim::Channel(action) => {
                        let channel_info = &problem.hardware.channel_info[action.channel.0];
                        self.release_mem_value_read_lock(action.value, channel_info.mem_source);
                        self.mark_mem_value_available(action.value, channel_info.mem_dest, ValueState::AvailableNow { read_lock_count: 0, read_count: 0 });
                    }
                }

                self.release_group(group);
            }
        }
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

        // metadata
        let size_bits = problem.graph.node_info[value.0].size_bits;
        let time_delta = channel_info.time_to_transfer(size_bits);
        let energy_delta = channel_info.energy_to_transfer(size_bits);

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

    fn start_action_common(&mut self, action: Action, time_end: f64, energy_delta: f64) {
        self.actions_taken.push(action);
        self.minimum_time = f64::max(self.minimum_time, time_end);
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

    fn value_mem_dom_key_min(&self, value: Node, mem: Memory) -> f64 {
        if self.value_remaining_unstarted_uses[value.0] == 0 {
            // dead, best possible case
            return f64::NEG_INFINITY;
        }

        match self.value_mem_availability(value, mem) {
            // available now
            // TODO use current time here?
            Some(ValueState::AvailableNow { .. }) => 0.0,
            // available later
            // TODO subtract current time here?
            Some(ValueState::AvailableAtTime(time)) => time,
            // not even scheduled, worst case
            // TODO is that really true? what if we decide to schedule a state afterwards?
            None => f64::INFINITY,
        }
    }

    pub fn dom_key_min(&self, problem: &Problem) -> Vec<f64> {
        let mut key = vec![];

        // basics
        key.push(self.curr_time);
        key.push(self.curr_energy);
        key.push(self.minimum_time);

        // group availability
        for group in problem.hardware.groups() {
            key.push(match self.state_group[group.0] {
                None => self.curr_time,
                Some(action) => action.time().end,
            });
        }

        // value availability
        for mem in problem.hardware.memories() {
            for value in problem.graph.nodes() {
                key.push(self.value_mem_dom_key_min(value, mem));
            }
        }

        key
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
                // fits if fits, trigger if fit changed
                let used = self.state.mem_space_used(problem, mem);
                let (before, after) = self.state.trigger_mem_usage_decreased[mem.0];

                self.result(
                    used + size_bits <= mem_size,
                    before + size_bits > mem_size && after + size_bits < mem_size,
                )
            }
        }
    }

    #[must_use]
    pub fn check_mem_value_available(&mut self, mem: Memory, value: Node) -> bool {
        self.result(
            self.state.value_available_in_mem_now(value, mem),
            self.state.trigger_value_mem_available[value.0][mem.0],
        )
    }

    #[must_use]
    pub fn check_mem_value_not_available(&mut self, mem: Memory, value: Node) -> bool {
        // TODO use drop trigger here
        self.result(
            !self.state.value_available_in_mem_now(value, mem),
            false,
        )
    }

    #[must_use]
    pub fn was_triggered(self) -> bool {
        assert!(self.valid);
        self.triggered
    }
}

impl Dominance for State {
    type Aux = Problem;

    /// A state is better than another state if the second state can be reached from the first one
    /// by only taking useless or harmful actions. The exhaustive list of these actions is:
    /// * burn an arbitrary positive amount of energy
    /// * waste an arbitrary positive amount of time
    ///     * either right now (ie. increase curr_time)
    ///     * or as a future promise (ie. increase min_time)
    ///     * or as part of operations (ie. keep a core operation running for a bit longer then necessary)
    /// * delete values from memories
    /// * apply a problem automorphism (harmless)
    fn dominance(&self, other: &Self, problem: &Problem) -> DomDir {
        // TODO explicitly create the list of useless actions so this function can be double-checked?
        // TODO relax: we could also take extra actions (eg. copy a value over) to reach dominance, but maybe
        //   that's second-order stuff that should be kept in the search itself
        // TODO double-check that this function is both correct and complete
        //   look at examples of reject/accept state pairs!

        let mut dom = DomBuilder::new(self, other);

        // basics
        dom.minimize(|s| s.curr_time);
        dom.minimize(|s| s.minimum_time);
        dom.minimize(|s| s.curr_energy);
        dom_early_check!(dom);

        // group availability
        for group in problem.hardware.groups() {
            dom.minimize(|s| {
                match s.state_group[group.0] {
                    None => s.curr_time,
                    Some(action) => action.time().end,
                }
            });
            dom_early_check!(dom);
        }

        // value availability
        for mem in problem.hardware.memories() {
            for &value in chain(self.state_memory_node[mem.0].keys(), other.state_memory_node[mem.0].keys()) {
                dom.minimize(|s| s.value_mem_dom_key_min(value, mem));
                dom_early_check!(dom);
            }
        }

        // TODO memory space?
        // TODO times for locked values? (really this is again just memory space)
        //    => combine both into a "future memory space" temporal sequence?

        dom.finish()
    }
}

impl Dominance for Cost {
    type Aux = ();
    fn dominance(&self, other: &Self, _: &()) -> DomDir {
        let mut dom = DomBuilder::new(self, other);
        dom.minimize(|s| s.time);
        dom.minimize(|s| s.energy);
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