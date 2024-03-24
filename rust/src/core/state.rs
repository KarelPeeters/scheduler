use std::collections::{HashMap, HashSet};

use itertools::{enumerate, Itertools, zip_eq};

use crate::core::frontier::{DomBuilder, DomDir, Dominance};
use crate::core::problem::{Allocation, Channel, Core, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionCore, ActionWait};
use crate::dom_early_check;
use crate::util::mini::{IterFloatExt, min_f64};

#[derive(Clone)]
pub struct State {
    // minimal state
    pub actions_taken: Vec<Action>,

    // memoized information
    pub curr_time: f64,
    pub curr_energy: f64,
    pub minimum_time: f64,

    pub state_core: Vec<Option<ActionCore>>,
    pub state_channel: Vec<Option<ActionChannel>>,
    pub state_memory_node: Vec<HashMap<Node, ValueAvailability>>,

    pub unstarted_nodes: HashSet<Node>,
    pub value_remaining_unstarted_uses: Vec<u32>,

    // triggers
    // TODO add value dropped trigger
    pub trigger_everything: bool,
    pub trigger_core_free: Vec<bool>,
    pub trigger_channel_free: Vec<bool>,
    pub trigger_value_mem_available: Vec<Vec<bool>>,
    pub trigger_value_mem_unlocked: Vec<Vec<bool>>,
    pub trigger_mem_usage_decreased: Vec<(u64, u64)>,

    // filtering
    pub tried_allocs: HashSet<Allocation>,
    pub tried_transfers: HashSet<(Channel, Node, bool)>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Cost {
    time: f64,
    energy: f64,
}

#[derive(Clone, Copy)]
pub struct Trigger<'s> {
    state: &'s State,
    triggered: bool,
    valid: bool,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ValueAvailability {
    AvailableNow { read_lock_count: u64 },
    AvailableAtTime(f64),
}

impl State {
    pub fn new(problem: &Problem) -> Self {
        // aliases
        let graph = &problem.graph;
        let hardware = &problem.hardware;

        let node_count = graph.nodes().len();
        let mem_count = hardware.memories().len();
        let channel_count = hardware.channels().len();
        let core_count = hardware.cores().len();

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
            let prev = state_memory_node[mem.0].insert(value, ValueAvailability::AvailableNow { read_lock_count: 0 });
            assert!(prev.is_none());
        }

        // construct
        State {
            actions_taken: vec![],
            curr_time: 0.0,
            curr_energy: 0.0,
            minimum_time: 0.0,
            state_core: vec![None; core_count],
            state_channel: vec![None; channel_count],
            state_memory_node,
            unstarted_nodes,
            value_remaining_unstarted_uses,
            trigger_everything: true,
            trigger_core_free: vec![false; core_count],
            trigger_channel_free: vec![false; channel_count],
            trigger_value_mem_available: vec![vec![false; mem_count]; node_count],
            trigger_value_mem_unlocked: vec![vec![false; mem_count]; node_count],
            trigger_mem_usage_decreased: vec![],
            tried_allocs: HashSet::new(),
            tried_transfers: HashSet::new(),
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

        for state in &self.state_channel {
            if let Some(state) = state {
                let channel_info = &problem.hardware.channel_info[state.channel.0];
                let (mem_source, mem_dest) = match state.dir_a_to_b {
                    true => (channel_info.mem_a, channel_info.mem_b),
                    false => (channel_info.mem_b, channel_info.mem_a),
                };

                *expected_state_memory_read_lock_count[mem_source.0].entry(state.value).or_insert(0) += 1;
                assert_eq!(self.state_memory_node[mem_dest.0].get(&state.value).unwrap(), &ValueAvailability::AvailableAtTime(state.time_end));
            }
        }

        for state in &self.state_core {
            if let Some(state) = state {
                let alloc = &problem.allocation_info[state.alloc.0];
                for (&input, &mem) in zip_eq(&graph.node_info[alloc.node.0].inputs, &alloc.input_memories) {
                    *expected_state_memory_read_lock_count[mem.0].entry(input).or_insert(0) += 1;
                }
                assert_eq!(self.state_memory_node[alloc.output_memory.0].get(&alloc.node).unwrap(), &ValueAvailability::AvailableAtTime(state.time_end));
            }
        }

        for mem in hardware.memories() {
            let expected = &expected_state_memory_read_lock_count[mem.0];
            let actual = &self.state_memory_node[mem.0];

            for (node, &node_actual) in actual {
                match node_actual {
                    ValueAvailability::AvailableNow { read_lock_count } => {
                        assert_eq!(expected.get(node).copied().unwrap_or(0), read_lock_count)
                    }
                    ValueAvailability::AvailableAtTime(_) => assert!(!expected.contains_key(node)),
                }
            }

            for (node, &read_lock_count) in expected {
                assert_eq!(actual.get(node).unwrap(), &ValueAvailability::AvailableNow { read_lock_count });
            }
        }
    }

    pub fn best_case_cost(&self, problem: &Problem) -> Cost {
        // TODO precompute these values for each node
        let min_additional_energy = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.energy)
                .min_float().unwrap()
        }).sum::<f64>();

        let min_additional_time = self.unstarted_nodes.iter().map(|node| {
            problem.allocation_info.iter()
                .filter(|alloc| alloc.node == *node)
                .map(|alloc| alloc.time)
                .min_float().unwrap()
        }).sum::<f64>();

        Cost {
            time: f64::max(self.minimum_time, self.curr_time + min_additional_time),
            energy: self.curr_energy + min_additional_energy,
        }
    }

    // TODO use this function in a bunch more places
    pub fn value_mem_availability(&self, value: Node, mem: Memory) -> Option<ValueAvailability> {
        self.state_memory_node[mem.0].get(&value).copied()
    }

    pub fn value_available_in_mem_now(&self, value: Node, mem: Memory) -> bool {
        match self.value_mem_availability(value, mem) {
            Some(ValueAvailability::AvailableNow { .. }) => true,
            Some(ValueAvailability::AvailableAtTime(_)) => false,
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
        let mut result = f64::INFINITY;

        for action in &self.state_core {
            if let Some(action) = action {
                result = min_f64(result, action.time_end);
            }
        }
        for action in &self.state_channel {
            if let Some(action) = action {
                result = min_f64(result, action.time_end);
            }
        }

        if result == f64::INFINITY {
            None
        } else {
            Some(result)
        }
    }

    #[must_use]
    pub fn clone_and_then(&self, mut f: impl FnMut(&mut Self)) -> Self {
        let mut next = self.clone();
        f(&mut next);
        next
    }

    pub fn clear_triggers_and_tried(&mut self) {
        self.trigger_everything = false;
        self.trigger_core_free.iter_mut().for_each(|x| *x = false);
        self.trigger_channel_free.iter_mut().for_each(|x| *x = false);
        self.trigger_value_mem_available.iter_mut().for_each(|x| x.iter_mut().for_each(|x| *x = false));
        self.trigger_value_mem_unlocked.iter_mut().for_each(|x| x.iter_mut().for_each(|x| *x = false));
        self.trigger_mem_usage_decreased.clear();

        self.tried_allocs.clear();
        self.tried_transfers.clear();
    }

    fn add_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();

        match availability {
            ValueAvailability::AvailableNow { read_lock_count } => {
                *read_lock_count += 1;
            }
            ValueAvailability::AvailableAtTime(_) => panic!(),
        }
    }

    fn release_mem_value_read_lock(&mut self, value: Node, mem: Memory) {
        let availability = self.state_memory_node[mem.0].get_mut(&value).unwrap();
        match availability {
            ValueAvailability::AvailableNow { read_lock_count } => {
                assert!(*read_lock_count > 0);
                *read_lock_count -= 1;
                if *read_lock_count == 0 {
                    self.trigger_value_mem_unlocked[value.0][mem.0] = true;
                }
            }
            ValueAvailability::AvailableAtTime(_) => panic!(),
        }
    }

    fn mark_mem_value_available(&mut self, value: Node, mem: Memory, availability: ValueAvailability) {
        let prev = self.state_memory_node[mem.0].insert(value, availability);

        match availability {
            ValueAvailability::AvailableNow { .. } => {
                assert!(matches!(prev, Some(ValueAvailability::AvailableAtTime(_))));

                let trigger = &mut self.trigger_value_mem_available[value.0][mem.0];
                assert!(!*trigger);
                *trigger = true;
            }
            ValueAvailability::AvailableAtTime(_) => {
                assert_eq!(prev, None);
            }
        }
    }

    fn claim_core(&mut self, core: Core, action: ActionCore) {
        let state = &mut self.state_core[core.0];
        assert!(state.is_none());
        *state = Some(action);
    }

    fn release_core(&mut self, core: Core) {
        let state = &mut self.state_core[core.0];
        assert!(state.is_some());
        *state = None;

        let trigger = &mut self.trigger_core_free[core.0];
        assert!(!*trigger);
        *trigger = true;
    }

    fn claim_channel(&mut self, channel: Channel, action: ActionChannel) {
        let state = &mut self.state_channel[channel.0];
        assert!(state.is_none());
        *state = Some(action);
    }

    fn release_channel(&mut self, channel: Channel) {
        let state = &mut self.state_channel[channel.0];
        assert!(state.is_some());
        *state = None;

        let trigger = &mut self.trigger_channel_free[channel.0];
        assert!(!*trigger);
        *trigger = true;
    }

    pub fn do_action_wait(&mut self, problem: &Problem, time_end: f64) {
        // metadata
        assert!(time_end >= self.curr_time);
        assert!(time_end <= self.minimum_time);
        self.curr_time = time_end;
        self.actions_taken.push(Action::Wait(ActionWait { time_start: self.curr_time, time_end }));

        // TODO avoid cloning these in the first place!
        self.clear_triggers_and_tried();

        // complete core operations
        for core_i in 0..self.state_core.len() {
            let core = Core(core_i);
            if let Some(action) = self.state_core[core_i] {
                if action.time_end <= time_end {
                    let alloc = &problem.allocation_info[action.alloc.0];
                    for (&input_value, &input_mem) in zip_eq(&problem.graph.node_info[alloc.node.0].inputs, &alloc.input_memories) {
                        self.release_mem_value_read_lock(input_value, input_mem);
                    }
                    self.mark_mem_value_available(alloc.node, alloc.output_memory, ValueAvailability::AvailableNow { read_lock_count: 0 });
                    self.release_core(core);
                }
            }
        }

        // complete channel operations
        for channel_i in 0..self.state_channel.len() {
            let channel = Channel(channel_i);
            if let Some(action) = self.state_channel[channel_i] {
                let channel_info = &problem.hardware.channel_info[channel.0];
                let (mem_source, mem_dest) = channel_info.mem_source_dest(action.dir_a_to_b);

                if action.time_end <= time_end {
                    self.release_mem_value_read_lock(action.value, mem_source);
                    self.mark_mem_value_available(action.value, mem_dest, ValueAvailability::AvailableNow { read_lock_count: 0 });
                    self.release_channel(channel);
                }
            }
        }
    }

    pub fn do_action_core(&mut self, problem: &Problem, alloc: Allocation) {
        let alloc_info = &problem.allocation_info[alloc.0];
        let node_info = &problem.graph.node_info[alloc_info.node.0];

        // metadata
        let time_delta = alloc_info.time;
        let energy_delta = alloc_info.energy;

        let time_end =  self.curr_time + time_delta;
        let action = ActionCore {
            time_start: self.curr_time,
            time_end,
            alloc,
        };

        // real changes
        assert!(self.unstarted_nodes.remove(&alloc_info.node));
        for (&input_value, &input_mem) in zip_eq(&node_info.inputs, &alloc_info.input_memories) {
            self.add_mem_value_read_lock(input_value, input_mem);
        }
        self.mark_mem_value_available(alloc_info.node, alloc_info.output_memory, ValueAvailability::AvailableAtTime(time_end));
        self.claim_core(alloc_info.core, action);

        // default
        self.start_action_common(Action::Core(action), time_end, energy_delta);
    }

    pub fn do_action_channel(&mut self, problem: &Problem, channel: Channel, value: Node, dir_a_to_b: bool) {
        let channel_info = &problem.hardware.channel_info[channel.0];

        // metadata
        let size_bits = problem.graph.node_info[value.0].size_bits;
        let time_delta = channel_info.time_to_transfer(size_bits);
        let energy_delta = channel_info.energy_to_transfer(size_bits);

        let time_end = self.curr_time + time_delta;
        let action = ActionChannel {
            time_start: self.curr_time,
            time_end,
            channel,
            dir_a_to_b,
            value,
        };

        // real changes
        let (mem_source, mem_dest) = channel_info.mem_source_dest(dir_a_to_b);
        self.add_mem_value_read_lock(value, mem_source);
        self.mark_mem_value_available(value, mem_dest, ValueAvailability::AvailableAtTime(time_end));
        self.claim_channel(channel, action);

        // common
        self.start_action_common(Action::Channel(action), time_end, energy_delta);
    }

    fn start_action_common(&mut self, action: Action, time_end: f64, energy_delta: f64) {
        self.actions_taken.push(action);
        self.minimum_time = f64::max(self.minimum_time, time_end);
        self.curr_energy += energy_delta;
    }

    fn value_mem_dom_key_min(&self, value: Node, mem: Memory) -> impl PartialOrd {
        if self.value_remaining_unstarted_uses[value.0] == 0 {
            // dead, best possible case
            return (0, 0.0);
        }

        match self.value_mem_availability(value, mem) {
            // available now
            Some(ValueAvailability::AvailableNow { .. }) => (1, 0.0),
            // available later
            // TODO subtract current time here?
            Some(ValueAvailability::AvailableAtTime(time)) => (2, -time),
            // not even scheduled, worst case
            None => (3, 0.0),
        }

    }
}

impl Trigger<'_> {
    #[must_use]
    fn result(&mut self, valid: bool, trigger: bool) -> bool {
        if trigger {
            assert!(valid);
        }
        self.triggered |= trigger;
        valid
    }

    #[must_use]
    pub fn check_core_free(&mut self, core: Core) -> bool {
        self.result(
            self.state.state_core[core.0].is_none(),
            self.state.trigger_core_free[core.0],
        )
    }

    #[must_use]
    pub fn check_channel_free(&mut self, channel: Channel) -> bool {
        self.result(
            self.state.state_channel[channel.0].is_none(),
            self.state.trigger_channel_free[channel.0],
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
        // TODO use min_time here?
        dom.minimize(|s| s.curr_time);
        dom.minimize(|s| s.curr_energy);
        dom_early_check!(dom);

        // value availability
        for mem in problem.hardware.memories() {
            for value in problem.graph.nodes() {
                dom.minimize(|s| s.value_mem_dom_key_min(value, mem));
            }
            dom_early_check!(dom);
        }

        // channel and core availability
        for channel in problem.hardware.channels() {
            dom.minimize(|s| {
                match s.state_channel[channel.0] {
                    None => f64::NEG_INFINITY,
                    Some(action) => action.time_end,
                }
            });
            dom_early_check!(dom);
        }
        for core in problem.hardware.cores() {
            dom.minimize(|s| {
                match s.state_core[core.0] {
                    None => f64::NEG_INFINITY,
                    Some(action) => action.time_end,
                }
            });
            dom_early_check!(dom);
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
