use std::cmp::Ordering;
use std::collections::BinaryHeap;

use itertools::zip_eq;

use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::state::{Cost, State, ValueState};

pub trait Reporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &LinearFrontier, queue: &BinaryHeap<OrdState>, state: &State);
}

#[derive(Debug, Copy, Clone)]
pub struct DummyReporter;

impl Reporter for DummyReporter {
    fn report_new_schedule(&mut self, _: &Problem, _: &Frontier<Cost, State>, _: Cost, _: &State) {}
    fn report_new_state(&mut self, _: &Problem, _: &LinearFrontier, _: &BinaryHeap<OrdState>, _: &State) {}
}

pub fn solve(problem: &Problem, reporter: &mut impl Reporter) -> Frontier<Cost, State> {
    let root_state = State::new(problem);

    let mut frontier_done = Frontier::new();
    let mut frontier_partial = LinearFrontier::new(root_state.dom_key_min(problem).1);

    // TODO why only by cost and not by the full pareto key?
    let mut queue = BinaryHeap::new();

    // add root state
    if root_state.is_done(problem) {
        assert!(frontier_done.add(&root_state.current_cost(), &(), || root_state.clone()));
        reporter.report_new_schedule(problem, &frontier_done, root_state.current_cost(), &root_state);
        return frontier_done;
    }
    queue.push(OrdState::new(problem, root_state));

    // main loop
    while let Some(OrdState { cost: _, mut state }) = queue.pop() {
        if cfg!(debug_assertions) {
            state.assert_valid(problem);
        }

        // done should have been caught in next already
        assert!(!state.is_done(problem));

        // compare against existing done states
        if !frontier_done.would_add(&state.best_case_cost(problem), &()) {
            continue;
        }

        // drop dead values
        //   this needs to be done after every action, not just after waiting:
        //   actions might have made duplicate values in other memories dead
        // TODO is this the right place to do this? or should we already do this in next?
        //    this needs to happen before frontier_partial for sure
        if state.drop_dead_values(problem).is_err() {
            // pruned, dead unused value was used
            continue;
        }

        // compare against all existing states
        let added_linear = frontier_partial.add_if_not_dominated(state.dom_key_min(problem).0);
        if !added_linear {
            continue;
        }
        reporter.report_new_state(problem, &frontier_partial, &queue, &state);

        // expand the child states
        // TODO only do done check on states that have just waited?
        // TODO change this to a state class, this is just messy and confusing
        let mut next = |next_state: State| {
            // immediate report done states
            // TODO check how much this helps vs only reporting them when visited
            // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
            //   similar to the idea to improve the pruning in the other place?
            if next_state.is_done(problem) {
                let was_added = frontier_done.add(&next_state.current_cost(), &(), || next_state.clone());
                if was_added {
                    reporter.report_new_schedule(problem, &frontier_done, next_state.current_cost(), &next_state);
                }
                return;
            }

            // immediately skip bad states here
            // (don't do full comparison yet, that can get expensive and maybe we never need to visit this state again)
            // TODO check how much this helps (and if doing the full would make it much slower)
            if !frontier_done.would_add(&next_state.best_case_cost(problem), &()) {
                return;
            }

            queue.push(OrdState::new(problem, next_state));
        };

        expand(problem, state, &mut next);
    }

    frontier_done
}

// TODO split this up into smaller functions
#[inline(never)]
fn expand(problem: &Problem, mut state: State, next: &mut impl FnMut(State)) {
    // maybe drop non-dead values in limited-size memories
    for mem in problem.hardware.memories() {
        if problem.hardware.mem_info[mem.0].size_bits.is_some() {
            expand_try_drop(problem, &state, next, mem);
        }
    }

    // maybe start core operations
    for alloc in problem.allocations() {
        expand_try_alloc(problem, &mut state, next, alloc);
    }

    // maybe start channel operations
    for channel in problem.hardware.channels() {
        expand_try_channel(problem, &mut state, next, channel);
    }

    // wait for first operation to finish
    // we only do this after core and channel operations to get extra pruning form actions we've chosen _not_ to take
    if let Some(first_done_time) = state.first_done_time() {
        let mut state_next = state.clone();
        state_next.do_action_wait(problem, first_done_time);
        next(state_next);
    }
}

// TODO instead of early dropping, only drop if we actually need more space?
#[inline(never)]
fn expand_try_drop(problem: &Problem, state: &State, next: &mut impl FnMut(State), mem: Memory) {
    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes() {
        // println!("maybe dropping {:?} in {:?}", value, mem);

        // TODO can't drop last available instance of non-dead value
        //   careful, (how) does this interact with triggers?

        match state.state_memory_node[mem.0].get(&value) {
            // can't drop value that's not even available
            None | Some(ValueState::AvailableAtTime(_)) => {
                continue;
            }
            Some(&ValueState::AvailableNow { read_lock_count, read_count }) => {
                // can't drop value that's locked, and
                //   no reason to drop unused value: it should not have put it there in the first place
                if read_lock_count > 0 || read_count == 0 {
                    continue;
                }
                assert!(state.value_remaining_unstarted_uses[value.0] > 0, "dead values should have been dropped already");

                // trigger check
                let mut trigger = state.new_trigger();
                if !trigger.check_mem_value_unlocked(mem, value) || !trigger.was_triggered() {
                    continue
                }

                // try dropping the value
                let mut next_state = state.clone();
                next_state.drop_value(problem, mem, value);
                next(next_state);

                // no need to mark as tried, the trigger stuff already handles that
            }
        }
    }
}

#[inline(never)]
fn expand_try_alloc(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), alloc: Allocation) {
    // aliases
    let alloc_info = &problem.allocation_info[alloc.0];
    let node = alloc_info.node;
    let node_info = &problem.graph.node_info[node.0];

    // basic checks
    if state.tried_allocs.contains_key(&alloc) {
        return;
    }
    if !state.unstarted_nodes.contains(&node) {
        return;
    }

    // trigger checks
    let mut trigger = state.new_trigger();

    if !trigger.check_group_free(alloc_info.group) {
        return;
    }
    if !trigger.check_mem_space_available(problem, alloc_info.output_memory, node_info.size_bits) {
        return;
    }
    for (input_value, input_mem) in zip_eq(&node_info.inputs, &alloc_info.input_memories) {
        if !trigger.check_mem_value_available(*input_mem, *input_value) {
            return;
        }
    }

    if !trigger.was_triggered() {
        return;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| time_range = Some(n.do_action_core(problem, alloc)));
    next(state_next);

    // mark as tried
    let prev = state.tried_allocs.insert(alloc, time_range.unwrap()).is_none();
    assert!(prev);
}

#[inline(never)]
fn expand_try_channel(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), channel: Channel) {
    // aliases
    let channel_info = &problem.hardware.channel_info[channel.0];

    // check that channel is actually free before going through the following loops
    if state.state_group[channel_info.group.0].is_some() {
        return;
    }

    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes() {
        if state.state_memory_node[channel_info.mem_source.0].contains_key(&value) {
            expand_try_channel_transfer(problem, state, next, channel, value);
        }
    }
}

#[inline(never)]
fn expand_try_channel_transfer(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), channel: Channel, value: Node) {
    // aliases
    let value_info = &problem.graph.node_info[value.0];
    let channel_info = &problem.hardware.channel_info[channel.0];

    // basic checks
    let tried_key = (channel, value);
    if state.tried_transfers.contains_key(&tried_key) {
        return;
    }
    // don't bother copying dead values around
    if state.value_remaining_unstarted_uses[value.0] == 0 {
        return;
    }

    // trigger checks
    let mut trigger = state.new_trigger();
    //   group free was already checked earlier
    assert!(trigger.check_group_free(channel_info.group));
    if !trigger.check_mem_value_available(channel_info.mem_source, value) {
        return;
    }
    // TODO what is this supposed to check? something like "is the given value no longer available in the target memory"?
    // TODO think about the right way to handle dropping+copying operations
    if !trigger.check_mem_value_no_availability(channel_info.mem_dest, value) {
        return;
    }
    if !trigger.check_mem_space_available(problem, channel_info.mem_dest, value_info.size_bits) {
        return;
    }
    if !trigger.was_triggered() {
        return;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| time_range = Some(n.do_action_channel(problem, channel, value)));
    next(state_next);

    // mark as tried
    let prev = state.tried_transfers.insert((channel, value), time_range.unwrap());
    assert!(prev.is_none());
}

pub struct OrdState {
    cost: Cost,
    state: State,
}

impl OrdState {
    pub fn new(_: &Problem, state: State) -> Self {
        // TODO why is using best_case_cost so much worse here?
        Self { cost: state.current_cost(), state }
    }
}

impl PartialEq for OrdState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for OrdState {}

impl PartialOrd for OrdState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdState {
    fn cmp(&self, other: &Self) -> Ordering {
        let Cost { time: self_time, energy: self_energy } = self.cost;
        let Cost { time: other_time, energy: other_energy } = other.cost;
        // TODO which order to pick here? make user-configurable?
        (self_time, self_energy).partial_cmp(&(other_time, other_energy)).unwrap()
        // (self_energy, self_time).partial_cmp(&(other_energy, other_time)).unwrap()
    }
}