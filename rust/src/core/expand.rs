use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionDrop};
use crate::core::state::{SkippedDropInfo, State, ValueState};

#[inline(never)]
pub fn expand(problem: &Problem, mut state: State, next: &mut impl FnMut(State)) {
    // drop dead values
    //   this needs to be done after every action, not just after waiting:
    //   actions might have made duplicate values in other memories dead
    if state.drop_dead_values(problem).is_err() {
        // prune, unused value turns out to be dead (so it should not have been created anyway)
        // TODO instead of pruning, recursively subtract all the costs associated with it and just continue?
        //   maybe even go back and add all partial states?
        return;
    }

    // TODO where to call this exactly?
    state.prune_skipped_actions(problem);
    
    // drop non-dead values
    // TODO only drop dead values if necessary for the actions that are starting at the current time?
    //   this can be implemented as a prune at the start of wait for simplicity
    //   (prune if more space than necessary was created)
    for (mem, mem_info) in &problem.hardware.memories {
        // memory has unlimited size, no point in ever dropping things
        if mem_info.size_bits.is_none() {
            continue;
        }
        expand_try_drop(problem, &mut state, next, mem);
    }

    // maybe start core operations
    for alloc in problem.allocations.keys() {
        expand_try_alloc(problem, &mut state, next, alloc);
    }

    // maybe start channel operations
    for channel in problem.hardware.channels.keys() {
        expand_try_channel(problem, &mut state, next, channel);
    }

    // wait for first operation to finish
    // we only do this after core and channel operations to get extra pruning form actions we've chosen _not_ to take
    if let Some(first_done_time) = state.first_done_time() {
        let mut state_next = state.clone();
        match state_next.do_action_wait(problem, first_done_time) {
            // success, continue with the next state
            Ok(_) => next(state_next),
            // pruned, don't do anything
            Err(_) => {}
        }
    }
}

// TODO instead of early dropping, only drop if we actually need more space?
//   alternatively, prune value drops that didn't end up being necessary
#[inline(never)]
fn expand_try_drop(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), mem: Memory) {
    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes.keys() {
        let action = ActionDrop { mem, value };

        // check if action should be tried 
        if state.skipped_drops.contains_key(&action) {
            continue;
        }
        if !state.could_start_action_now(problem, Action::Drop(action)) {
            continue;
        }

        // do action
        let mut state_next = state.clone();
        state_next.drop_value(problem, mem, value);
        expand(problem, state_next, next);

        // don't do action, mark as skipped
        let info = SkippedDropInfo {
            time: state.curr_time,
            read_count: match state.value_mem_availability(value, mem) {
                Some(ValueState::AvailableNow { read_count, .. }) => read_count,
                _ => unreachable!()
            },
        };

        let prev = state.skipped_drops.insert(action, info);
        assert!(prev.is_none());
    }
}

#[inline(never)]
fn expand_try_alloc(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), alloc: Allocation) {
    // check if action can run
    if state.skipped_allocs.contains_key(&alloc) {
        return;
    }
    if !state.could_start_action_now(problem, Action::Core(alloc)) {
        return;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| time_range = Some(n.do_action_core(problem, alloc)));
    expand(problem, state_next, next);

    // don't do action, mark as skipped
    let prev = state.skipped_allocs.insert(alloc, time_range.unwrap()).is_none();
    assert!(prev);
}

#[inline(never)]
fn expand_try_channel(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), channel: Channel) {
    // check that channel is actually free before going through the following loops
    let channel_info = &problem.hardware.channels[channel];
    if state.state_group[channel_info.group].is_some() {
        return;
    }

    // TODO switch to indexmap for deterministic iteration order, instead of this slower loop-and-check workaround
    for value in problem.graph.nodes.keys() {
        if state.state_memory_node[channel_info.mem_source].contains_key(&value) {
            expand_try_channel_transfer(problem, state, next, channel, value);
        }
    }
}

#[inline(never)]
fn expand_try_channel_transfer(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), channel: Channel, value: Node) {
    let action = ActionChannel { channel, value };

    // check if action can run
    if state.skipped_transfers.contains_key(&action) {
        return;
    }
    let action = ActionChannel { channel, value };
    if !state.could_start_action_now(problem, Action::Channel(action)) {
        return;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| {
        time_range = Some(n.do_action_channel(problem, action))
    });
    expand(problem, state_next, next);

    // don't do action, mark as skipped
    let prev = state.skipped_transfers.insert(action, time_range.unwrap());
    assert!(prev.is_none());
}
