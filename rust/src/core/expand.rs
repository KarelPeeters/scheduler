use itertools::zip_eq;

use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::state::{State, ValueState};

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
    
    // drop non-dead values
    // TODO only drop dead values if necessary for the actions that are starting at the current time?
    //   this can be implemented as a prune at the start of wait for simplicity
    //   (prune if more space than necessary was created)
    for mem in problem.hardware.memories() {
        // memory has unlimited size, no point in ever dropping things
        if problem.hardware.mem_info[mem.0].size_bits.is_none() {
            continue;
        }
        expand_try_drop(problem, &state, next, mem);
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
        match state_next.do_action_wait(problem, first_done_time) {
            // success, continue with the next state
            Ok(_) => next(state_next),
            // pruned, don't do anything
            Err(_) => {}
        }
    }
}

// TODO instead of early dropping, only drop if we actually need more space?
#[inline(never)]
fn expand_try_drop(problem: &Problem, state: &State, next: &mut impl FnMut(State), mem: Memory) {
    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes() {
        match state.state_memory_node[mem.0].get(&value) {
            // can't drop value that's not even available
            None | Some(ValueState::AvailableAtTime(_)) => {
                continue;
            }
            Some(&ValueState::AvailableNow { read_lock_count, read_count: _, since: _ }) => {
                let mut trigger = state.new_trigger();

                if read_lock_count == 0 {
                    // TODO instead of this assert, prune states with dead values that are still being copied or calculated
                    assert!(state.value_remaining_unstarted_uses[value.0] > 0, "dead values should have been dropped already");
                }

                // can't drop value that's locked, and
                //   no reason to drop unused value: it should not have been put here in the first place
                if !trigger.check_mem_value_unlocked_and_read(mem, value) {
                    continue;
                }
                // don't drop last instance of live value
                if !trigger.check_not_single_live_instance(value) {
                    continue;
                }

                if !trigger.was_triggered() {
                    continue;
                }

                // try dropping the value
                let mut state_next = state.clone();
                state_next.drop_value(problem, mem, value);
                expand(problem, state_next, next);

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
    expand(problem, state_next, next);

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
    expand(problem, state_next, next);

    // mark as tried
    let prev = state.tried_transfers.insert((channel, value), time_range.unwrap());
    assert!(prev.is_none());
}
