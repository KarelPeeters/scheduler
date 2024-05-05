use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionDrop};
use crate::core::state::{EarliestPruneReason, SkippedDropInfo, State, ValueState};
use crate::util::float::min_option;

#[inline(never)]
#[must_use]
pub fn expand(problem: &Problem, mut state: State, next: &mut impl FnMut(State) -> Option<EarliestPruneReason>) -> Option<EarliestPruneReason> {
    // drop dead values
    //   this needs to be done after every action, not just after waiting:
    //   actions might have made duplicate values in other memories dead
    if let Err(reason) = state.drop_dead_values(problem) {
        // prune, unused value turns out to be dead (so it should not have been created anyway)
        // TODO instead of pruning, recursively subtract all the costs associated with it and just continue?
        //   maybe even go back and add all partial states?
        return Some(reason);
    }

    // TODO where to call this exactly?
    state.prune_skipped_actions(problem);

    let mut reason = None;
    
    // drop non-dead values
    // TODO only drop dead values if necessary for the actions that are starting at the current time?
    //   this can be implemented as a prune at the start of wait for simplicity
    //   (prune if more space than necessary was created)
    for (mem, mem_info) in &problem.hardware.memories {
        // memory has unlimited size, no point in ever dropping things
        if mem_info.size_bits.is_none() {
            continue;
        }
        reason = min_option(reason, expand_try_drop(problem, &mut state, next, mem));
    }

    // maybe start core operations
    for alloc in problem.allocations.keys() {
        reason = min_option(reason, expand_try_alloc(problem, &mut state, next, alloc));
    }

    // maybe start channel operations
    for channel in problem.hardware.channels.keys() {
        reason = min_option(reason, expand_try_channel(problem, &mut state, next, channel));
    }

    // wait for first operation to finish
    // we only do this after core and channel operations to get extra pruning form actions we've chosen _not_ to take
    if let Some(first_done_time) = state.first_done_time() {
        let mut state_next = state.clone();
        match state_next.do_action_wait(problem, first_done_time) {
            // success, continue with the next state
            Ok(()) => {
                reason = min_option(reason, next(state_next));
            },
            // pruned, don't do anything
            Err(r) => {
                reason = min_option(reason, Some(r));
            }
        }
    }

    reason
}

// TODO instead of early dropping, only drop if we actually need more space?
//   alternatively, prune value drops that didn't end up being necessary
#[inline(never)]
#[must_use]
fn expand_try_drop(problem: &Problem, state: &mut State, next: &mut impl FnMut(State) -> Option<EarliestPruneReason>, mem: Memory) -> Option<EarliestPruneReason> {
    let mut reason = None;
    
    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes.keys() {
        let action = ActionDrop { mem, value };

        // check if action should be tried 
        if let Some(prev) = state.skipped_drops.get(&action) {
            reason = min_option(reason, Some(EarliestPruneReason(prev.time)));
            continue;
        }
        if !state.could_start_action_now(problem, Action::Drop(action)) {
            continue;
        }

        // do action
        let mut state_next = state.clone();
        state_next.drop_value(problem, mem, value);
        reason = min_option(reason, expand(problem, state_next, next));

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

    reason
}

#[inline(never)]
#[must_use]
fn expand_try_alloc(problem: &Problem, state: &mut State, next: &mut impl FnMut(State) -> Option<EarliestPruneReason>, alloc: Allocation) -> Option<EarliestPruneReason> {
    let mut reason = None;
    
    // check if action can run
    if let Some(info) = state.skipped_allocs.get(&alloc) {
        reason = min_option(reason, Some(EarliestPruneReason(info.start)));
        return reason;
    }
    if !state.could_start_action_now(problem, Action::Core(alloc)) {
        return reason;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| time_range = Some(n.do_action_core(problem, alloc)));
    reason = min_option(reason, expand(problem, state_next, next));

    // don't do action, mark as skipped
    let prev = state.skipped_allocs.insert(alloc, time_range.unwrap()).is_none();
    assert!(prev);

    reason
}

#[inline(never)]
#[must_use]
fn expand_try_channel(problem: &Problem, state: &mut State, next: &mut impl FnMut(State) -> Option<EarliestPruneReason>, channel: Channel) -> Option<EarliestPruneReason> {
    let mut reason = None;
    
    // check that channel is actually free before going through the following loops
    let channel_info = &problem.hardware.channels[channel];
    if state.state_group[channel_info.group].is_some() {
        return reason;
    }

    // TODO switch to indexmap for deterministic iteration order, instead of this slower loop-and-check workaround
    for value in problem.graph.nodes.keys() {
        if state.state_memory_node[channel_info.mem_source].contains_key(&value) {
            reason = min_option(reason, expand_try_channel_transfer(problem, state, next, channel, value));
        }
    }

    reason
}

#[inline(never)]
#[must_use]
fn expand_try_channel_transfer(problem: &Problem, state: &mut State, next: &mut impl FnMut(State) -> Option<EarliestPruneReason>, channel: Channel, value: Node) -> Option<EarliestPruneReason> {
    let mut reason = None;
    
    let action = ActionChannel { channel, value };

    // check if action can run
    if let Some(info) = state.skipped_transfers.get(&action) {
        reason = min_option(reason, Some(EarliestPruneReason(info.start)));
        return reason;
    }
    let action = ActionChannel { channel, value };
    if !state.could_start_action_now(problem, Action::Channel(action)) {
        return reason;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| {
        time_range = Some(n.do_action_channel(problem, action))
    });
    reason = min_option(reason, expand(problem, state_next, next));

    // don't do action, mark as skipped
    let prev = state.skipped_transfers.insert(action, time_range.unwrap());
    assert!(prev.is_none());

    reason
}
