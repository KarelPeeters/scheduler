use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::schedule::{Action, ActionChannel, ActionDrop};
use crate::core::state::State;

// TODO deduplicate again, now that "tried actions" have been removed
//   either reintroduce them as within-expand mini state, or rely on the recurse cache to fix it?
#[inline(never)]
pub fn expand(problem: &Problem, mut state: State, next: &mut impl FnMut(State)) {
    // drop dead values
    //   this needs to be done after every action, not just after waiting:
    //   actions might have made duplicate values in other memories dead
    state.drop_dead_values(problem);

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
        state_next.do_action_wait(problem, first_done_time);
        next(state_next);
    }
}

// TODO instead of early dropping, only drop if we actually need more space?
//   alternatively, prune value drops that didn't end up being necessary
// TODO can we just implicitly treat all states as having possibly dropped all existing values?
#[inline(never)]
fn expand_try_drop(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), mem: Memory) {
    // TODO switch to indexmap for deterministic iteration order?
    for value in problem.graph.nodes.keys() {
        let action = ActionDrop { mem, value };

        // check if action should be tried
        if !state.could_start_action_now(problem, Action::Drop(action)) {
            continue;
        }

        // do action
        let mut state_next = state.clone();
        state_next.drop_value(problem, mem, value);
        expand(problem, state_next, next);
    }
}

#[inline(never)]
fn expand_try_alloc(problem: &Problem, state: &mut State, next: &mut impl FnMut(State), alloc: Allocation) {
    // check if action can run
    if !state.could_start_action_now(problem, Action::Core(alloc)) {
        return;
    }

    // do action
    let mut time_range = None;
    let state_next = state.clone_and_then(|n| time_range = Some(n.do_action_core(problem, alloc)));
    expand(problem, state_next, next);
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
    // check if action can run
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
}
