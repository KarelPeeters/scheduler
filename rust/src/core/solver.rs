use itertools::{Itertools, zip_eq};

use crate::core::frontier::Frontier;
use crate::core::problem::{Allocation, Channel, Direction, Memory, Node, Problem};
use crate::core::state::{Cost, State, ValueAvailability};

pub fn solve(problem: &Problem) {
    let state = State::new(problem);
    let mut frontier_done = Frontier::new();
    let mut frontier = Frontier::new();
    recurse(problem, state, &mut frontier_done, &mut frontier);
}

// TODO split this up into smaller functions
fn recurse(problem: &Problem, mut state: State, frontier_done: &mut Frontier<Cost>, frontier_partial: &mut Frontier<State>) {
    state.assert_valid(problem);

    // bookkeeping
    if !frontier_done.would_add(state.best_case_cost(problem), &()) {
        return;
    }
    if !frontier_partial.add(&state, problem) {
        return;
    }

    if state.is_done(problem) {
        assert_eq!(state.curr_time, state.minimum_time);
        let cost = state.current_cost();
        let was_added = frontier_done.add(&cost, &());

        if was_added {
            println!("Found new frontier cost {:?}", cost);
        }

        return;
    }

    // drop dead values from memories
    // TODO only do this after wait?
    for mem_i in 0..state.state_memory_node.len() {
        let mem = Memory(mem_i);
        let used_before = state.mem_space_used(problem, mem);

        let mem_content = &mut state.state_memory_node[mem.0];
        mem_content.retain(|value, &mut availability| {
            if let ValueAvailability::AvailableNow { read_lock_count } = availability {
                read_lock_count > 0 || state.value_remaining_unstarted_uses[value.0] > 0
            } else {
                true
            }
        });

        let used_after = state.mem_space_used(problem, mem);
        if used_before != used_after {
            state.trigger_mem_usage_decreased.push((used_before, used_after));
        }
    }

    // wait for first operation to finish
    if let Some(first_done_time) = state.first_done_time() {
        let state_next = state.clone_and_then(|n| n.do_action_wait(problem, first_done_time));
        recurse(problem, state_next, frontier_done, frontier_partial);
    }

    // start core operations
    for alloc in problem.allocations() {
        recurse_try_alloc(problem, &mut state, frontier_done, frontier_partial, alloc);
    }

    // start channel operations
    for channel in problem.hardware.channels() {
        recurse_try_channel(problem, &mut state, frontier_done, frontier_partial, channel);
    }
}

fn recurse_try_alloc(problem: &Problem, state: &mut State, frontier_done: &mut Frontier<Cost>, frontier_partial: &mut Frontier<State>, alloc: Allocation) {
    let alloc_info = &problem.allocation_info[alloc.0];
    let node = alloc_info.node;
    let node_info = &problem.graph.node_info[node.0];

    // basic checks
    if state.tried_allocs.contains(&alloc) {
        return;
    }
    if !state.unstarted_nodes.contains(&node) {
        return;
    }

    // trigger checks
    let mut trigger = state.new_trigger();

    if !trigger.check_core_free(alloc_info.core) {
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
    let state_next = state.clone_and_then(|n| n.do_action_core(problem, alloc));
    recurse(problem, state_next, frontier_done, frontier_partial);

    // mark as tried
    assert!(state.tried_allocs.insert(alloc));
}

fn recurse_try_channel(problem: &Problem, state: &mut State, frontier_done: &mut Frontier<Cost>, frontier_partial: &mut Frontier<State>, channel: Channel) {
    let channel_info = &problem.hardware.channel_info[channel.0];

    // check that channel is actually free before going through the following loops
    if state.state_channel[channel.0].is_some() {
        return;
    }

    // try allowed directions directions
    let dirs: &[bool] = match channel_info.dir {
        Direction::AtoB => &[true],
        Direction::BtoA => &[false],
        Direction::Both => &[true, false],
    };

    // try different values
    for &dir_a_to_b in dirs {
        let mem_source = if dir_a_to_b { channel_info.mem_a } else { channel_info.mem_b };
        // TODO avoid copy?
        let values = state.state_memory_node[mem_source.0].keys().copied().collect_vec();
        for value in values {
            recurse_try_channel_transfer(problem, state, frontier_done, frontier_partial, channel, value, dir_a_to_b);
        }
    }
}

fn recurse_try_channel_transfer(problem: &Problem, state: &mut State, frontier_done: &mut Frontier<Cost>, frontier_partial: &mut Frontier<State>, channel: Channel, value: Node, dir_a_to_b: bool) {
    let value_info = &problem.graph.node_info[value.0];
    let channel_info = &problem.hardware.channel_info[channel.0];
    let (mem_source, mem_dest) = channel_info.mem_source_dest(dir_a_to_b);

    // basic checks
    let tried_key = (channel, value, dir_a_to_b);
    if state.tried_transfers.contains(&tried_key) {
        return;
    }
    // don't bother copying dead values around
    if state.value_remaining_unstarted_uses[value.0] == 0 {
        return;
    }

    // trigger checks
    let mut trigger = state.new_trigger();
    assert!(trigger.check_channel_free(channel));
    if !trigger.check_mem_value_available(mem_source, value) {
        return;
    }
    if !trigger.check_mem_value_not_available(mem_dest, value) {
        return;
    }
    if !trigger.check_mem_space_available(problem, mem_dest, value_info.size_bits) {
        return;
    }
    if !trigger.was_triggered() {
        return;
    }

    // do action
    let state_next = state.clone_and_then(|n| n.do_action_channel(problem, channel, value, dir_a_to_b));
    recurse(problem, state_next, frontier_done, frontier_partial);

    // mark as tried
    assert!(state.tried_transfers.insert(tried_key));
}
