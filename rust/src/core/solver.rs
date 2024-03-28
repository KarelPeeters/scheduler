use itertools::{Itertools, zip_eq};

use crate::core::frontier::Frontier;
use crate::core::problem::{Allocation, Channel, Direction, Memory, Node, Problem};
use crate::core::state::{Cost, State, ValueState};

pub trait Reporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier: &Frontier<State, ()>, state: &State);
}

#[derive(Debug, Copy, Clone)]
pub struct DummyReporter;

impl Reporter for DummyReporter {
    fn report_new_schedule(&mut self, _: &Problem, _: &Frontier<Cost, State>, _: Cost, _: &State) {}
    fn report_new_state(&mut self, _: &Problem, _: &Frontier<State, ()>, _: &State) {}
}

pub struct Context<'p, 'r, 'f, R: Reporter> {
    problem: &'p Problem,
    reporter: &'r mut R,
    frontier_done: &'f mut Frontier<Cost, State>,
    frontier_partial: &'f mut Frontier<State, ()>,
}

pub fn solve(problem: &Problem, reporter: &mut impl Reporter) {
    let mut frontier_done = Frontier::new();
    let mut frontier_partial = Frontier::new();

    let mut ctx = Context {
        problem,
        reporter,
        frontier_done: &mut frontier_done,
        frontier_partial: &mut frontier_partial,
    };

    let state = State::new(problem);
    recurse(&mut ctx, state);
}

// TODO split this up into smaller functions
fn recurse<R: Reporter>(ctx: &mut Context<R>, mut state: State) {
    let problem = ctx.problem;
    state.assert_valid(problem);

    // bookkeeping
    if state.is_done(problem) {
        assert_eq!(state.curr_time, state.minimum_time);

        let cost = state.current_cost();
        let was_added = ctx.frontier_done.add(&cost, &(), || state.clone());
        if was_added {
            ctx.reporter.report_new_schedule(problem, ctx.frontier_done, cost, &state);
        }

        return;
    }

    if !ctx.frontier_done.would_add(&state.best_case_cost(problem), &()) {
        return;
    }
    if !ctx.frontier_partial.add(&state, problem, || ()) {
        return;
    }
    ctx.reporter.report_new_state(problem, ctx.frontier_partial, &state);

    // drop dead values from memories
    // TODO only do this after wait?
    for mem_i in 0..state.state_memory_node.len() {
        let mem = Memory(mem_i);
        let used_before = state.mem_space_used(problem, mem);

        let mem_content = &mut state.state_memory_node[mem.0];
        let mut exit = false;
        
        mem_content.retain(|value, &mut availability| {
            if let ValueState::AvailableNow { read_lock_count, read_count } = availability {
                let dead = state.value_remaining_unstarted_uses[value.0] == 0;
                
                if dead && read_count == 0 {
                    // prune this branch if we're dropping values that haven't been used,
                    //   we should have avoided copying them in the first place!
                    exit = true;
                    return true;
                }
                
                read_lock_count > 0 || !dead
            } else {
                true
            }
        });
        
        if exit {
            return
        }

        let used_after = state.mem_space_used(problem, mem);
        if used_before != used_after {
            state.trigger_mem_usage_decreased.push((used_before, used_after));
        }
    }

    // wait for first operation to finish
    if let Some(first_done_time) = state.first_done_time() {
        let state_next = state.clone_and_then(|n| n.do_action_wait(problem, first_done_time));
        recurse(ctx, state_next);
    }

    // start core operations
    for alloc in problem.allocations() {
        recurse_try_alloc(ctx, &mut state, alloc);
    }

    // start channel operations
    for channel in problem.hardware.channels() {
        recurse_try_channel(ctx, &mut state, channel);
    }
}

fn recurse_try_alloc<R: Reporter>(ctx: &mut Context<R>, state: &mut State, alloc: Allocation) {
    // aliases
    let problem = ctx.problem;
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
    recurse(ctx, state_next);

    // mark as tried
    assert!(state.tried_allocs.insert(alloc));
}

fn recurse_try_channel<R: Reporter>(ctx: &mut Context<R>, state: &mut State, channel: Channel) {
    // aliases
    let problem = ctx.problem;
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
            recurse_try_channel_transfer(ctx, state, channel, value, dir_a_to_b);
        }
    }
}

fn recurse_try_channel_transfer<R: Reporter>(ctx: &mut Context<R>, state: &mut State, channel: Channel, value: Node, dir_a_to_b: bool) {
    // aliases
    let problem = ctx.problem;
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
    recurse(ctx, state_next);

    // mark as tried
    assert!(state.tried_transfers.insert(tried_key));
}
