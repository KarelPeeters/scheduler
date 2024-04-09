use itertools::{Itertools, zip_eq};

use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::new_frontier::NewFrontier;
use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::state::{Cost, State, ValueState};

pub trait Reporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier: &mut Frontier<State, ()>, frontier_new: &mut NewFrontier, frontier_linear: &mut LinearFrontier, state: &State);
}

#[derive(Debug, Copy, Clone)]
pub struct DummyReporter;

impl Reporter for DummyReporter {
    fn report_new_schedule(&mut self, _: &Problem, _: &Frontier<Cost, State>, _: Cost, _: &State) {}
    fn report_new_state(&mut self, _: &Problem, _: &mut Frontier<State, ()>, _: &mut NewFrontier, _: &mut LinearFrontier, _: &State) {}
}

pub struct Context<'p, 'r, 'f, R: Reporter> {
    problem: &'p Problem,
    reporter: &'r mut R,
    frontier_done: &'f mut Frontier<Cost, State>,
    frontier_partial: &'f mut Frontier<State, ()>,
    frontier_partial_new: &'f mut NewFrontier,
    frontier_partial_linear: &'f mut LinearFrontier,
}

pub fn solve(problem: &Problem, reporter: &mut impl Reporter) {
    let state = State::new(problem);
    
    let mut frontier_done = Frontier::new();
    let mut frontier_partial = Frontier::new();
    let mut frontier_partial_new = NewFrontier::new(state.dom_key_min(problem).1, 1);
    let mut frontier_partial_linear = LinearFrontier::new(state.dom_key_min(problem).1);

    let mut ctx = Context {
        problem,
        reporter,
        frontier_done: &mut frontier_done,
        frontier_partial: &mut frontier_partial,
        frontier_partial_new: &mut frontier_partial_new,
        frontier_partial_linear: &mut frontier_partial_linear,
    };

    recurse(&mut ctx, state);
}

// TODO split this up into smaller functions
#[inline(never)]
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
    // let added = ctx.frontier_partial.add(&state, problem, || ());
    // let added_new = ctx.frontier_partial_new.add_if_not_dominated(state.dom_key_min(problem).0);
    let added_linear = ctx.frontier_partial_linear.add_if_not_dominated(state.dom_key_min(problem).0);

    // assert_eq!(added_new, added_linear);
    // assert_eq!(ctx.frontier_partial_new.len(), ctx.frontier_partial_linear.len());

    if !added_linear {
        return;
    }
    ctx.reporter.report_new_state(problem, ctx.frontier_partial, ctx.frontier_partial_new, ctx.frontier_partial_linear, &state);

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
                    // TODO don't do this, just delete the operation but keep going?
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

    // start core operations
    for alloc in problem.allocations() {
        recurse_try_alloc(ctx, &mut state, alloc);
    }

    // start channel operations
    for channel in problem.hardware.channels() {
        recurse_try_channel(ctx, &mut state, channel);
    }

    // wait for first operation to finish
    // we only do this after core and channel operations to get extra pruning form actions we've chosen _not_ to take
    if let Some(first_done_time) = state.first_done_time() {
        let state_next = state.clone_and_then(|n| n.do_action_wait(problem, first_done_time));
        recurse(ctx, state_next);
    }
}

#[inline(never)]
fn recurse_try_alloc<R: Reporter>(ctx: &mut Context<R>, state: &mut State, alloc: Allocation) {
    // aliases
    let problem = ctx.problem;
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
    recurse(ctx, state_next);

    // mark as tried
    let prev = state.tried_allocs.insert(alloc, time_range.unwrap()).is_none();
    assert!(prev);
}

#[inline(never)]
fn recurse_try_channel<R: Reporter>(ctx: &mut Context<R>, state: &mut State, channel: Channel) {
    // aliases
    let problem = ctx.problem;
    let channel_info = &problem.hardware.channel_info[channel.0];

    // check that channel is actually free before going through the following loops
    if state.state_group[channel_info.group.0].is_some() {
        return;
    }

    // TODO avoid copy
    let values = state.state_memory_node[channel_info.mem_source.0].keys().copied().collect_vec();
    for value in values {
        recurse_try_channel_transfer(ctx, state, channel, value);
    }
}

#[inline(never)]
fn recurse_try_channel_transfer<R: Reporter>(ctx: &mut Context<R>, state: &mut State, channel: Channel, value: Node) {
    // aliases
    let problem = ctx.problem;
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
    if !trigger.check_mem_value_not_available(channel_info.mem_dest, value) {
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
    recurse(ctx, state_next);

    // mark as tried
    let prev = state.tried_transfers.insert((channel, value), time_range.unwrap());
    assert!(prev.is_none());
}
