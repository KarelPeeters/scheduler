use std::collections::HashMap;

use itertools::Itertools;

use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::state::{Cost, EarliestPruneReason, State};

pub trait ReporterRecurse {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, state: &State);
}

pub struct Context<'p, 'r, 'f, R: ReporterRecurse> {
    problem: &'p Problem,
    target: CostTarget,
    reporter: &'r mut R,
    frontier_done: &'f mut Frontier<Cost, State>,
    frontier_partial: &'f mut LinearFrontier,
    cache: &'f mut HashMap<Vec<i64>, CostFrontier>
}

pub fn solve_recurse(problem: &Problem, target: CostTarget, reporter: &mut impl ReporterRecurse) -> Frontier<Cost, State> {
    let state = State::new(problem);

    let mut frontier_done = Frontier::new();
    let mut frontier_partial_linear = LinearFrontier::new(state.dom_key_min(problem, target).1);
    let mut cache = HashMap::new();

    let mut ctx = Context {
        problem,
        target,
        reporter,
        frontier_done: &mut frontier_done,
        frontier_partial: &mut frontier_partial_linear,
        cache: &mut cache,
    };

    let result = recurse(&mut ctx, state, 10);

    println!("Recursion result:");
    println!("  prune: {:?}", result.earliest_prune_reason);
    println!("  frontier:");
    for r in result.frontier.to_vec() {
        println!("    {:?}", r);
    }

    frontier_done
}

struct RecurseResult {
    frontier: CostFrontier,
    earliest_prune_reason: Option<EarliestPruneReason>,
}

#[inline(never)]
fn recurse<R: ReporterRecurse>(ctx: &mut Context<R>, state: State, depth: u32) -> RecurseResult {
    // TODO prevent infinite recursion with stack
    if depth == 0 {
        return RecurseResult { earliest_prune_reason: None, frontier: CostFrontier::empty() };
    }

    let problem = ctx.problem;
    state.assert_valid(problem);

    // bookkeeping
    // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
    //   similar to the idea to improve the pruning in the other place?
    let curr_time = state.curr_time;
    if state.is_done(problem) {
        assert_eq!(curr_time, state.minimum_time);

        let cost = state.current_cost();
        let added_done = ctx.frontier_done.add(&cost, &ctx.target, || state.clone());
        if added_done {
            ctx.reporter.report_new_schedule(problem, ctx.frontier_done, cost, &state);
        }

        return RecurseResult { earliest_prune_reason: None, frontier: CostFrontier::single(Cost::default()) };
    }

    let achievement = state.achievement(problem);
    if let Some(prev) = ctx.cache.get(&achievement) {
        // println!("Cache hit");
        return RecurseResult { earliest_prune_reason: None, frontier: prev.clone() };
    }

    // pruning
    // TODO re-add based on frontier passed as parameter
    // if !ctx.frontier_done.would_add(&state.estimate_final_cost_conservative(problem), &ctx.target) {
    //     return;
    // }

    // TODO can we just add this back as-is? not really!
    // let added_partial = ctx.frontier_partial.add_if_not_dominated(state.dom_key_min(problem, ctx.target).0);
    // if !added_partial {
    //     return;
    // }

    ctx.reporter.report_new_state(problem, ctx.frontier_partial, &state);

    let mut curr_frontier = CostFrontier::empty();
    let curr_cost = state.current_cost();

    let earliest_prune_reason = expand(problem, state, &mut |next_state| {
        let next_cost = next_state.current_cost();
        let RecurseResult { earliest_prune_reason, frontier } = recurse(ctx, next_state, depth - 1);

        for c in frontier.iter() {
            curr_frontier.add(ctx.target, c + next_cost - curr_cost);
        }

        earliest_prune_reason
    });

    // only insert if allowed by prune reason
    if earliest_prune_reason.map_or(true, |earliest_prune_reason| curr_time < earliest_prune_reason.0) {
        // println!("Inserting into cache");
        let prev = ctx.cache.insert(achievement, curr_frontier.clone());
        assert!(prev.is_none());
    }

    RecurseResult { frontier: curr_frontier, earliest_prune_reason }
}

#[derive(Clone)]
struct CostFrontier {
    inner: Frontier<Cost, ()>,
}

impl CostFrontier {
    pub fn empty() -> CostFrontier {
        CostFrontier { inner: Frontier::new() }
    }

    pub fn single(cost: Cost) -> CostFrontier {
        // using any cost target is fine here
        let mut result = Self::empty();
        assert!(result.add(CostTarget::Full, cost));
        result
    }

    pub fn iter(&self) -> impl Iterator<Item=Cost> + '_ {
        self.inner.iter_arbitrary().map(|e| *e.0)
    }

    pub fn to_vec(&self) -> Vec<Cost> {
        let mut v = self.iter().collect_vec();
        v.sort_by_key(|e| e.time);
        v
    }

    pub fn add(&mut self, target: CostTarget, cost: Cost) -> bool {
        self.inner.add(&cost, &target, || ())
    }

    pub fn add_delta(&mut self, delta: Cost) {
        self.inner.mutate_preserve_dominance(|k, _| *k = *k + delta);
    }

    pub fn sub_delta(&mut self, delta: Cost) {
        self.inner.mutate_preserve_dominance(|k, _| *k = *k - delta);
    }
}