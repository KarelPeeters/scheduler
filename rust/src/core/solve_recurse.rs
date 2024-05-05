use std::collections::hash_map::Entry;
use std::collections::HashMap;
use itertools::Itertools;
use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::state::{Cost, State};

pub trait ReporterRecurse {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, state: &State);
}

enum CacheEntry {
    Placeholder,
    Completed(CostFrontier),
}

pub struct Context<'p, 'r, 'f, R: ReporterRecurse> {
    problem: &'p Problem,
    target: CostTarget,
    reporter: &'r mut R,
    frontier_done: &'f mut Frontier<Cost, State>,
    frontier_partial: &'f mut LinearFrontier,
    cache: &'f mut HashMap<Vec<i64>, CacheEntry>
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

    let result = recurse(&mut ctx, state);

    println!("Recursion result:");
    for r in result.to_vec() {
        println!("{:?}", r);
    }

    // TODO actually recover states
    let mut clean = Frontier::new();
    for (&c, _) in result.inner.iter_arbitrary() {
        assert!(clean.add(&c, &target, || State::new(problem)));
    }
    clean
}

#[inline(never)]
fn recurse<R: ReporterRecurse>(ctx: &mut Context<R>, state: State) -> CostFrontier {
    let problem = ctx.problem;
    state.assert_valid(problem);

    // bookkeeping
    // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
    //   similar to the idea to improve the pruning in the other place?
    if state.is_done(problem) {
        assert_eq!(state.curr_time, state.minimum_time);

        let cost = state.current_cost();
        let added_done = ctx.frontier_done.add(&cost, &ctx.target, || state.clone());
        if added_done {
            ctx.reporter.report_new_schedule(problem, ctx.frontier_done, cost, &state);
        }

        return CostFrontier::single(Cost::default());
    }

    let achievement = state.achievement(problem);
    match ctx.cache.entry(achievement.clone()) {
        Entry::Occupied(entry) => {
            return match entry.get() {
                // hit loop, which is useless
                CacheEntry::Placeholder => CostFrontier::empty(),
                // cache hit
                CacheEntry::Completed(frontier) => frontier.clone(),
            }
        }
        Entry::Vacant(entry) => {
            // first time seeing this state, mark for loop detection
            entry.insert(CacheEntry::Placeholder);
        }
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

    let mut frontier = CostFrontier::empty();
    let curr_cost = state.current_cost();

    expand(problem, state, &mut |next_state| {
        let next_cost = next_state.current_cost();
        let next_frontier = recurse(ctx, next_state);

        for c in next_frontier.iter() {
            frontier.add(ctx.target, c + next_cost - curr_cost);
        }
    });

    let prev = ctx.cache.insert(achievement, CacheEntry::Completed(frontier.clone()));
    assert!(matches!(prev, Some(CacheEntry::Placeholder)));
    
    frontier
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