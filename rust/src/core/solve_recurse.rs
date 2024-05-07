use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt::format;
use itertools::{Itertools, rev};
use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::schedule::Action;
use crate::core::state::{Cost, State};
use unwrap_match::unwrap_match;

pub trait ReporterRecurse {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, state: &State);
}

type CostFrontier = Frontier<Cost, VecDeque<Action>>;

enum CacheEntry {
    Placeholder,
    Completed(CostFrontier),
}

pub struct Context<'p, 'r, 'f, R: ReporterRecurse> {
    problem: &'p Problem,
    target: CostTarget,
    reporter: &'r mut R,
    cache: &'f mut HashMap<Vec<i64>, CacheEntry>
}

pub fn solve_recurse(problem: &Problem, target: CostTarget, reporter: &mut impl ReporterRecurse) -> Frontier<Cost, State> {
    let state = State::new(problem);

    let mut cache = HashMap::new();

    let mut ctx = Context {
        problem,
        target,
        reporter,
        cache: &mut cache,
    };

    let result = recurse(&mut ctx, state);

    // TODO change state representation to be much more minimal and orthogonal to action list
    let mut clean = Frontier::empty();
    for (&c, actions) in result.iter_arbitrary() {
        let mut state = State::new(problem);
        for &a in actions {
            state.do_action(problem, a);
        }
        assert!(clean.add(&c, &target, || state));
    }

    // it only makes sense to report done states at the end
    ctx.reporter.report_new_schedule(problem, &clean, &State::new(problem));

    println!("Recurse final cache size: {}", cache.len());

    // println!("Cache contents:");
    // for (k, v) in &cache {
    //     let v = match(v) {
    //         CacheEntry::Placeholder => unreachable!(),
    //         CacheEntry::Completed(v) => v,
    //     };
    //     let k_str = format!("{:?}", k);
    //     let k_str = k_str.replace(&i64::MIN.to_string(), "-inf");
    //     let k_str = k_str.replace(&i64::MAX.to_string(), "inf");
    //     println!("  {} {:?}", k_str, v.to_sorted_vec());
    // }

    clean
}

// TODO instead of dragging vecs around, just reconstruct based on the cache afterwards?
#[inline(never)]
fn recurse<R: ReporterRecurse>(ctx: &mut Context<R>, state: State) -> CostFrontier {
    let problem = ctx.problem;
    state.assert_valid(problem);

    // bookkeeping
    // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
    //   similar to the idea to improve the pruning in the other place?
    if state.is_done(problem) {
        assert_eq!(state.curr_time, state.minimum_time);
        return CostFrontier::single(Cost::default(), VecDeque::new());
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

    ctx.reporter.report_new_state(problem, &state);

    let mut frontier = CostFrontier::empty();
    let curr_cost = state.current_cost();
    let curr_actions_len = state.actions_taken.len();

    expand(problem, state, &mut |next_state| {
        // TODO avoid this clone?
        let delta_cost = next_state.current_cost() - curr_cost;
        let delta_actions = next_state.actions_taken[curr_actions_len..].to_vec();

        let next_frontier = recurse(ctx, next_state);

        for (entry_cost, mut entry_actions) in next_frontier.into_iter_arbitrary() {
            for &a in rev(&delta_actions) {
                entry_actions.push_front(a.inner);
            }
            frontier.add(&(entry_cost + delta_cost), &ctx.target, || entry_actions);
        }
    });

    let prev = ctx.cache.insert(achievement, CacheEntry::Completed(frontier.clone()));
    assert!(matches!(prev, Some(CacheEntry::Placeholder)));

    frontier
}

// TODO pick a better place for this
impl<V> Frontier<Cost, V> {
    pub fn to_sorted_vec(&self) -> Vec<Cost> {
        let mut v = self.keys().copied().collect_vec();
        v.sort_by_key(|e| e.time);
        v
    }
}
