use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::Entry;

use itertools::{Itertools, rev};

use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::schedule::Action;
use crate::core::state::{Cost, State};
use crate::util::mini::min_option;

pub trait ReporterRecurse {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, state: &State);
}

pub type CostFrontier = Frontier<Cost, VecDeque<Action>>;
pub type RecurseCache = HashMap<Vec<i64>, CompletedCacheEntry>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct LoopDepth(u32);

#[derive(Clone)]
enum CacheEntry {
    Placeholder(LoopDepth),
    Completed(Option<LoopDepth>, CompletedCacheEntry),
}

#[derive(Clone)]
pub struct CompletedCacheEntry {
    pub example_state: State,
    pub frontier: CostFrontier,
}

pub struct Context<'p, 'r, 'f, R: ReporterRecurse> {
    problem: &'p Problem,
    target: CostTarget,
    reporter: &'r mut R,
    cache: &'f mut HashMap<Vec<i64>, CacheEntry>
}

pub fn solve_recurse(problem: &Problem, target: CostTarget, reporter: &mut impl ReporterRecurse) -> (Frontier<Cost, State>, RecurseCache) {
    let state = State::new(problem);

    let mut cache = HashMap::new();

    let mut ctx = Context {
        problem,
        target,
        reporter,
        cache: &mut cache,
    };

    let (_, result) = recurse(&mut ctx, state, 0);

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

    let clean_cache = cache.into_iter().map(|(k, v)| (k.to_owned(), match v {
        CacheEntry::Placeholder { .. } => unreachable!(),
        CacheEntry::Completed(_, v) => v,
    })).collect();

    (clean, clean_cache)
}

#[inline(never)]
fn recurse<R: ReporterRecurse>(ctx: &mut Context<R>, state: State, depth: u32) -> (Option<LoopDepth>, CostFrontier) {
    // println!("recurse, depth={}", depth);

    let problem = ctx.problem;
    state.assert_valid(problem);

    // bookkeeping
    // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
    //   similar to the idea to improve the pruning in the other place?
    if state.is_done(problem) {
        assert_eq!(state.curr_time, state.minimum_time);
        return (None, CostFrontier::single(Cost::default(), VecDeque::new()));
    }

    let achievement = state.achievement(problem);
    let state_clone = state.clone();
    match ctx.cache.entry(achievement.clone()) {
        Entry::Occupied(entry) => {
            match *entry.get() {
                // hit loop, which is useless
                CacheEntry::Placeholder(depth) => {
                    return (Some(depth), CostFrontier::empty());
                }
                // cache hit
                CacheEntry::Completed(cache_depth, ref entry) => {
                    let accept = match cache_depth {
                        None => true,
                        Some(LoopDepth(cache_depth)) => cache_depth >= depth,
                    };

                    if accept {
                        return (None, entry.frontier.clone());
                    }
                },
            }
        }
        Entry::Vacant(entry) => {
            // first time seeing this state, mark for loop detection
            entry.insert(CacheEntry::Placeholder(LoopDepth(depth)));
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
    let mut min_loop_depth = None;

    let curr_cost = state.current_cost();
    let curr_actions_len = state.actions_taken.len();

    expand(problem, state, &mut |next_state| {
        // TODO avoid this clone?
        let delta_cost = next_state.current_cost() - curr_cost;
        let delta_actions = next_state.actions_taken[curr_actions_len..].to_vec();

        let (next_min_loop_depth, next_frontier) = recurse(ctx, next_state, depth+1);
        min_loop_depth = min_option(min_loop_depth, next_min_loop_depth);

        for (entry_cost, mut entry_actions) in next_frontier.into_iter_arbitrary() {
            for &a in rev(&delta_actions) {
                entry_actions.push_front(a.inner);
            }
            frontier.add(&(entry_cost + delta_cost), &ctx.target, || entry_actions);
        }
    });

    println!("depth={depth}, min_loop_depth={min_loop_depth:?}");

    // TODO LT or LEQ?
    let insert_in_cache = min_loop_depth.map_or(true, |min_loop_depth| depth < min_loop_depth.0);
    println!("insert_in_cache={}", insert_in_cache);

    // let cache_prev = if insert_in_cache {
        let entry = CacheEntry::Completed(min_loop_depth, CompletedCacheEntry {
            example_state: state_clone,
            frontier: frontier.clone(),
        });
        let cache_prev = ctx.cache.insert(achievement, entry);
    // } else {
    //     ctx.cache.remove(&achievement)
    // };
    assert!(matches!(cache_prev, Some(CacheEntry::Placeholder(LoopDepth(_)))));

    (min_loop_depth, frontier)
}

// TODO pick a better place for this
impl<V> Frontier<Cost, V> {
    pub fn to_sorted_vec(&self) -> Vec<Cost> {
        let mut v = self.keys().copied().collect_vec();
        v.sort_by_key(|e| e.time);
        v
    }
}
