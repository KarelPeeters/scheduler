use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::state::{Cost, State};

pub trait ReporterQueue {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, schedule: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, queue: &BinaryHeap<OrdState>, state: &State);
}

#[inline(never)]
pub fn solve_queue(problem: &Problem, target: CostTarget, reporter: &mut impl ReporterQueue) -> Frontier<Cost, State> {
    let root_state = State::new(problem);

    let mut frontier_done = Frontier::new();
    let mut frontier_partial = LinearFrontier::new(root_state.dom_key_min(problem, target).1);

    // TODO why only by cost and not by the full pareto key?
    let mut queue = BinaryHeap::new();

    // add root state
    if root_state.is_done(problem) {
        assert!(frontier_done.add(&root_state.current_cost(), &target, || root_state.clone()));
        reporter.report_new_schedule(problem, &frontier_done, root_state.current_cost(), &root_state);
        return frontier_done;
    }
    queue.push(OrdState::new(problem, target, root_state));

    // main loop
    while let Some(state) = queue.pop() {
        let mut state = state.state;
        if cfg!(debug_assertions) {
            state.assert_valid(problem);
        }

        // done should have been caught in next already
        assert!(!state.is_done(problem));

        // compare against existing done states
        if !frontier_done.would_add(&state.estimate_final_cost_conservative(problem), &target) {
            continue;
        }

        // drop dead values
        //   this needs to be done after every action, not just after waiting:
        //   actions might have made duplicate values in other memories dead
        // TODO is this the right place to do this? or should we already do this in next?
        //    this needs to happen before frontier_partial for sure
        if state.drop_dead_values(problem).is_err() {
            // pruned, dead unused value was used
            continue;
        }

        // compare against all existing states
        let added_linear = frontier_partial.add_if_not_dominated(state.dom_key_min(problem, target).0);
        if !added_linear {
            continue;
        }
        reporter.report_new_state(problem, &mut frontier_partial, &queue, &state);

        // expand the child states
        // TODO only do done check on states that have just waited?
        // TODO change this to a state class, this is just messy and confusing
        let mut next = |next_state: State| {
            if cfg!(debug_assertions) {
                next_state.assert_valid(problem);
            }

            // immediate report done states
            // TODO check how much this helps vs only reporting them when visited
            // TODO if not idle just cancel all those non-idle actions and report the better solution we get from it,
            //   similar to the idea to improve the pruning in the other place?
            if next_state.is_done(problem) {
                let was_added = frontier_done.add(&next_state.current_cost(), &target, || next_state.clone());
                if was_added {
                    reporter.report_new_schedule(problem, &frontier_done, next_state.current_cost(), &next_state);
                }
                return;
            }

            // immediately skip bad states here
            // (don't do full comparison yet, that can get expensive and maybe we never need to visit this state again)
            // TODO check how much this helps (and if doing the full would make it much slower)
            if !frontier_done.would_add(&next_state.estimate_final_cost_conservative(problem), &target) {
                return;
            }

            queue.push(OrdState::new(problem, target, next_state));
        };

        expand(problem, state, &mut next);
    }

    // println!("Partial frontier stats:");
    // println!("  add calls: {}", frontier_partial.add_calls);
    // println!("  success: {}, {}", frontier_partial.add_success, frontier_partial.add_success as f64 / frontier_partial.add_calls as f64);
    // println!("  dropped_old: {}, {}", frontier_partial.add_dropped_old, frontier_partial.add_dropped_old as f64 / frontier_partial.add_calls as f64);
    // println!("  total dropped: {}", frontier_partial.add_calls - frontier_partial.len() as u64);

    frontier_done
}

#[allow(dead_code)]
pub struct OrdState {
    target: CostTarget,
    state: State,
    cost: Cost,
}

impl OrdState {
    pub fn new(problem: &Problem, target: CostTarget, state: State) -> Self {
        let cost = state.estimate_final_cost_conservative(problem);
        Self { target, state, cost }
    }
}

impl PartialEq for OrdState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for OrdState {}

impl PartialOrd for OrdState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdState {
    fn cmp(&self, other: &Self) -> Ordering {
        let key = match self.target {
            // TODO is there anything we can do for full that's better than tiebreak towards time?
            CostTarget::Full | CostTarget::Time => |s: &OrdState| (s.cost.time.0, s.cost.energy.0),
            CostTarget::Energy => |s: &OrdState| (s.cost.energy.0, s.cost.time.0),
        };

        key(self).partial_cmp(&key(other)).unwrap().reverse()
    }
}