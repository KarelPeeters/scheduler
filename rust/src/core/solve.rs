use std::collections::BinaryHeap;

use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::solve_queue::{OrdState, ReporterQueue, solve_queue};
use crate::core::solve_recurse::{RecurseCache, ReporterRecurse, solve_recurse};
use crate::core::state::{Cost, State};

#[derive(Debug, Copy, Clone)]
pub enum SolveMethod {
    Queue,
    Recurse,
}

impl SolveMethod {
    pub fn solve(self, problem: &Problem, target: CostTarget, reporter: &mut impl CommonReporter) -> (Frontier<Cost, State>, Option<RecurseCache>) {
        match self {
            SolveMethod::Queue => (solve_queue(problem, target, reporter), None),
            SolveMethod::Recurse => {
                let (frontier, cache) = solve_recurse(problem, target, reporter);
                (frontier, Some(cache))
            },
        }
    }
}

pub trait CommonReporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, state: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: Option<&mut LinearFrontier>, queue: Option<&BinaryHeap<OrdState>>, state: &State);
}

impl<R: CommonReporter> ReporterQueue for R {
    #[inline(never)]
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, state: &State) {
        self.report_new_schedule(problem, frontier, state)
    }

    #[inline(never)]
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, queue: &BinaryHeap<OrdState>, state: &State) {
        self.report_new_state(problem, Some(frontier_partial), Some(queue), state)
    }
}

impl<R: CommonReporter> ReporterRecurse for R {
    #[inline(never)]
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, schedule: &State) {
        self.report_new_schedule(problem, frontier_done, schedule)
    }

    #[inline(never)]
    fn report_new_state(&mut self, problem: &Problem, state: &State) {
        self.report_new_state(problem, None, None, state)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DummyReporter;

impl CommonReporter for DummyReporter {
    fn report_new_schedule(&mut self, _: &Problem, _: &Frontier<Cost, State>, _: &State) {}
    fn report_new_state(&mut self, _: &Problem, _: Option<&mut LinearFrontier>, _: Option<&BinaryHeap<OrdState>>, _: &State) {}
}
