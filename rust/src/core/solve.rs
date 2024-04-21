use std::collections::BinaryHeap;

use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::solve_queue::{OrdState, ReporterQueue, solve_queue};
use crate::core::solve_recurse::{ReporterRecurse, solve_recurse};
use crate::core::state::{Cost, State};

#[derive(Debug, Copy, Clone)]
pub enum SolveMethod {
    Queue,
    Recurse,
}

impl SolveMethod {
    pub fn solve(self, problem: &Problem, target: CostTarget, reporter: &mut impl CommonReporter) -> Frontier<Cost, State> {
        match self {
            SolveMethod::Queue => solve_queue(problem, target, reporter),
            SolveMethod::Recurse => solve_recurse(problem, target, reporter),
        }
    }
}

pub trait CommonReporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, state: &State);
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, queue: Option<&BinaryHeap<OrdState>>, state: &State);
}

impl<R: CommonReporter> ReporterQueue for R {
    #[inline(never)]
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, cost: Cost, state: &State) {
        self.report_new_schedule(problem, frontier, cost, state)
    }

    #[inline(never)]
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, queue: &BinaryHeap<OrdState>, state: &State) {
        self.report_new_state(problem, frontier_partial, Some(queue), state)
    }
}

impl<R: CommonReporter> ReporterRecurse for R {
    #[inline(never)]
    fn report_new_schedule(&mut self, problem: &Problem, frontier_done: &Frontier<Cost, State>, cost: Cost, schedule: &State) {
        self.report_new_schedule(problem, frontier_done, cost, schedule)
    }

    #[inline(never)]
    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, state: &State) {
        self.report_new_state(problem, frontier_partial, None, state)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DummyReporter;

impl CommonReporter for DummyReporter {
    fn report_new_schedule(&mut self, _: &Problem, _: &Frontier<Cost, State>, _: Cost, _: &State) {}
    fn report_new_state(&mut self, _: &Problem, _: &mut LinearFrontier, _: Option<&BinaryHeap<OrdState>>, _: &State) {}
}
