use crate::core::expand::expand;
use crate::core::frontier::Frontier;
use crate::core::linear_frontier::LinearFrontier;
use crate::core::problem::{CostTarget, Problem};
use crate::core::state::{Cost, State};

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
}

pub fn solve_recurse(problem: &Problem, target: CostTarget, reporter: &mut impl ReporterRecurse) -> Frontier<Cost, State> {
    let state = State::new(problem);

    let mut frontier_done = Frontier::new();
    let mut frontier_partial_linear = LinearFrontier::new(state.dom_key_min(problem, target).1);

    let mut ctx = Context {
        problem,
        target,
        reporter,
        frontier_done: &mut frontier_done,
        frontier_partial: &mut frontier_partial_linear,
    };

    recurse(&mut ctx, state);

    frontier_done
}

// TODO split this up into smaller functions
#[inline(never)]
fn recurse<R: ReporterRecurse>(ctx: &mut Context<R>, mut state: State) {
    let problem = ctx.problem;
    state.assert_valid(problem);

    // drop dead values
    //   this needs to be done after every action, not just after waiting:
    //   actions might have made duplicate values in other memories dead
    if state.drop_dead_values(problem).is_err() {
        // pruned, dead unused value was used
        return;
    }

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

        return;
    }

    // pruning
    if !ctx.frontier_done.would_add(&state.estimate_final_cost_conservative(problem), &ctx.target) {
        return;
    }
    let added_partial = ctx.frontier_partial.add_if_not_dominated(state.dom_key_min(problem, ctx.target).0);
    if !added_partial {
        return;
    }
    ctx.reporter.report_new_state(problem, ctx.frontier_partial, &state);

    expand(problem, state, &mut |next_state| {
        recurse(ctx, next_state);
    });
}
