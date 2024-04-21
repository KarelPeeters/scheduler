#![allow(dead_code)]

use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::time::Instant;

use itertools::{enumerate, Itertools};
use ordered_float::OrderedFloat;

use rust::core::frontier::Frontier;
use rust::core::linear_frontier::LinearFrontier;
use rust::core::problem::{CostTarget, Problem};
use rust::core::solve::{CommonReporter, SolveMethod};
use rust::core::solve_queue::OrdState;
use rust::core::state::{Cost, State};
use rust::examples::{DEFAULT_CHANNEL_COST_EXT, DEFAULT_CHANNEL_COST_INT};
use rust::examples::params::{test_problem, TestGraphParams, TestHardwareParams};
use rust::util::float::IterFloatExt;

fn main() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 4,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    let target = CostTarget::Full;
    let method = SolveMethod::Queue;
    let partial_plot_frequency = 1000;
    
    main_solver(&problem, target, method, partial_plot_frequency);
}

fn main_milp(problem: &Problem) {
    let mut max_time = 0.0;
    let mut deltas = HashSet::new();

    for alloc in &problem.allocation_info {
        deltas.insert(OrderedFloat(alloc.time));
    }
    for node in problem.graph.nodes() {
        let node_info = &problem.graph.node_info[node.0];

        for channel in &problem.hardware.channel_info {
            let channel_value_delta = channel.cost.time_to_transfer(node_info.size_bits);
            deltas.insert(OrderedFloat(channel_value_delta));

            // TODO we may need to copy some values across a channel multiple times, so this is not a perfect bound
            max_time += channel_value_delta;
        }

        max_time += problem.allocation_info.iter().filter(|a| a.node == node).map(|a| a.time).min_f64().unwrap();
    }

    // collect all possible times
    // TODO upper bound for this?
    let mut visited = HashSet::new();
    let mut todo = VecDeque::new();
    todo.push_back(0.0);
    while let Some(time_curr) = todo.pop_front() {
        if time_curr > max_time {
            continue
        }

        if !visited.insert(OrderedFloat(time_curr)) {
            continue
        }
        for delta in &deltas {
            todo.push_back(time_curr + delta.0);
        }
    }

    let visited = visited.iter().copied().sorted().map(|x| x.0).collect_vec();
    println!("{:?}", visited);
    println!("Distinct time count: {}", visited.len());
    println!("Max time: {}", max_time);
}

fn main_solver(problem: &Problem, target: CostTarget, method: SolveMethod, partial_plot_frequency: u64) {
    let _ = std::fs::remove_dir_all("ignored/schedules/");
    std::fs::create_dir_all("ignored/schedules/done/").unwrap();
    std::fs::create_dir_all("ignored/schedules/partial/").unwrap();
    std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

    problem.hardware.to_graphviz(problem.core_connected_memories()).export("ignored/hardware.svg").unwrap();
    problem.graph.to_graphviz().export("ignored/graph.svg").unwrap();
    problem.assert_valid();

    let mut reporter = CustomReporter {
        next_done_index: 0,
        next_partial_index: 0,
        state_counter: 0,
        start: Instant::now(),
        old_frontier_costs: vec![],
        partial_plot_frequency,
    };

    let start = Instant::now();

    println!("Starting solver");
    let frontier = method.solve(problem, target, &mut reporter);
    
    let solver_elapsed = start.elapsed();

    println!("Frontier:");
    for (c, _) in frontier.iter_arbitrary().sorted_by_key(|(c, _)| (OrderedFloat(c.time), OrderedFloat(c.energy))) {
        println!("  {:?}", c);
    }

    println!("Solver took {:?}s", solver_elapsed.as_secs_f64());
}

struct CustomReporter {
    next_done_index: u64,
    next_partial_index: u64,
    state_counter: u64,
    start: Instant,
    old_frontier_costs: Vec<Cost>,
    partial_plot_frequency: u64,
}

impl CommonReporter for CustomReporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, _cost: Cost, state: &State) {
        let index = self.next_done_index;
        self.next_done_index += 1;

        // println!("New done schedule, index={}, cost={:?}, frontier_size={}", index, cost, frontier.len());
        state.write_svg_to_file(&problem, format!("ignored/schedules/done/{index}.svg")).unwrap();

        // clear frontier dir
        std::fs::remove_dir_all("ignored/schedules/frontier/").unwrap();
        std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

        // save entire frontier
        let mut frontier_pairs = frontier.iter_arbitrary().collect_vec();
        frontier_pairs.sort_by_key(|(c, _)| OrderedFloat(c.time));
        for (i, (_, state)) in enumerate(frontier_pairs) {
            state.write_svg_to_file(&problem, format!("ignored/schedules/frontier/{i}.svg")).unwrap();
        }

        for (&cost, _) in frontier.iter_arbitrary() {
            if !self.old_frontier_costs.contains(&cost) {
                self.old_frontier_costs.push(cost);
            }
        }

        // TODO include values states that were ever in the frontier?
        //   impl: just store in this reporter and change the plot function to take a list
        frontier.write_svg_to_file(&self.old_frontier_costs, "ignored/schedules/frontier.svg").unwrap();
    }

    fn report_new_state(&mut self, problem: &Problem, frontier_partial: &mut LinearFrontier, queue: Option<&BinaryHeap<OrdState>>, state: &State) {
        self.state_counter += 1;

        if self.state_counter % self.partial_plot_frequency == 0 {
            let index = self.next_partial_index;
            self.next_partial_index += 1;

            let queue_len = queue.map(|q| q.len());
            let success = frontier_partial.add_success as f64 / frontier_partial.add_calls as f64;
            let dropped = frontier_partial.add_dropped_old as f64 / frontier_partial.add_calls as f64;
            let checked = frontier_partial.entries_checked as f64 / frontier_partial.add_calls as f64;
            println!(
                "Partial state: index={}: queue_len={:?}, frontier_len={}, success={:.04}, dropped={:.04}, checked={:.04}",
                index, queue_len, frontier_partial.len(), success, dropped, checked
            );
            frontier_partial.clear_stats();

            // let depths = format!("{:?}", frontier_partial.collect_entry_depths());
            // std::fs::write("ignored/depths_linear.txt", &depths).unwrap();

            state.write_svg_to_file(&problem, format!("ignored/schedules/partial/{index}.svg")).unwrap();
            std::fs::write(format!("ignored/schedules/partial/{index}.txt"), state.summary_string(problem)).unwrap();
        }
    }
}
