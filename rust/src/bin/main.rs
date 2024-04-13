#![allow(dead_code)]

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use itertools::{enumerate, Itertools};
use ordered_float::OrderedFloat;

use rust::core::frontier::Frontier;
use rust::core::linear_frontier::LinearFrontier;
use rust::core::new_frontier::NewFrontier;
use rust::core::problem::Problem;
use rust::core::solver::{Reporter, solve};
use rust::core::state::{Cost, State};
use rust::examples::params::{test_problem, TestGraphParams, TestHardwareParams};
use rust::util::mini::IterFloatExt;

fn main() {
    let problem = test_problem(
        TestGraphParams {
            depth: 0,
            branches: 0,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    
    problem.hardware.to_graphviz(problem.core_connected_memories()).export("ignored/hardware.svg").unwrap();
    problem.graph.to_graphviz().export("ignored/graph.svg").unwrap();
    problem.assert_valid();

    main_solver(&problem);
    // main_milp(&problem);
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
            let channel_value_delta = channel.time_to_transfer(node_info.size_bits);
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

fn main_solver(problem: &Problem) {
    let _ = std::fs::remove_dir_all("ignored/schedules/");
    std::fs::create_dir_all("ignored/schedules/done/").unwrap();
    std::fs::create_dir_all("ignored/schedules/partial/").unwrap();
    std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

    let mut reporter = CustomReporter {
        next_done_index: 0,
        next_partial_index: 0,
        state_counter: 0,
        start: Instant::now(),
        old_frontier_costs: vec![],
        partial_plot_frequency: 1000,
    };

    let start = Instant::now();

    let frontier = solve(&problem, &mut reporter);

    println!("Frontier:");
    for (c, _) in frontier.iter_arbitrary().sorted_by_key(|(c, _)| (OrderedFloat(c.time), OrderedFloat(c.energy))) {
        println!("  {:?}", c);
    }

    println!("Solver took {:?}s", start.elapsed().as_secs_f64());
}

struct CustomReporter {
    next_done_index: u64,
    next_partial_index: u64,
    state_counter: u64,
    start: Instant,
    old_frontier_costs: Vec<Cost>,
    partial_plot_frequency: u64,
}

impl Reporter for CustomReporter {
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

    fn report_new_state(&mut self, problem: &Problem, frontier: &mut Frontier<State, ()>, frontier_new: &mut NewFrontier, frontier_linear: &mut LinearFrontier, state: &State) {
        self.state_counter += 1;

        if self.state_counter % self.partial_plot_frequency == 0 {
            // println!("New state, state_counter={}, elapsed={:?}", self.state_counter, self.start.elapsed());

            if frontier.len() > 0 {
                println!("frontier:");
                println!("  len={}", frontier.len());
                println!("  add_count={}", frontier.count_add_try);
                println!("  add_success={}", frontier.count_add_success as f64 / frontier.count_add_try as f64);
                println!("  add_removed={}", frontier.count_add_removed as f64 / frontier.count_add_try as f64);
                frontier.count_add_try = 0;
                frontier.count_add_success = 0;
                frontier.count_add_removed = 0;
            }

            if frontier_new.len() > 0 {
                println!("frontier_new: len={}", frontier_new.len());

                let depths = format!("{:?}", frontier_new.collect_entry_depths());
                std::fs::write("ignored/depths_new.txt", &depths).unwrap();
            }

            if frontier_linear.len() > 0 {
                println!("frontier_linear: len={}", frontier_linear.len());

                let depths = format!("{:?}", frontier_linear.collect_entry_depths());
                std::fs::write("ignored/depths_linear.txt", &depths).unwrap();
            }

            let index = self.next_partial_index;
            self.next_partial_index += 1;
            state.write_svg_to_file(&problem, format!("ignored/schedules/partial/{index}.svg")).unwrap();
        }
    }
}
