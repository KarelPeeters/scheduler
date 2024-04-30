#![allow(dead_code)]

use std::alloc::alloc;
use std::any::Any;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::env::vars;
use std::os::unix::raw::ino_t;
use std::time::Instant;

use itertools::{enumerate, Itertools};
use ordered_float::OrderedFloat;

use rust::core::frontier::Frontier;
use rust::core::linear_frontier::LinearFrontier;
use rust::core::problem::{CostTarget, Graph, Group, Node, NodeInfo, Problem};
use rust::core::solve::{CommonReporter, SolveMethod};
use rust::core::solve_queue::OrdState;
use rust::core::state::{Cost, State};
use rust::examples::params::{test_problem, TestGraphParams, TestHardwareParams};
use rust::examples::{DEFAULT_CHANNEL_COST_EXT, DEFAULT_CHANNEL_COST_INT};
use rust::util::float::IterFloatExt;
use z3::ast::{Ast, Int};
use z3::{Config, Context, Optimize};

fn main() {
    let problem = test_problem(
        TestGraphParams {
            depth: 5,
            branches: 4,
            cross: true,
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
    // let target = CostTarget::Full;
    // let method = SolveMethod::Queue;
    // let partial_plot_frequency = 1000;

    // main_solver(&problem, target, method, partial_plot_frequency);
    main_z3(&problem);
}

fn main_z3(problem: &Problem) {
    // println!("{:?}", problem);
    let time_quant = 1.0;
    let energy_quant = 1.0;

    // z3::set_global_param("verbose", "10");

    let cfg = Config::new();
    let ctx = &Context::new(&cfg);
    let opt = Optimize::new(ctx);

    // TODO solve real problem:
    // * allow multiple allocations per node
    // * implement value copying
    // TODO investigate
    // * push/pop for partial models, eg. when changing the optimization target?
    // * what does model_completion mean exactly?
    // TODO double check that all lt/gt constraints include equality?

    let var_zero = Int::from_u64(ctx, 0);
    let var_max_end = Int::new_const(ctx, "max_end");

    let mut groups: HashMap<Group, Vec<(Node, Int, Int)>> = HashMap::new();

    let var_node_start_end = problem
        .graph
        .nodes()
        .map(|node| {
            if problem.graph.inputs.contains(&node) {
                (None, var_zero.clone())
            } else {
                let allocation = &problem.allocation_info[problem
                    .allocations()
                    .find(|a| problem.allocation_info[a.0].node == node)
                    .unwrap()
                    .0];
                let time = (allocation.time / time_quant) as u64;

                let start = Int::new_const(ctx, format!("start_{}", node.0));
                let end = start.clone() + time;

                groups.entry(allocation.group).or_default().push((
                    node,
                    start.clone(),
                    end.clone(),
                ));

                (Some(start), end)
            }
        })
        .collect_vec();

    // prevent group overlaps
    for ranges in groups.values() {
        for i in 0..ranges.len() {
            let &(i_node, ref i_start, ref i_end) = &ranges[i];

            for j in i + 1..ranges.len() {
                let &(j_node, ref j_start, ref j_end) = &ranges[j];

                // no need to constraint nodes that already depend on each other
                let i_depends_j = graph_node_depends_on(&problem.graph, i_node, j_node);
                let j_depends_i = graph_node_depends_on(&problem.graph, j_node, i_node);
                if i_depends_j || j_depends_i {
                    continue;
                }

                // ranges can't overlap
                let i_before = i_end.le(j_start);
                let i_after = i_start.ge(j_end);
                opt.assert(&(i_before | i_after));
            }
        }
    }

    for node in problem.graph.nodes() {
        let (node_start, node_end) = &var_node_start_end[node.0];

        // node finishes before max end
        opt.assert(&node_end.le(&var_max_end));

        // input nodes don't need constraints
        if let Some(node_start) = node_start {
            // node starts at positive time
            opt.assert(&node_start.ge(&var_zero));

            let node_info = &problem.graph.node_info[node.0];
            for &input in &node_info.inputs {
                // input finishes before this node starts
                let input_end = &var_node_start_end[input.0].1;
                opt.assert(&(node_start.ge(input_end)));
            }
        }
    }

    // TODO add more constraints that skip multiple levels in the graph?
    //   or can the solver easily generate these?
    for i_node in problem.graph.nodes() {
        if let Some(i_start) = &var_node_start_end[i_node.0].0 {
            for j_node in problem.graph.nodes() {
                if i_node == j_node {
                    continue;
                }

                let j_end = &var_node_start_end[j_node.0].1;
                if graph_node_depends_on(&problem.graph, i_node, j_node) {
                    opt.assert(&(i_start.ge(j_end)));
                }
            }
        }
    }

    opt.minimize(&var_max_end);

    println!("Model: {}", opt);

    let start = Instant::now();
    let result = opt.check(&[]);
    println!("{:?}, took {:?}", result, start.elapsed());

    let model = opt.get_model().unwrap();

    for node in problem.graph.nodes() {
        let (node_start, node_end) = &var_node_start_end[node.0];
        let node_start = node_start.as_ref().map(|s| model.get_const_interp(s));
        let node_end = model.eval(node_end, false);

        println!("Node {:?}: {:?}..{:?}", node, node_start, node_end);
    }
}

fn graph_node_depends_on(graph: &Graph, node: Node, potential: Node) -> bool {
    (node == potential)
        || graph.node_info[node.0]
            .inputs
            .iter()
            .any(|&input| graph_node_depends_on(graph, input, potential))
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

        max_time += problem
            .allocation_info
            .iter()
            .filter(|a| a.node == node)
            .map(|a| a.time)
            .min_f64()
            .unwrap();
    }

    // collect all possible times
    // TODO upper bound for this?
    let mut visited = HashSet::new();
    let mut todo = VecDeque::new();
    todo.push_back(0.0);
    while let Some(time_curr) = todo.pop_front() {
        if time_curr > max_time {
            continue;
        }

        if !visited.insert(OrderedFloat(time_curr)) {
            continue;
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

fn main_solver(
    problem: &Problem,
    target: CostTarget,
    method: SolveMethod,
    partial_plot_frequency: u64,
) {
    let _ = std::fs::remove_dir_all("ignored/schedules/");
    std::fs::create_dir_all("ignored/schedules/done/").unwrap();
    std::fs::create_dir_all("ignored/schedules/partial/").unwrap();
    std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

    problem
        .hardware
        .to_graphviz(problem.core_connected_memories())
        .export("ignored/hardware.svg")
        .unwrap();
    problem
        .graph
        .to_graphviz()
        .export("ignored/graph.svg")
        .unwrap();
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
    for (c, _) in frontier
        .iter_arbitrary()
        .sorted_by_key(|(c, _)| (OrderedFloat(c.time), OrderedFloat(c.energy)))
    {
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
    fn report_new_schedule(
        &mut self,
        problem: &Problem,
        frontier: &Frontier<Cost, State>,
        _cost: Cost,
        state: &State,
    ) {
        let index = self.next_done_index;
        self.next_done_index += 1;

        // println!("New done schedule, index={}, cost={:?}, frontier_size={}", index, cost, frontier.len());
        state
            .write_svg_to_file(&problem, format!("ignored/schedules/done/{index}.svg"))
            .unwrap();

        // clear frontier dir
        std::fs::remove_dir_all("ignored/schedules/frontier/").unwrap();
        std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

        // save entire frontier
        let mut frontier_pairs = frontier.iter_arbitrary().collect_vec();
        frontier_pairs.sort_by_key(|(c, _)| OrderedFloat(c.time));
        for (i, (_, state)) in enumerate(frontier_pairs) {
            state
                .write_svg_to_file(&problem, format!("ignored/schedules/frontier/{i}.svg"))
                .unwrap();
        }

        for (&cost, _) in frontier.iter_arbitrary() {
            if !self.old_frontier_costs.contains(&cost) {
                self.old_frontier_costs.push(cost);
            }
        }

        // TODO include values states that were ever in the frontier?
        //   impl: just store in this reporter and change the plot function to take a list
        frontier
            .write_svg_to_file(&self.old_frontier_costs, "ignored/schedules/frontier.svg")
            .unwrap();
    }

    fn report_new_state(
        &mut self,
        problem: &Problem,
        frontier_partial: &mut LinearFrontier,
        queue: Option<&BinaryHeap<OrdState>>,
        state: &State,
    ) {
        self.state_counter += 1;

        if self.state_counter % self.partial_plot_frequency == 0 {
            let index = self.next_partial_index;
            self.next_partial_index += 1;

            let queue_len = queue.map(|q| q.len());
            let success = frontier_partial.add_success as f64 / frontier_partial.add_calls as f64;
            let dropped =
                frontier_partial.add_dropped_old as f64 / frontier_partial.add_calls as f64;
            let checked =
                frontier_partial.entries_checked as f64 / frontier_partial.add_calls as f64;
            println!(
                "Partial state: index={}: queue_len={:?}, frontier_len={}, success={:.04}, dropped={:.04}, checked={:.04}",
                index, queue_len, frontier_partial.len(), success, dropped, checked
            );
            frontier_partial.clear_stats();

            // let depths = format!("{:?}", frontier_partial.collect_entry_depths());
            // std::fs::write("ignored/depths_linear.txt", &depths).unwrap();

            state
                .write_svg_to_file(&problem, format!("ignored/schedules/partial/{index}.svg"))
                .unwrap();
            std::fs::write(
                format!("ignored/schedules/partial/{index}.txt"),
                state.summary_string(problem),
            )
            .unwrap();
        }
    }
}
