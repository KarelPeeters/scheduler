#![allow(dead_code)]

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use itertools::{enumerate, Itertools};
use ordered_float::OrderedFloat;

use rust::core::frontier::Frontier;
use rust::core::problem::{AllocationInfo, ChannelInfo, CoreInfo, Direction, Graph, Hardware, MemoryInfo, NodeInfo, Problem};
use rust::core::solver::{Reporter, solve};
use rust::core::state::{Cost, State};
use rust::util::mini::IterFloatExt;

fn main() {
    let problem = build_problem();
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
        
        max_time += problem.allocation_info.iter().filter(|a| a.node == node).map(|a| a.time).min_float().unwrap();
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
        next_index: 0,
        state_counter: 0,
        start: Instant::now(),
    };

    solve(&problem, &mut reporter);
}

struct CustomReporter {
    next_index: u64,
    state_counter: u64,
    start: Instant,
}

impl Reporter for CustomReporter {
    fn report_new_schedule(&mut self, problem: &Problem, frontier: &Frontier<Cost, State>, cost: Cost, state: &State) {
        let index = self.next_index;
        self.next_index += 1;

        println!("New done schedule, index={}, cost={:?}, frontier_size={}", index, cost, frontier.len());
        state.write_svg_to_file(&problem, format!("ignored/schedules/done/{index}.svg")).unwrap();

        // clear frontier dir
        std::fs::remove_dir_all("ignored/schedules/frontier/").unwrap();
        std::fs::create_dir_all("ignored/schedules/frontier/").unwrap();

        // save entire frontier
        for (i, (_, state)) in enumerate(frontier.iter_arbitrary()) {
            state.write_svg_to_file(&problem, format!("ignored/schedules/frontier/{i}.svg")).unwrap();
        }
    }

    fn report_new_state(&mut self, problem: &Problem, frontier: &Frontier<State, ()>, state: &State) {
        self.state_counter += 1;

        if self.state_counter % 1_000 == 0 {
            println!("New state, state_counter={}, frontier_len={}, elapsed={:?}", self.state_counter, frontier.len(), self.start.elapsed());

            let index = self.next_index;
            self.next_index += 1;
            state.write_svg_to_file(&problem, format!("ignored/schedules/partial/{index}.svg")).unwrap();
        }
    }
}

fn build_problem() -> Problem {
    // parameters
    let hardware_depth = 4;
    let mem_size_ext = None;
    let mem_size_chip = None;
    let bandwidth_ext = 1.0;
    let bandwidth_chip = 2.0;
    let energy_ext = 2.0;
    let energy_chip = 1.0;
    let alloc_time = 4000.0;
    let alloc_energy = 100.0;

    let graph_depth = 4;
    let graph_branching = 2;
    let graph_node_size = 1000;
    let graph_cross = true;

    // hardware
    let mut hardware = Hardware::new("hardware");

    let mem_ext = hardware.add_memory(MemoryInfo { id: "mem_ext".to_owned(), size_bits: mem_size_ext });

    let mut mem_core = vec![];
    let mut cores = vec![];

    for i in 0..hardware_depth {
        cores.push(hardware.add_core(CoreInfo { id: format!("core_{}", i) }));

        let mem_curr = hardware.add_memory(MemoryInfo { id: format!("mem_chip_{}", i), size_bits: mem_size_chip });
        mem_core.push(mem_curr);

        let (id, prev, bandwidth, energy) = if i == 0 {
            (format!("channel_ext_0"), mem_ext, bandwidth_ext, energy_ext)
        } else {
            (format!("channel_chip_{}", i), mem_core[i - 1], bandwidth_chip, energy_chip)
        };

        let channel_info = ChannelInfo {
            id,
            mem_a: prev,
            mem_b: mem_curr,
            dir: Direction::Both,
            latency: 0.0,
            time_per_bit: 1.0 / bandwidth,
            energy_per_bit: energy,
        };
        hardware.add_channel(channel_info);
    }

    // graph
    let mut graph = Graph::new("graph");
    let node_input = graph.add_node(NodeInfo {
        id: format!("node-input"),
        size_bits: graph_node_size,
        inputs: vec![],
    });
    graph.add_input(node_input);
    let mut prev = vec![node_input];
    for i_depth in 0..graph_depth {
        let next = (0..graph_branching).map(|i_branch| {
           let inputs = if graph_cross || i_depth == 0 {
               prev.clone()
           } else {
               vec![prev[i_branch]]
           };
            graph.add_node(NodeInfo {
                id: format!("node-{}{}", i_depth, (b'a' + i_branch as u8) as char),
                size_bits: graph_node_size,
                inputs,
            })
        }).collect_vec();
        prev = next;
    }
    let node_output = graph.add_node(NodeInfo {
        id: format!("node-output"),
        size_bits: graph_node_size,
        inputs: prev,
    });
    graph.add_output(node_output);

    // allocations
    let mut allocations = vec![];
    for (i, &core) in enumerate(&cores) {
        for node in graph.nodes() {
            allocations.push(AllocationInfo {
                id: format!("basic"),
                core,
                node,
                input_memories: vec![mem_core[i]; graph.node_info[node.0].inputs.len()],
                output_memory: mem_core[i],
                time: alloc_time,
                energy: alloc_energy,
            })
        }
    }

    // boundary conditions
    let input_placements = vec![mem_ext; graph.inputs.len()];
    let output_placements = vec![mem_ext; graph.outputs.len()];

    Problem {
        id: "problem".to_owned(),
        hardware,
        graph,
        allocation_info: allocations,
        input_placements,
        output_placements,
    }
}
