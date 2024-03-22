use itertools::{enumerate, Itertools};

use rust::core::problem::{AllocationInfo, ChannelInfo, CoreInfo, Direction, Graph, Hardware, MemoryInfo, NodeInfo, Problem};

fn main() {
    let problem = build_problem();

    problem.hardware.to_graphviz(problem.core_connected_memories()).export("ignored/hardware.svg").unwrap();
    problem.graph.to_graphviz().export("ignored/graph.svg").unwrap();

    problem.assert_valid();
}

fn build_problem() -> Problem {
    // parameters
    let hardware_depth = 2;
    let mem_size_ext = None;
    let mem_size_chip = None;
    let bandwidth_ext = 1.0;
    let bandwidth_chip = 2.0;
    let energy_ext = 2.0;
    let energy_chip = 1.0;
    let alloc_time = 4000.0;
    let alloc_energy = 100.0;

    let graph_depth = 2;
    let graph_branching = 3;
    let graph_node_size = 1000;

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
    let mut prev = vec![node_input; graph_branching];
    for i_depth in 0..graph_depth {
        prev = prev.iter().enumerate().map(|(i_branch, &prev)| graph.add_node(NodeInfo {
            id: format!("node-{}{}", i_depth, (b'a' + i_branch as u8) as char),
            size_bits: graph_node_size,
            inputs: vec![prev],
        })).collect_vec();
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
                input_memories: vec![mem_core[i]],
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
        allocations,
        input_placements,
        output_placements,
    }
}
