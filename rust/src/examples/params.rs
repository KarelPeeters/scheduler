use itertools::{enumerate, Itertools};
use crate::core::problem::{AllocationInfo, ChannelInfo, Graph, GroupInfo, Hardware, MemoryInfo, NodeInfo, Problem};

#[derive(Debug, Clone)]
pub struct TestGraphParams {
    pub depth: usize,
    pub branches: usize,
    pub cross: bool,

    pub node_size: u64,
    pub weight_size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TestHardwareParams {
    pub core_count: usize,
    // TODO option to share both channels with the core?
    pub share_group: bool,

    pub mem_size_ext: Option<u64>,
    pub mem_size_int: Option<u64>,

    pub time_per_bit_ext: f64,
    pub time_per_bit_int: f64,
    pub energy_per_bit_ext: f64,
    pub energy_per_bit_int: f64,
}

pub fn test_problem(graph_params: TestGraphParams, hardware_params: TestHardwareParams, allocs_time_energy: &[(&str, f64, f64)]) -> Problem {
    // hardware
    let mut hardware = Hardware::new("hardware");

    let mem_ext = hardware.add_memory(MemoryInfo { id: "mem_ext".to_owned(), size_bits: hardware_params.mem_size_ext });

    let mut mem_core = vec![];
    let mut core_groups = vec![];

    for i_core in 0..hardware_params.core_count {
        // core group
        let core_group = hardware.add_group(GroupInfo { id: format!("core_{i_core}") });
        core_groups.push(core_group);

        // memory
        let mem_curr = hardware.add_memory(MemoryInfo { id: format!("mem_chip_{}", i_core), size_bits: hardware_params.mem_size_int });
        mem_core.push(mem_curr);

        // channel
        let (channel_id, mem_prev, time_per_bit, energy_per_bit) = if i_core == 0 {
            ("channel_ext_0".to_string(), mem_ext, hardware_params.time_per_bit_ext, hardware_params.energy_per_bit_ext)
        } else {
            (format!("channel_chip_{}", i_core), mem_core[i_core - 1], hardware_params.time_per_bit_int, hardware_params.energy_per_bit_int)
        };
        let channel_group = if hardware_params.share_group {
            core_group
        } else {
            hardware.add_group(GroupInfo { id: channel_id.clone() })
        };
        for (dir, mem_source, mem_dest) in [("fwd", mem_prev, mem_curr), ("bck", mem_curr, mem_prev)] {
            let channel_info = ChannelInfo {
                id: format!("{channel_id}_{dir}"),
                group: channel_group,
                mem_source,
                mem_dest,
                latency: 0.0,
                time_per_bit,
                energy_per_bit,
            };
            hardware.add_channel(channel_info);
        }
    }

    // graph
    assert!(graph_params.branches > 0 || graph_params.depth == 0);
    
    let mut graph = Graph::new("graph");
    let node_input = graph.add_node(NodeInfo {
        id: "node-input".to_string(),
        size_bits: graph_params.node_size,
        inputs: vec![],
    });
    graph.add_input(node_input);
    let mut prev = vec![node_input];
    for i_depth in 0..graph_params.depth {
        let next = (0..graph_params.branches).map(|i_branch| {
            let mut inputs = if graph_params.cross || i_depth == 0 {
                prev.clone()
            } else {
                vec![prev[i_branch]]
            };

            if let Some(graph_weight_size) = graph_params.weight_size {
                let weight = graph.add_node(NodeInfo {
                    id: format!("weight-{}{}", i_depth, (b'a' + i_branch as u8) as char),
                    size_bits: graph_weight_size,
                    inputs: vec![],
                });
                graph.add_input(weight);
                inputs.push(weight);
            }

            graph.add_node(NodeInfo {
                id: format!("node-{}{}", i_depth, (b'a' + i_branch as u8) as char),
                size_bits: graph_params.node_size,
                inputs,
            })
        }).collect_vec();
        prev = next;
    }
    let node_output = graph.add_node(NodeInfo {
        id: "node-output".to_string(),
        size_bits: graph_params.node_size,
        inputs: prev,
    });
    graph.add_output(node_output);

    // allocations
    let mut allocations = vec![];
    for (i, &core_group) in enumerate(&core_groups) {
        for node in graph.nodes() {
            // inputs don't get allocations, they already exist
            if graph.inputs.contains(&node) {
                continue;
            }
            
            for &(ref name, time, energy) in allocs_time_energy {
                allocations.push(AllocationInfo {
                    id: name.to_string(),
                    group: core_group,
                    node,
                    input_memories: vec![mem_core[i]; graph.node_info[node.0].inputs.len()],
                    output_memory: mem_core[i],
                    time,
                    energy,
                });
            }
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