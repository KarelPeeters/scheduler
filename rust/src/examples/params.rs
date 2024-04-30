use std::vec;

use itertools::enumerate;

use crate::core::problem::{AllocationInfo, ChannelCost, Graph, GroupInfo, Hardware, MemoryInfo, Node, NodeInfo, Problem};
use crate::core::wrapper::{Energy, Time, TypedVec};

#[derive(Debug, Clone)]
pub struct TestGraphParams {
    pub depth: usize,
    pub branches: usize,
    pub cross: CrossBranches,
    pub node_size: u64,
    pub weight_size: Option<u64>,
    pub share_weights: bool,
    pub constrain_order: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum CrossBranches {
    Never,
    EveryNth(usize),
    ConvWise,
}

#[derive(Debug, Clone)]
pub struct TestHardwareParams {
    pub core_count: usize,
    // TODO option to share both channels with the core?
    pub share_group: bool,
    pub mem_size_ext: Option<u64>,
    pub mem_size_int: Option<u64>,
    pub channel_cost_ext: ChannelCost,
    pub channel_cost_int: ChannelCost,
}

pub fn test_problem(graph_params: TestGraphParams, hardware_params: TestHardwareParams, allocs_time_energy: &[(&str, i64, i64)]) -> Problem {
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
        let (channel_id, mem_prev, channel_cost) = if i_core == 0 {
            ("channel_ext_0".to_string(), mem_ext, hardware_params.channel_cost_ext)
        } else {
            (format!("channel_chip_{}", i_core), mem_core[i_core - 1], hardware_params.channel_cost_int)
        };
        let channel_group = if hardware_params.share_group {
            core_group
        } else {
            hardware.add_group(GroupInfo { id: channel_id.clone() })
        };
        hardware.create_channel(format!("{channel_id}_fwd"), channel_group, mem_prev, mem_curr, channel_cost);
        hardware.create_channel(format!("{channel_id}_bck"), channel_group, mem_curr, mem_prev, channel_cost);
    }

    // graph
    assert!(graph_params.branches > 0 || graph_params.depth == 0);
    
    let mut graph = Graph::new("graph");
    let node_input = graph.add_node(NodeInfo {
        id: "node-input".to_string(),
        size_bits: graph_params.node_size,
        inputs: vec![],
        start_after: vec![],
    });
    graph.add_input(node_input);
    let mut prev = vec![node_input];
    for i_depth in 0..graph_params.depth {
        let mut shared_weight = None;
        
        let mut next = vec![];
        
        for i_branch in 0..graph_params.branches {
            let mut inputs = if i_depth == 0 {
                prev.clone()
            } else {
                graph_params.cross.pick_inputs(i_depth, i_branch, &prev)
            };

            if let Some(graph_weight_size) = graph_params.weight_size {
                let weight = match shared_weight {
                    None => {
                        // TODO add "constant" utility constructor
                        let weight = graph.add_node(NodeInfo {
                            id: format!("weight-{}{}", i_depth, (b'a' + i_branch as u8) as char),
                            size_bits: graph_weight_size,
                            inputs: vec![],
                            start_after: vec![],
                        });
                        graph.add_input(weight);
                        weight
                    }
                    Some(shared_weight) => shared_weight,
                };

                if graph_params.share_weights {
                    shared_weight = Some(weight);
                }
                
                inputs.push(weight);
            }

            let start_after = if graph_params.constrain_order { next.clone() } else { vec![] };

            let node = graph.add_node(NodeInfo {
                id: format!("node-{}{}", i_depth, (b'a' + i_branch as u8) as char),
                size_bits: graph_params.node_size,
                inputs,
                start_after,
            });
            next.push(node);
        }

        prev = next;
    }
    let node_output = graph.add_node(NodeInfo {
        id: "node-output".to_string(),
        size_bits: graph_params.node_size,
        inputs: prev,
        start_after: vec![],
    });
    graph.add_output(node_output);

    // allocations
    let mut allocations = TypedVec::new();
    for (i, &core_group) in enumerate(&core_groups) {
        for (node, node_info) in &graph.nodes {
            // inputs don't get allocations, they already exist
            if graph.inputs.contains(&node) {
                continue;
            }
            
            for &(ref name, time, energy) in allocs_time_energy {
                allocations.push(AllocationInfo {
                    id: name.to_string(),
                    group: core_group,
                    node,
                    input_memories: vec![mem_core[i]; node_info.inputs.len()],
                    output_memory: mem_core[i],
                    time: Time(time),
                    energy: Energy(energy),
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
        allocations,
        input_placements,
        output_placements,
    }
}

impl CrossBranches {
    fn pick_inputs(self, i_depth: usize, i_branch: usize, prev: &[Node]) -> Vec<Node> {
        match self {
            CrossBranches::Never => vec![prev[i_branch]],
            CrossBranches::EveryNth(n) => {
                if i_depth % n == 0 { prev.to_vec() } else { vec![prev[i_branch]] }
            }
            CrossBranches::ConvWise => {
                let mut result = vec![];
                if 0 < i_branch {
                    result.push(prev[i_branch - 1]);
                }
                result.push(prev[i_branch]);
                if i_branch + 1 < prev.len() {
                    result.push(prev[i_branch + 1]);
                }
                result
            }
        }
    }
}
