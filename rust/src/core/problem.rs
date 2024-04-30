use std::collections::HashSet;

use itertools::Itertools;

use crate::core::wrapper::{Energy, Time, TypedVec};
use crate::define_typed_index;
use crate::util::graphviz::GraphViz;

// problem
#[derive(Debug)]
pub struct Problem {
    pub id: String,
    pub hardware: Hardware,
    pub graph: Graph,

    pub allocations: TypedVec<Allocation, AllocationInfo>,

    pub input_placements: Vec<Memory>,
    pub output_placements: Vec<Memory>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CostTarget {
    // Minimize both time and energy, getting the full pareto front
    Full,
    // Primarily minimize time, still optimizing energy as the secondary objective.
    // This will return at most one solution.
    Time,
    // Primarily minimize energy, still optimizing time as the secondary objective.
    // This will return at most one solution.
    Energy,
}

define_typed_index!(Allocation);

// TODO rename this to something better, maybe mapping?
#[derive(Debug)]
pub struct AllocationInfo {
    pub id: String,
    pub group: Group,
    pub node: Node,

    pub input_memories: Vec<Memory>,
    pub output_memory: Memory,

    pub time: Time,
    pub energy: Energy,
}

// graph
#[derive(Debug)]
pub struct Graph {
    pub id: String,
    pub nodes: TypedVec<Node, NodeInfo>,

    pub inputs: Vec<Node>,
    pub outputs: Vec<Node>,
}

define_typed_index!(Node);

#[derive(Debug)]
pub struct NodeInfo {
    pub id: String,
    pub size_bits: u64,
    pub inputs: Vec<Node>,

    // Only start this node after the given set of nodes has been started.
    // This is useful to manually remove symmetries from the optimization problem,
    // making it easier (faster!) to solve.
    pub start_after: Vec<Node>,
}

// hardware
#[derive(Debug)]
pub struct Hardware {
    pub id: String,
    pub memories: TypedVec<Memory, MemoryInfo>,
    pub groups: TypedVec<Group, GroupInfo>,
    pub channels: TypedVec<Channel, ChannelInfo>,
}

// TODO rename group?
define_typed_index!(Group);

define_typed_index!(Channel);

define_typed_index!(Memory);

#[derive(Debug)]
pub struct GroupInfo {
    pub id: String,
}

#[derive(Debug)]
pub struct ChannelInfo {
    pub id: String,
    pub group: Group,
    pub mem_source: Memory,
    pub mem_dest: Memory,
    pub cost: ChannelCost,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ChannelCost {
    // TODO should we use dedicated allocations per value instead?
    //   this is just halfway limited again
    pub latency: Time,
    pub time_per_bit: Time,
    pub energy_per_bit: Energy,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ChannelDir {
    Single,
    Both,
}

#[derive(Debug)]
pub struct MemoryInfo {
    pub id: String,
    pub size_bits: Option<u64>,
}

// implementations
impl Problem {
    pub fn assert_valid(&self) {
        self.hardware.assert_valid();
        self.graph.assert_valid();

        for (_, alloc_info) in &self.allocations {
            assert!(self.hardware.groups.has_key(alloc_info.group));
            assert!(self.graph.nodes.has_key(alloc_info.node));
            for &mem in &alloc_info.input_memories {
                assert!(self.hardware.memories.has_key(mem));
            }
            assert!(self.hardware.memories.has_key(alloc_info.output_memory));

            assert_eq!(alloc_info.input_memories.len(), self.graph.nodes[alloc_info.node].inputs.len());
        }

        assert_eq!(self.input_placements.len(), self.graph.inputs.len());
        for &mem in &self.input_placements {
            assert!(self.hardware.memories.has_key(mem));
        }

        assert_eq!(self.output_placements.len(), self.graph.outputs.len());
        for &mem in &self.output_placements {
            assert!(self.hardware.memories.has_key(mem));
        }
    }

    pub fn core_connected_memories(&self) -> Vec<(HashSet<Memory>, HashSet<Memory>)> {
        // TODO this has lost even more meaning
        vec![]
        
        // let mut result = vec![(HashSet::new(), HashSet::new()); self.hardware.cores().len()];
        // 
        // for alloc in &self.allocation_info {
        //     let (inputs, outputs) = &mut result[alloc.core.0];
        //     inputs.extend(alloc.input_memories.iter().copied());
        //     outputs.insert(alloc.output_memory);
        // }
        // 
        // result
    }
}

impl Graph {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), nodes: TypedVec::new(), inputs: vec![], outputs: vec![] }
    }

    pub fn add_node(&mut self, info: NodeInfo) -> Node {
        self.nodes.push(info)
    }

    pub fn create_node(
        &mut self,
        id: impl Into<String>,
        size_bits: u64,
        inputs: Vec<Node>,
        start_after: Vec<Node>,
    ) -> Node {
        let info = NodeInfo {
            id: id.into(),
            size_bits,
            inputs,
            start_after,
        };
        self.add_node(info)
    }

    pub fn add_input(&mut self, node: Node) {
        assert!(self.nodes.has_key(node));
        assert!(self.nodes[node].inputs.is_empty());
        assert!(!self.inputs.contains(&node));
        self.inputs.push(node);
    }

    pub fn add_output(&mut self, node: Node) {
        assert!(self.nodes.has_key(node));
        self.outputs.push(node);
    }

    pub fn to_graphviz(&self) -> GraphViz {
        let mut g = GraphViz::new();

        g.push(format!("label=<<B>{}</B>>", self.id));
        g.push("labelloc=t");

        for (node, node_info) in &self.nodes {
            let mut rows = vec![
                ("debug", format!("{:?}", node)),
                ("id", node_info.id.clone()),
                ("size_bits", node_info.size_bits.to_string()),
            ];

            let input = self.inputs.iter().position(|&x| x == node);
            let output = self.outputs.iter().position(|&x| x == node);
            if let Some(input) = input {
                rows.push(("input", input.to_string()));
            }
            if let Some(output) = output {
                rows.push(("output", output.to_string()));
            }

            let label = GraphViz::table("Node", rows);
            g.push(format!("node_{} [label=<{}>, shape=box, color=green]", node.0, label));

            let tail = format!("node_{}", node.0);

            for (j, input) in node_info.inputs.iter().enumerate() {
                let head = format!("node_{}", input.0);
                g.push(format!("{} -> {} [headlabel={}]", head, tail, j));
            }

            for &after in &node_info.start_after {
                let head = format!("node_{}", after.0);
                g.push(format!("{} -> {} [style=\"dotted\"]", head, tail));
            }
        }

        g
    }

    pub fn assert_valid(&self) {
        // TODO assert that graph is a DAG
        //   even through that should be true by construction, users can still mess around in the public fields

        for (_, info) in &self.nodes {
            for &x in &info.inputs {
                assert!(self.nodes.has_key(x));
            }
            for &x in &info.start_after {
                assert!(self.nodes.has_key(x));
            }
        }

        assert_eq!(self.inputs.iter().unique().count(), self.inputs.len());
        for &x in &self.inputs {
            assert!(self.nodes.has_key(x));
            assert!(self.nodes[x].inputs.is_empty());
        }
        assert_eq!(self.inputs.iter().unique().count(), self.inputs.len());
        for &x in &self.outputs {
            assert!(self.nodes.has_key(x));
        }
    }
}

impl Hardware {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), memories: TypedVec::new(), groups: TypedVec::new(), channels: TypedVec::new() }
    }

    pub fn add_memory(&mut self, info: MemoryInfo) -> Memory {
        self.memories.push(info)
    }

    pub fn create_memory(&mut self, id: impl Into<String>, size_bits: Option<u64>) -> Memory {
        self.add_memory(MemoryInfo { id: id.into(), size_bits })
    }

    pub fn add_group(&mut self, info: GroupInfo) -> Group {
        self.groups.push(info)
    }

    pub fn create_group(&mut self, id: impl Into<String>) -> Group {
        self.add_group(GroupInfo { id: id.into() })
    }
    
    pub fn add_channel(&mut self, info: ChannelInfo) -> Channel {
        let channel = self.channels.push(info);
        self.assert_channel_valid(channel);
        channel
    }

    pub fn create_channel(&mut self, id: impl Into<String>, group: Group, mem_source: Memory, mem_dest: Memory, cost: ChannelCost) {
        let info = ChannelInfo {
            id: id.into(),
            group,
            mem_source,
            mem_dest,
            cost,
        };
        self.add_channel(info);
    }

    pub fn create_channel_bidir(&mut self, id: impl Into<String>, group: Group, mem_left: Memory, mem_right: Memory, cost: ChannelCost) {
        let id = id.into();
        for (dir, mem_source, mem_dest) in [("fwd", mem_left, mem_right), ("bck", mem_right, mem_left)].iter().copied() {
            let info = ChannelInfo {
                id: format!("{}_{}", id, dir),
                group,
                mem_source,
                mem_dest,
                cost,
            };
            self.add_channel(info);
        }
    }

    // TODO render groups as a box around channels + a core if there are allocs in it?
    pub fn to_graphviz(&self, _: Vec<(HashSet<Memory>, HashSet<Memory>)>) -> GraphViz {
        let mut g = GraphViz::new();

        g.push(format!("label=<<B>{}</B>>", self.id));
        g.push("labelloc=t");

        for (mem, mem_info) in &self.memories {
            let size = if let Some(size) = mem_info.size_bits {
                size.to_string()
            } else {
                "inf".to_owned()
            };
            let rows = vec![
                ("debug", format!("{:?}", mem)),
                ("id", mem_info.id.clone()),
                ("size_bits", size),
            ];
            let label = GraphViz::table("Memory", rows);
            g.push(format!("mem_{} [label=<{}>, shape=box, color=blue]", mem.0, label));
        }

        for (channel, channel_info) in &self.channels {
            let rows = vec![
                ("debug", format!("{:?}", channel)),
                ("id", channel_info.id.clone()),
                ("group", self.groups[channel_info.group].id.clone()),
                ("latency", channel_info.cost.latency.0.to_string()),
                ("time_per_bit", channel_info.cost.time_per_bit.0.to_string()),
                ("energy_per_bit", channel_info.cost.energy_per_bit.0.to_string()),
            ];
            let mid = format!("channel_{}", channel.0);

            let label = GraphViz::table("Channel", rows);
            g.push(format!("{} [label=<{}>, shape=box, color=darkorange]", mid, label));

            // TODO fuse pairs of channels that are identical and form a group?
            let head = format!("mem_{}", channel_info.mem_source.0);
            let tail = format!("mem_{}", channel_info.mem_dest.0);

            let dir = "forward";
            g.push(format!("{} -> {} [dir={}, color=darkorange]", head, mid, dir));
            g.push(format!("{} -> {} [dir={}, color=darkorange]", mid, tail, dir));
        }
        
        // TODO render allocs and groups as cores?

        // for  core in self.cores() {
        //     let rows = vec![
        //         ("id", self.core_info[core.0].id.clone()),
        //     ];
        //     let label = GraphViz::table("Core", rows);
        //     g.push(format!("core_{} [label=<{}>, color=green]", core.0, label));
        // 
        //     for (j, mem) in self.memories().enumerate() {
        //         let (inputs, outputs) = &core_connected_memories[core.0];
        // 
        //         let dir = match (inputs.contains(&mem), outputs.contains(&mem)) {
        //             (true, true) => "both",
        //             (true, false) => "input",
        //             (false, true) => "output",
        //             (false, false) => continue,
        //         };
        // 
        //         let head = format!("mem_{}", j);
        //         let tail = format!("core_{}", core.0);
        //         g.push(format!("{} -> {} [dir={}, color=green]", head, tail, dir));
        //     }
        // }

        g
    }

    fn assert_channel_valid(&self, channel: Channel) {
        assert!(self.channels.has_key(channel));
        let channel_info = &self.channels[channel];
        assert!(self.memories.has_key(channel_info.mem_source));
        assert!(self.memories.has_key(channel_info.mem_dest));
    }

    pub fn assert_valid(&self) {
        for channel in self.channels.keys() {
            self.assert_channel_valid(channel);
        }
    }
}

impl ChannelCost {
    pub fn energy_to_transfer(&self, size_bits: u64) -> Energy {
        self.energy_per_bit * size_bits
    }

    pub fn time_to_transfer(&self, size_bits: u64) -> Time {
        self.latency + self.time_per_bit * size_bits
    }
}
