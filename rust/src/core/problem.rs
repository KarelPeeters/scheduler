use std::collections::HashSet;
use itertools::{chain, Itertools};

use crate::util::graphviz::GraphViz;

// problem
#[derive(Debug)]
pub struct Problem {
    pub id: String,
    pub hardware: Hardware,
    pub graph: Graph,

    pub allocations: Vec<AllocationInfo>,

    pub input_placements: Vec<Memory>,
    pub output_placements: Vec<Memory>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Allocation(usize);

#[derive(Debug)]
pub struct AllocationInfo {
    pub id: String,
    pub core: Core,
    pub node: Node,

    pub input_memories: Vec<Memory>,
    pub output_memory: Memory,

    pub time: f64,
    pub energy: f64,
}

// graph
#[derive(Debug)]
pub struct Graph {
    pub id: String,
    pub node_info: Vec<NodeInfo>,

    pub inputs: Vec<Node>,
    pub outputs: Vec<Node>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Node(usize);

#[derive(Debug)]
pub struct NodeInfo {
    pub id: String,
    pub size_bits: u64,
    pub inputs: Vec<Node>,
}

// hardware
#[derive(Debug)]
pub struct Hardware {
    pub id: String,
    pub mem_info: Vec<MemoryInfo>,
    pub channel_info: Vec<ChannelInfo>,
    pub core_info: Vec<CoreInfo>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Core(usize);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Channel(usize);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Memory(usize);

#[derive(Debug)]
pub struct CoreInfo {
    pub id: String,
}

#[derive(Debug)]
pub struct ChannelInfo {
    pub id: String,

    pub mem_a: Memory,
    pub mem_b: Memory,
    pub dir: Direction,

    pub latency: f64,
    pub time_per_bit: f64,
    pub energy_per_bit: f64,
}

// TODO replace direction with only uni-directional channels and constraint groups
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    AtoB,
    BtoA,
    Both,
}

#[derive(Debug)]
pub enum ChannelDir {
    AtoB,
    BtoA,
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

        for alloc in &self.allocations {
            assert!(alloc.core.0 < self.hardware.core_info.len());
            assert!(alloc.node.0 < self.graph.node_info.len());
            for mem in &alloc.input_memories {
                assert!(mem.0 < self.hardware.mem_info.len());
            }
            assert!(alloc.output_memory.0 < self.hardware.mem_info.len());
        }

        assert_eq!(self.input_placements.len(), self.graph.inputs.len());
        for mem in &self.input_placements {
            assert!(mem.0 < self.hardware.mem_info.len());
        }

        assert_eq!(self.output_placements.len(), self.graph.outputs.len());
        for mem in &self.output_placements {
            assert!(mem.0 < self.hardware.mem_info.len());
        }
    }

    pub fn core_connected_memories(&self) -> Vec<(HashSet<Memory>, HashSet<Memory>)> {
        let mut result = vec![(HashSet::new(), HashSet::new()); self.hardware.core_info.len()];

        for alloc in &self.allocations {
            let (inputs, outputs) = &mut result[alloc.core.0];
            inputs.extend(alloc.input_memories.iter().copied());
            outputs.insert(alloc.output_memory);
        }

        result
    }
}

impl Graph {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), node_info: vec![], inputs: vec![], outputs: vec![] }
    }

    pub fn nodes(&self) -> impl Iterator<Item=Node> {
        (0..self.node_info.len()).map(Node)
    }

    pub fn add_node(&mut self, info: NodeInfo) -> Node {
        let node = Node(self.node_info.len());
        self.node_info.push(info);
        node
    }

    pub fn add_input(&mut self, node: Node) {
        assert!(node.0 < self.node_info.len());
        assert!(!self.inputs.contains(&node));
        self.inputs.push(node);
    }

    pub fn add_output(&mut self, node: Node) {
        assert!(node.0 < self.node_info.len());
        self.outputs.push(node);
    }

    pub fn to_graphviz(&self) -> GraphViz {
        let mut g = GraphViz::new();

        g.push(format!("label=<<B>{}</B>>", self.id));
        g.push("labelloc=t");

        for (i, node) in self.nodes().enumerate() {
            let mut rows = vec![
                ("id", self.node_info[node.0].id.clone()),
                ("size_bits", self.node_info[node.0].size_bits.to_string()),
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
            g.push(format!("node_{} [label=<{}>, shape=box, color=green]", i, label));

            for (j, input) in self.node_info[node.0].inputs.iter().enumerate() {
                let head = format!("node_{}", self.nodes().position(|x| x == *input).unwrap());
                let tail = format!("node_{}", i);
                g.push(format!("{} -> {} [headlabel={}]", head, tail, j));
            }
        }

        g
    }

    pub fn assert_valid(&self) {
        // TODO assert that graph is a DAG
        for info in &self.node_info {
            for x in &info.inputs {
                assert!(x.0 < self.node_info.len());
            }
        }

        assert_eq!(self.inputs.iter().unique().count(), self.inputs.len());
        for x in &self.inputs {
            assert!(x.0 < self.node_info.len());
        }
        assert_eq!(self.inputs.iter().unique().count(), self.inputs.len());
        for x in &self.outputs {
            assert!(x.0 < self.node_info.len());
        }
    }
}

impl Hardware {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into(), mem_info: vec![], core_info: vec![], channel_info: vec![] }
    }

    pub fn memories(&self) -> impl Iterator<Item=Memory> {
        (0..self.mem_info.len()).map(Memory)
    }

    pub fn channels(&self) -> impl Iterator<Item=Channel> {
        (0..self.channel_info.len()).map(Channel)
    }

    pub fn cores(&self) -> impl Iterator<Item=Core> {
        (0..self.core_info.len()).map(Core)
    }

    pub fn add_memory(&mut self, info: MemoryInfo) -> Memory {
        let memory = Memory(self.mem_info.len());
        self.mem_info.push(info);
        memory
    }

    pub fn add_core(&mut self, info: CoreInfo) -> Core {
        let core = Core(self.core_info.len());
        self.core_info.push(info);
        core
    }

    pub fn add_channel(&mut self, info: ChannelInfo) -> Channel {
        let channel = Channel(self.channel_info.len());
        self.channel_info.push(info);
        self.assert_channel_valid(channel);
        channel
    }

    pub fn to_graphviz(&self, core_connected_memories: Vec<(HashSet<Memory>, HashSet<Memory>)>) -> GraphViz {
        let mut g = GraphViz::new();

        g.push(format!("label=<<B>{}</B>>", self.id));
        g.push("labelloc=t");

        for mem in self.memories() {
            let size = if let Some(size) = self.mem_info[mem.0].size_bits {
                size.to_string()
            } else {
                "inf".to_owned()
            };
            let rows = vec![
                ("id", self.mem_info[mem.0].id.clone()),
                ("size_bits", size),
            ];
            let label = GraphViz::table("Memory", rows);
            g.push(format!("mem_{} [label=<{}>, shape=box, color=blue]", mem.0, label));
        }

        for channel in self.channels() {
            let info = &self.channel_info[channel.0];
            let rows = vec![
                ("id", info.id.clone()),
                ("latency", info.latency.to_string()),
                ("time_per_bit", info.time_per_bit.to_string()),
                ("energy_per_bit", info.energy_per_bit.to_string()),
            ];
            let mid = format!("channel_{}", channel.0);

            let label = GraphViz::table("Channel", rows);
            g.push(format!("{} [label=<{}>, shape=box, color=darkorange]", mid, label));

            let head = format!("mem_{}", info.mem_a.0);
            let tail = format!("mem_{}", info.mem_b.0);

            let dir = match info.dir {
                Direction::AtoB => "forward",
                Direction::BtoA => "backward",
                Direction::Both => "both",
            };

            g.push(format!("{} -> {} [dir={}, color=darkorange]", head, mid, dir));
            g.push(format!("{} -> {} [dir={}, color=darkorange]", mid, tail, dir));
        }

        for  core in self.cores() {
            let rows = vec![
                ("id", self.core_info[core.0].id.clone()),
            ];
            let label = GraphViz::table("Core", rows);
            g.push(format!("core_{} [label=<{}>, color=green]", core.0, label));

            for (j, mem) in self.memories().enumerate() {
                let (inputs, outputs) = &core_connected_memories[core.0];

                let dir = match (inputs.contains(&mem), outputs.contains(&mem)) {
                    (true, true) => "both",
                    (true, false) => "input",
                    (false, true) => "output",
                    (false, false) => continue,
                };

                let head = format!("mem_{}", j);
                let tail = format!("core_{}", core.0);
                g.push(format!("{} -> {} [dir={}, color=green]", head, tail, dir));
            }
        }

        g
    }

    fn assert_channel_valid(&self, channel: Channel) {
        assert!(channel.0 < self.channel_info.len());
        let channel = &self.channel_info[channel.0];
        assert!(channel.mem_a.0 < self.mem_info.len());
        assert!(channel.mem_b.0 < self.mem_info.len());
        assert!(channel.latency >= 0.0 && channel.time_per_bit >= 0.0 && channel.energy_per_bit >= 0.0);
    }

    pub fn assert_valid(&self) {
        for i in 0..self.channel_info.len() {
            self.assert_channel_valid(Channel(i));
        }
    }
}
