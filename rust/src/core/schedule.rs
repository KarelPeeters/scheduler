use crate::core::problem::{Allocation, Channel, ChannelDir, Hardware, Node};

#[derive(Debug)]
pub struct Schedule {
    actions: Vec<Action>,
}

#[derive(Debug)]
pub struct Action {
    time_start: f64,
    time_end: f64,
    energy: f64,
    kind: ActionKind,
}

#[derive(Debug)]
pub enum ActionKind {
    Wait,
    StartCore { alloc: Allocation },
    StartChannel { channel: Channel, dir: ChannelDir, value: Node },
}

impl Schedule {
    fn to_svg(&self, hardware: Hardware) -> String {
        todo!()
    }
}