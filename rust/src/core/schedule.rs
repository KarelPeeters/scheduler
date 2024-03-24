use crate::core::problem::{Allocation, Channel, Hardware, Node};

#[derive(Debug)]
pub struct Schedule {
    actions: Vec<Action>,
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Wait(ActionWait),
    Core(ActionCore),
    Channel(ActionChannel),
}

#[derive(Debug, Clone, Copy)]
pub struct ActionWait {
    pub time_start: f64,
    pub time_end: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ActionCore {
    pub time_start: f64,
    pub time_end: f64,
    pub alloc: Allocation,
}

#[derive(Debug, Clone, Copy)]
pub struct ActionChannel {
    pub time_start: f64,
    pub time_end: f64,
    pub channel: Channel,
    pub dir_a_to_b: bool,
    pub value: Node,
}

impl Schedule {
    fn to_svg(&self, hardware: Hardware) -> String {
        // TODO include all metadata, including memory contents, usage, lock count, ...
        //    for easy visual debugging
        todo!()
    }
}

