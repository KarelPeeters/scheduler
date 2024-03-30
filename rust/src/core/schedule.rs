use crate::core::problem::{Allocation, Channel, Node};

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Wait(ActionWait),
    Core(ActionCore),
    Channel(ActionChannel),
}

#[derive(Debug, Clone, Copy)]
pub struct ActionWait {
    pub time: TimeRange,
}

#[derive(Debug, Clone, Copy)]
pub struct ActionCore {
    pub time: TimeRange,
    pub alloc: Allocation,
}

#[derive(Debug, Clone, Copy)]
pub struct ActionChannel {
    pub time: TimeRange,
    pub channel: Channel,
    pub value: Node,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}
