use crate::core::problem::{Allocation, Channel, Memory, Node};

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Wait(ActionWait),
    Core(ActionCore),
    Channel(ActionChannel),
    Drop(ActionDrop)
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
pub struct ActionDrop {
    pub time: f64,
    pub value: Node,
    pub mem: Memory,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}


impl TimeRange {
    pub fn overlaps(self, other: TimeRange) -> bool {
        assert!(!self.start.is_nan() && !self.end.is_nan() && !other.start.is_nan() && !other.end.is_nan());
        self.start < other.end && other.start < self.end
    }
}