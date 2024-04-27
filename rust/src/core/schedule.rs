use crate::core::problem::{Allocation, Channel, Memory, Node};
use crate::core::wrapper::Time;

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
    pub time: Time,
    pub value: Node,
    pub mem: Memory,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeRange {
    pub start: Time,
    /// exclusive
    pub end: Time,
}

impl TimeRange {
    pub fn overlaps(self, other: TimeRange) -> bool {
        assert!(self.start <= self.end);
        assert!(other.start <= other.end);
        self.start < other.end && other.start < self.end
    }
}
