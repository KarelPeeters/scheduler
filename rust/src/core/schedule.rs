use std::fmt::{Debug, Formatter};
use crate::core::problem::{Allocation, Channel, Memory, Node, Problem};
use crate::core::wrapper::{Energy, Time};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Timed<T> {
    pub time: TimeRange,
    pub inner: T,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Action {
    Wait(Time),
    Core(Allocation),
    Channel(ActionChannel),
    Drop(ActionDrop)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ActionChannel {
    pub channel: Channel,
    pub value: Node,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ActionDrop {
    pub value: Node,
    pub mem: Memory,
}

// TODO move next to Time
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct TimeRange {
    pub start: Time,
    /// exclusive
    pub end: Time,
}

impl Debug for TimeRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TimeRange({:?}..{:?})", self.start, self.end)
    }
}

impl TimeRange {
    pub fn instant(time: Time) -> Self {
        TimeRange {
            start: time,
            end: time,
        }
    }
    
    pub fn overlaps(self, other: TimeRange) -> bool {
        assert!(self.start <= self.end);
        assert!(other.start <= other.end);
        self.start < other.end && other.start < self.end
    }

    pub fn len(self) -> Time {
        self.end - self.start
    }
}

impl Action {
    pub fn time(self, problem: &Problem) -> Time {
        match self {
            Action::Wait(time) => time,
            Action::Core(alloc) => problem.allocations[alloc].time,
            Action::Channel(action) => action.time(problem),
            Action::Drop(_) => Time(0),
        }
    }

    pub fn energy(self, problem: &Problem) -> Energy {
        match self {
            Action::Wait(_) | Action::Drop(_) => Energy(0),
            Action::Core(alloc) => problem.allocations[alloc].energy,
            Action::Channel(action) => action.energy(problem),
        }
    }
}

impl ActionChannel {
    pub fn time(self, problem: &Problem) -> Time {
        let ActionChannel { channel, value } = self;
        problem.hardware.channels[channel].cost.time_to_transfer(problem.graph.nodes[value].size_bits)
    }

    pub fn energy(self, problem: &Problem) -> Energy {
        let ActionChannel { channel, value } = self;
        problem.hardware.channels[channel].cost.energy_to_transfer(problem.graph.nodes[value].size_bits)
    }
}