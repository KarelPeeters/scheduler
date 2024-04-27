use crate::core::problem::ChannelCost;
use crate::core::wrapper::{Energy, Time};

pub mod params;

pub const DEFAULT_CHANNEL_COST_EXT: ChannelCost = ChannelCost {
    latency: Time(0),
    time_per_bit: Time(2),
    energy_per_bit: Energy(4),
};
pub const DEFAULT_CHANNEL_COST_INT: ChannelCost = ChannelCost {
    latency: Time(0),
    time_per_bit: Time(1),
    energy_per_bit: Energy(2),
};

#[cfg(test)]
pub mod tests;
