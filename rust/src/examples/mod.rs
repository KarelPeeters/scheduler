use crate::core::problem::ChannelCost;

pub mod params;

pub const DEFAULT_CHANNEL_COST_EXT: ChannelCost = ChannelCost { latency: 0.0, time_per_bit: 1.0, energy_per_bit: 2.0 };
pub const DEFAULT_CHANNEL_COST_INT: ChannelCost = ChannelCost { latency: 0.0, time_per_bit: 0.5, energy_per_bit: 1.0 };

#[cfg(test)]
pub mod tests;
