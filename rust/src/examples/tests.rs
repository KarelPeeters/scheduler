use itertools::Itertools;

use crate::core::problem::{ChannelCost, CostTarget, Graph, Hardware, Problem};
use crate::core::solve::{DummyReporter, SolveMethod};
use crate::core::state::Cost;
use crate::core::wrapper::{Energy, Time};
use crate::examples::{DEFAULT_CHANNEL_COST_EXT, DEFAULT_CHANNEL_COST_INT};
use crate::examples::params::{CrossBranches, test_problem, TestGraphParams, TestHardwareParams};

#[test]
fn empty() {
    let problem = Problem {
        id: "empty".to_string(),
        hardware: Hardware::new("hardware"),
        graph: Graph::new("graph"),
        allocation_info: vec![],
        input_placements: vec![],
        output_placements: vec![],
    };
    expect_solution(&problem, vec![Cost { time: Time(0), energy: Energy(0) }]);
}

#[test]
fn single_normal() {
    let problem = test_problem(
        TestGraphParams {
            depth: 0,
            branches: 0,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![Cost { time: Time(6000), energy: Energy(5000) }]);
}

#[test]
fn single_zero_sized_node() {
    let problem = test_problem(
        TestGraphParams {
            depth: 0,
            branches: 0,
            cross: CrossBranches::Never,
            node_size: 0,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![Cost { time: Time(4000), energy: Energy(1000) }]);
}

#[test]
fn linear_use_single() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 1,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![Cost { time: Time(22_000), energy: Energy(9000) }]);
}

#[test]
fn split_use_both() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![
        Cost { time: Time(23_000), energy: Energy(15000) },
        Cost { time: Time(38_000), energy: Energy(13_000) },
    ]);
}

#[test]
fn split_use_single() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: ChannelCost {
                latency: Time(0),
                time_per_bit: Time(20),
                energy_per_bit: Energy(1),
            },
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![Cost { time: Time(38_000), energy: Energy(13_000) }]);
}

#[test]
fn single_tradeoff() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 1,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("slow", 8000, 750), ("mid", 4000, 1000), ("fast", 2000, 1500)],
    );
    expect_solution(&problem, vec![
        Cost { time: Time(12000), energy: Energy(11500) },
        Cost { time: Time(14000), energy: Energy(11000) },
        Cost { time: Time(16000), energy: Energy(10500) },
        Cost { time: Time(18000), energy: Energy(10000) },
        Cost { time: Time(20000), energy: Energy(9500) },
        Cost { time: Time(22000), energy: Energy(9000) },
        Cost { time: Time(26000), energy: Energy(8750) },
        Cost { time: Time(30000), energy: Energy(8500) },
        Cost { time: Time(34000), energy: Energy(8250) },
        Cost { time: Time(38000), energy: Energy(8000) },
        Cost { time: Time(42000), energy: Energy(7750) },
    ]);
}

#[test]
fn split_tradeoff_deep() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("mid", 4000, 1000), ("fast", 2000, 1500)],
    );
    expect_solution(&problem, vec![
        Cost { time: Time(13000), energy: Energy(19500) },
        Cost { time: Time(14000), energy: Energy(19000) },
        Cost { time: Time(15000), energy: Energy(18500) },
        Cost { time: Time(16000), energy: Energy(18000) },
        Cost { time: Time(17000), energy: Energy(17500) },
        Cost { time: Time(18000), energy: Energy(17000) },
        Cost { time: Time(19000), energy: Energy(16500) },
        Cost { time: Time(20000), energy: Energy(16000) },
        Cost { time: Time(21000), energy: Energy(15500) },
        Cost { time: Time(23000), energy: Energy(15000) },
        Cost { time: Time(32000), energy: Energy(14500) },
        Cost { time: Time(34000), energy: Energy(14000) },
        Cost { time: Time(36000), energy: Energy(13500) },
        Cost { time: Time(38000), energy: Energy(13000) },
    ]);
}

#[test]
fn split_tradeoff_shallow() {
    let problem = test_problem(
        TestGraphParams {
            depth: 2,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("slow", 6000, 500), ("mid", 4000, 1000), ("fast", 2000, 1500)],
    );
    expect_solution(&problem, vec![
        Cost { time: Time(9000), energy: Energy(13500) },
        Cost { time: Time(10000), energy: Energy(13000) },
        Cost { time: Time(11000), energy: Energy(12500) },
        Cost { time: Time(12000), energy: Energy(11500) },
        Cost { time: Time(14000), energy: Energy(11000) },
        Cost { time: Time(15000), energy: Energy(10500) },
        Cost { time: Time(16000), energy: Energy(10000) },
        Cost { time: Time(17000), energy: Energy(9500) },
        Cost { time: Time(19000), energy: Energy(9000) },
        Cost { time: Time(21000), energy: Energy(8500) },
        Cost { time: Time(26000), energy: Energy(8000) },
        Cost { time: Time(28000), energy: Energy(7500) },
        Cost { time: Time(30000), energy: Energy(7000) },
        Cost { time: Time(32000), energy: Energy(6500) },
    ]);
}

#[test]
fn single_memory_drop() {
    let problem = test_problem(
        TestGraphParams {
            depth: 2,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: Some(500),
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: Some(1500),
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![Cost { time: Time(29_000), energy: Energy(23_000) }]);
}

#[test]
fn tricky_drop_case() {
    let problem = test_problem(
        TestGraphParams {
            depth: 1,
            branches: 2,
            cross: CrossBranches::Never,
            node_size: 500,
            weight_size: Some(500),
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: Some(1500),
            channel_cost_ext: DEFAULT_CHANNEL_COST_EXT,
            channel_cost_int: DEFAULT_CHANNEL_COST_INT,
        },
        &[("basic", 4000, 1000)],
    );
    expect_solution(&problem, vec![
        Cost { time: Time(12500), energy: Energy(14000) },
        Cost { time: Time(16500), energy: Energy(13000) },
    ]);
}

// TODO add test cases with empty output (ie. unsolvable)
// TODO shuffle graph and hardware indices to check that everything is order-independent
#[track_caller]
pub fn expect_solution(problem: &Problem, mut expected: Vec<Cost>) {
    let key = |c: &Cost| (c.time, c.energy);
    expected.sort_by_key(key);

    for target in [CostTarget::Full, CostTarget::Time, CostTarget::Energy] {
        for method in [SolveMethod::Recurse, SolveMethod::Queue] {
            let frontier = method.solve(problem, target, &mut DummyReporter);
            let mut actual = frontier.iter_arbitrary().map(|(k, _)| *k).collect_vec();
            actual.sort_by_key(key);

            let expected_for_target = match target {
                CostTarget::Full => expected.clone(),
                CostTarget::Time => expected.iter().copied().min_by_key(|e| e.time).into_iter().collect_vec(),
                CostTarget::Energy => expected.iter().copied().min_by_key(|e| e.energy).into_iter().collect_vec(),
            };

            assert_eq!(expected_for_target, actual, "solve output mismatch for target={:?}, method={:?}", target, method);
        }
    }
}
