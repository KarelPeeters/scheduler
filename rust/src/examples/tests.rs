use itertools::Itertools;
use ordered_float::OrderedFloat;

use crate::core::problem::{Graph, Hardware, Problem};
use crate::core::solver::{DummyReporter, solve};
use crate::core::state::Cost;
use crate::examples::params::{test_problem, TestGraphParams, TestHardwareParams};

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
    expect_solution(&problem, vec![Cost { time: 0.0, energy: 0.0 }]);
}

#[test]
fn single_normal() {
    let problem = test_problem(
        TestGraphParams {
            depth: 0,
            branches: 0,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    expect_solution(&problem, vec![Cost { time: 6000.0, energy: 5000.0 }]);
}

#[test]
fn single_zero_sized_node() {
    let problem = test_problem(
        TestGraphParams {
            depth: 0,
            branches: 0,
            cross: false,
            node_size: 0,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    expect_solution(&problem, vec![Cost { time: 4000.0, energy: 1000.0 }]);
}

#[test]
fn linear_use_single() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 1,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    expect_solution(&problem, vec![Cost { time: 22_000.0, energy: 9000.0 }]);
}

#[test]
fn split_use_both() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    expect_solution(&problem, vec![
        Cost { time: 23_000.0, energy: 15000.0 },
        Cost { time: 38_000.0, energy: 13_000.0 },
    ]);
}

#[test]
fn split_use_single() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 20.0,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("basic", 4000.0, 1000.0)],
    );
    expect_solution(&problem, vec![Cost { time: 38_000.0, energy: 13_000.0 }]);
}

#[test]
fn single_tradeoff() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 1,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 1,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("slow", 4000.0 * 2.0, 1000.0 * 0.75), ("mid", 4000.0, 1000.0), ("fast", 4000.0 / 2.0, 1000.0 * 1.5)],
    );
    expect_solution(&problem, vec![
        Cost { time: 12000.0, energy: 11500.0 },
        Cost { time: 14000.0, energy: 11000.0 },
        Cost { time: 16000.0, energy: 10500.0 },
        Cost { time: 18000.0, energy: 10000.0 },
        Cost { time: 20000.0, energy: 9500.0 },
        Cost { time: 22000.0, energy: 9000.0 },
        Cost { time: 26000.0, energy: 8750.0 },
        Cost { time: 30000.0, energy: 8500.0 },
        Cost { time: 34000.0, energy: 8250.0 },
        Cost { time: 38000.0, energy: 8000.0 },
        Cost { time: 42000.0, energy: 7750.0 },
    ]);
}

#[test]
fn split_tradeoff_deep() {
    let problem = test_problem(
        TestGraphParams {
            depth: 4,
            branches: 2,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("mid", 4000.0, 1000.0), ("fast", 4000.0 / 2.0, 1000.0 * 1.5)],
    );
    expect_solution(&problem, vec![
        Cost { time: 13000.0, energy: 19500.0 },
        Cost { time: 14000.0, energy: 19000.0 },
        Cost { time: 15000.0, energy: 18500.0 },
        Cost { time: 16000.0, energy: 18000.0 },
        Cost { time: 17000.0, energy: 17500.0 },
        Cost { time: 18000.0, energy: 17000.0 },
        Cost { time: 19000.0, energy: 16500.0 },
        Cost { time: 20000.0, energy: 16000.0 },
        Cost { time: 21000.0, energy: 15500.0 },
        Cost { time: 23000.0, energy: 15000.0 },
        Cost { time: 32000.0, energy: 14500.0 },
        Cost { time: 34000.0, energy: 14000.0 },
        Cost { time: 36000.0, energy: 13500.0 },
        Cost { time: 38000.0, energy: 13000.0 },
    ]);
}

#[test]
fn split_tradeoff_shallow() {
    let problem = test_problem(
        TestGraphParams {
            depth: 2,
            branches: 2,
            cross: false,
            node_size: 1000,
            weight_size: None,
        },
        TestHardwareParams {
            core_count: 2,
            share_group: false,
            mem_size_ext: None,
            mem_size_int: None,
            time_per_bit_ext: 1.0,
            time_per_bit_int: 0.5,
            energy_per_bit_ext: 2.0,
            energy_per_bit_int: 1.0,
        },
        &[("slow", 4000.0 * 1.5, 1000.0 / 2.0), ("mid", 4000.0, 1000.0), ("fast", 4000.0 / 2.0, 1000.0 * 1.5)],
    );
    expect_solution(&problem, vec![
        Cost { time: 9000.0, energy: 13500.0 },
        Cost { time: 10000.0, energy: 13000.0 },
        Cost { time: 11000.0, energy: 12500.0 },
        Cost { time: 12000.0, energy: 11500.0 },
        Cost { time: 14000.0, energy: 11000.0 },
        Cost { time: 15000.0, energy: 10500.0 },
        Cost { time: 16000.0, energy: 10000.0 },
        Cost { time: 17000.0, energy: 9500.0 },
        Cost { time: 19000.0, energy: 9000.0 },
        Cost { time: 21000.0, energy: 8500.0 },
        Cost { time: 26000.0, energy: 8000.0 },
        Cost { time: 28000.0, energy: 7500.0 },
        Cost { time: 30000.0, energy: 7000.0 },
        Cost { time: 32000.0, energy: 6500.0 },
    ]);
}

#[track_caller]
pub fn expect_solution(problem: &Problem, mut expected: Vec<Cost>) {
    let frontier = solve(problem, &mut DummyReporter);
    let mut actual = frontier.iter_arbitrary().map(|(k, _)| *k).collect_vec();

    let key = |c: &Cost| (OrderedFloat(c.time), OrderedFloat(c.energy));
    actual.sort_by_key(key);
    expected.sort_by_key(key);

    assert_eq!(expected, actual);
}
