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
    expect_solution(&problem, vec![Cost { time: 6000.0, energy: 1000.0 }]);
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
