use std::ops::RangeInclusive;

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::core::frontier::{DomDir, Dominance};
use crate::core::new_frontier::SparseVec;
use crate::util::mini::{max_f64, min_f64};

pub struct LinearFrontier {
    dimensions: usize,
    len: usize,
    // TODO is doing our own indexing and garbage collection faster?
    root_node: Option<Node>,
    axis_order: Vec<usize>,
}

// TODO custom drop implementation that doesn't recurse as much?
enum Node {
    Leaf(SparseVec),
    Branch {
        axis: usize,
        entries: Vec<(f64, Node)>,
    },
}

impl LinearFrontier {
    pub fn new(dimensions: usize) -> Self {
        let mut order = (0..dimensions).collect_vec();
        let mut rng = SmallRng::seed_from_u64(0);
        order.shuffle(&mut rng);
        
        Self {
            dimensions,
            len: 0,
            root_node: None,
            axis_order: order, 
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn node_count(&self) -> usize {
        // TODO either delete or optimize (ie. compute incrementally like len)
        let mut count = 0;
        self.for_each_node(|_, _| count += 1);
        count
    }

    pub fn for_each_entry<'s>(&'s self, mut f: impl FnMut(usize, &'s SparseVec)) {
        self.for_each_node(|depth, node| {
            match node {
                Node::Branch { .. } => {}
                Node::Leaf(e) => f(depth, e),
            }
        })
    }

    fn recurse_for_each_entry<'s>(&self, node: &'s Node, depth: usize, mut f: impl FnMut(usize, &'s SparseVec)) {
        self.recurse_for_each_node(node, depth, &mut |_, node| {
            match node {
                Node::Branch { .. } => {}
                Node::Leaf(e) => f(depth, e),
            }
        })
    }

    fn for_each_node<'s>(&'s self, mut f: impl FnMut(usize, &'s Node)) {
        if let None = &self.root_node {} else if let Some(root_node) = &self.root_node {
            self.recurse_for_each_node(root_node, 0, &mut f);
        }
    }

    fn recurse_for_each_node<'s>(&self, node: &'s Node, depth: usize, f: &mut impl FnMut(usize, &'s Node)) {
        f(depth, node);

        match *node {
            Node::Branch { axis: _, ref entries } => {
                for (_, entry) in entries {
                    self.recurse_for_each_node(entry, depth + 1, f);
                }
            }
            Node::Leaf(_) => {}
        }
    }

    // TODO explicit stack vs recursion?
    // TODO state ptr vs arg+return
    // TODO rename: add and drop dominated
    #[inline(never)]
    pub fn add_if_not_dominated(&mut self, new: SparseVec) -> bool {
        // check if any old dominates new and remove dominated old entries
        let mut entries_checked = 0;
        if let Some(mut root_node) = std::mem::take(&mut self.root_node) {
            let (any_old_dom, empty) = self.recurse_drop_and_check_dom(&mut root_node, &new, false, false, &mut entries_checked);

            // put root node back (we just took it out for self mut reasons)
            if !empty {
                self.root_node = Some(root_node);
            }

            if any_old_dom {
                assert!(!empty);
                return false;
            }
        }

        // println!("entries_checked={}/{} = {}", entries_checked, self.len(), entries_checked as f64 / self.len() as f64);

        // insert the new entry
        // TODO move method to Node struct to make lifetimes easier
        match std::mem::take(&mut self.root_node) {
            None => {
                self.root_node = Some(Node::Leaf(new));
            }
            Some(mut root_node) => {
                self.recurse_add(&mut root_node, new, 0, 0);
                assert!(self.root_node.is_none());
                self.root_node = Some(root_node);
            }
        }
        self.len += 1;

        true
    }

    // TODO better name
    // TODO immediately insert or remember the index if we find the right splot?
    // TODO change return type to a proper enum, bools and tuples are confusing
    /// Drops old entries that are dominated by `new`, in the subtree of `node`.
    ///
    /// If during tree descent, we ever picked the side of a key value that implies
    /// we know that the corresponding `new` axis value will be better or worse for sure,
    /// this is reflected in `new_any_better` and `new_any_worse`.
    ///
    /// The return values are (old_dom, is_now_empty).
    /// The meaning of `old_dom` is:
    /// * `true`: an `old` entry exists that dominates or is equal to `new`.
    ///     Through the transitive property this also implies that no old values were dropped,
    ///     and that we can stop recursing early.
    /// * `false`: no `old` entry dominates or is equal to `new`.
    ///     Some old entries that were dominated by `new` may have been dropped.
    #[inline(never)]
    fn recurse_drop_and_check_dom(&mut self, node: &mut Node, new: &SparseVec, branch_new_any_better: bool, branch_new_any_worse: bool, entries_checked: &mut usize) -> (bool, bool) {
        assert!(!(branch_new_any_better && branch_new_any_worse), "should have been filtered out at a higher level");

        match node {
            Node::Branch { axis, entries } => {
                // TODO switch to linear search once the list is short enough
                let new_key = new.get(self.axis_order[*axis]);
                let (limit_lower, index_equal, start_higher) = match entries.binary_search_by(|(k, _)| k.total_cmp(&new_key)) {
                    Ok(index) => (index, Some(index), index + 1),
                    Err(index) => (index, None, index),
                };

                let mut next_index = 0;
                let mut any_exit = false;

                entries.retain_mut(|(_, child)| {
                    // TODO custom retain_mut that supports early exit?
                    if any_exit {
                        return true;
                    }

                    let index = next_index;
                    next_index += 1;

                    let (better, worse) = if index < limit_lower {
                        // TODO turn this into piecewise iteration instead
                        if branch_new_any_better {
                            return true;
                        }
                        
                        (branch_new_any_better, true)
                    } else if Some(index) == index_equal {
                        (branch_new_any_better, branch_new_any_worse)
                    } else {
                        assert!(index >= start_higher);

                        // TODO turn this into piecewise iteration instead
                        if branch_new_any_worse {
                            return true;
                        }
                        
                        (true, branch_new_any_worse)
                    };

                    let (exit, empty) = self.recurse_drop_and_check_dom(child, new, better, worse, entries_checked);
                    any_exit |= exit;
                    !empty
                });

                (any_exit, entries.is_empty())
            }
            Node::Leaf(old) => {
                *entries_checked += 1;

                match new.dominance(&old, &()) {
                    DomDir::Better => {
                        // drop old
                        self.len -= 1;
                        (false, true)
                    }
                    DomDir::Worse | DomDir::Equal => {
                        // report that new is dominated, keep old
                        (true, false)
                    }
                    DomDir::Incomparable => {
                        // report that new is not dominated, keep old
                        (false, false)
                    }
                }
            }
        }
    }

    fn get_subtree_sample<'n>(&self, node: &'n Node) -> &'n SparseVec {
        match node {
            Node::Branch { axis: _, entries } => self.get_subtree_sample(&entries[0].1),
            Node::Leaf(entry) => entry,
        }
    }

    // TODO remove self from some of these functions
    // TODO pick/learn/... an optimal key ordering to get below O(k)
    #[inline(never)]
    fn recurse_add(&mut self, node: &mut Node, new: SparseVec, depth: usize, next_axis: usize) {
        assert!(next_axis < self.dimensions);

        match node {
            &mut Node::Branch { axis: branch_axis, ref mut entries } => {
                // insert earlier branch if any difference is found
                // TODO only iterate over axes that either SparseVec has
                let old = self.get_subtree_sample(&entries[0].1);
                for split_axis in next_axis..branch_axis {
                    let old_key = old.get(self.axis_order[split_axis]);
                    let new_key = new.get(self.axis_order[split_axis]);
                    if old_key == new_key {
                        continue;
                    }

                    let old_entry = (old_key, Node::Branch { axis: branch_axis, entries: std::mem::take(entries) });
                    let new_entry = (new_key, Node::Leaf(new));
                    let entries = if old_key < new_key {
                        vec![old_entry, new_entry]
                    } else {
                        vec![new_entry, old_entry]
                    };

                    *node = Node::Branch {
                        axis: split_axis,
                        entries,
                    };
                    return;
                }

                let new_key = new.get(self.axis_order[branch_axis]);
                match entries.binary_search_by(|(k, _)| k.total_cmp(&new_key)) {
                    Ok(index) => {
                        // existing branch found, continue recursing
                        self.recurse_add(&mut entries[index].1, new, depth + 1, next_axis + 1);
                    }
                    Err(index) => {
                        // add as new branch
                        entries.insert(index, (new_key, Node::Leaf(new)));
                    }
                }
            }
            Node::Leaf(old) => {
                let old = std::mem::take(old);

                // find the first different axis
                // TODO only iterate over existing axes
                for split_axis in next_axis..self.dimensions {
                    let old_key = old.get(self.axis_order[split_axis]);
                    let new_key = new.get(self.axis_order[split_axis]);

                    if old_key == new_key {
                        continue;
                    }

                    let old_entry = (old_key, Node::Leaf(old));
                    let new_entry = (new_key, Node::Leaf(new));

                    let entries = if old_key < new_key {
                        vec![old_entry, new_entry]
                    } else {
                        vec![new_entry, old_entry]
                    };

                    *node = Node::Branch {
                        axis: split_axis,
                        entries,
                    };
                    return;
                }

                panic!("failed to find any split axis, are there identical tuples?")
            }
        }
    }

    pub fn print(&self, max_depth: usize) {
        println!("Tree: len={}", self.len);

        match &self.root_node {
            None => println!("  empty"),
            Some(root_node) => {
                self.recurse_print(root_node, 0, max_depth);
            }
        }
    }

    fn recurse_print(&self, node: &Node, depth: usize, max_depth: usize) {
        if depth > max_depth {
            return;
        }

        let indent = (depth + 1) * 2;
        match *node {
            Node::Branch { axis, ref entries } => {
                println!("{:indent$}branch len={}, axis={}", "", self.get_subtree_entry_count(node), axis);
                for (k, e) in entries {
                    println!("{:indent$}  k={}", "", k);
                    self.recurse_print(e, depth + 1, max_depth);
                }
            }
            Node::Leaf(_) => {
                println!("{:indent$}leaf", "");
            }
        }
    }
}

// debug utilities
impl LinearFrontier {
    pub fn assert_valid(&self) {
        // length and invariants
        match &self.root_node {
            None => {
                assert_eq!(self.len, 0);
            }
            Some(root_node) => {
                assert_eq!(self.len, self.get_subtree_entry_count(root_node));
                self.recurse_assert_valid(root_node)
            }
        }

        // check incomparable
        let mut entries = vec![];
        self.for_each_entry(|_, e| entries.push(e));
        assert_eq!(self.len, entries.len());
        for i in 0..entries.len() {
            for j in 0..entries.len() {
                let res = entries[i].dominance(&entries[j], &());
                let exp = if i == j { DomDir::Equal } else { DomDir::Incomparable };
                assert_eq!(exp, res);
            }
        }
    }

    fn recurse_assert_valid(&self, node: &Node) {
        match *node {
            Node::Branch { axis, ref entries } => {
                // counts must match
                let total = entries.iter().map(|(_, n)| self.get_subtree_entry_count(n)).sum::<usize>();
                assert_eq!(self.get_subtree_entry_count(node), total);

                for &(k, ref n) in entries {
                    // all branches must be nonempty
                    assert!(self.get_subtree_entry_count(n) > 0);

                    // ranges must match key
                    let range = self.get_subtree_axis_value_range(n, axis).unwrap();
                    assert_eq!(range, k..=k);
                    assert!(!range.is_empty());

                    // recurse
                    self.recurse_assert_valid(n);
                }
            }
            Node::Leaf(_) => {}
        }
    }

    // TODO combine
    fn get_subtree_entry_count(&self, node: &Node) -> usize {
        let mut count = 0;
        self.recurse_for_each_entry(node, 0, |_, _| count += 1);
        count
    }

    fn get_subtree_axis_value_range(&self, node: &Node, axis: usize) -> Option<RangeInclusive<f64>> {
        let mut range = None;
        self.recurse_for_each_entry(node, 0, |_, e| {
            let v = e.get(self.axis_order[axis]);
            range = Some(range.map_or((v, v), |(min, max)| {
                (min_f64(min, v), max_f64(max, v))
            }));
        });
        range.map(|(min, max)| min..=max)
    }

    pub fn collect_entry_depths(&self) -> Vec<usize> {
        let mut result = Vec::new();
        self.for_each_entry(|d, _| result.push(d));
        result
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use itertools::Itertools;
    use rand::{Rng, thread_rng};

    use crate::core::frontier::Frontier;
    use crate::core::linear_frontier::LinearFrontier;
    use crate::core::new_frontier::SparseVec;

    #[test]
    fn correctness() {
        let dimensions = 16;
        let n = 64;

        // let mut rng = SmallRng::seed_from_u64(0);
        let mut rng = thread_rng();
        let mut frontier = LinearFrontier::new(dimensions);

        let mut baseline = Frontier::new();

        let mut total_gen = 0.0;
        let mut total_add_new = 0.0;
        let mut total_add_old = 0.0;

        for i in 0..n {
            println!("N={}, Frontier length={}", i, frontier.len);
            // frontier.print(usize::MAX);
            // frontier.assert_valid();

            let start_gen = Instant::now();

            let full_value = (0..dimensions).map(|_| {
                match rng.gen_range(0..3) {
                    0 => f64::NEG_INFINITY,
                    1 => f64::INFINITY,
                    2 => rng.gen_range(0.0..1.0),
                    _ => unreachable!(),
                }
            }).collect_vec();
            let value = SparseVec::from_iter(full_value.iter().copied());
            total_gen += start_gen.elapsed().as_secs_f64();
            // println!("Adding {:?}", value);

            let start_old = Instant::now();
            let added_base = baseline.add(&full_value, &(), || ());
            total_add_old += start_old.elapsed().as_secs_f64();

            let start_new = Instant::now();
            let added = frontier.add_if_not_dominated(value);
            total_add_new += start_new.elapsed().as_secs_f64();

            assert_eq!(added, added_base);
            assert_eq!(frontier.len(), baseline.len());
        }

        let depths = format!("{:?}", frontier.collect_entry_depths());
        std::fs::write("ignored/depths.txt", &depths).unwrap();

        frontier.print(10);

        println!("Times:");
        println!("  gen=     {}s", total_gen);
        println!("  add_old= {}s", total_add_old);
        println!("  add_new= {}s", total_add_new);
    }

    #[test]
    fn performance() {
        // let dimensions = 512;
        // let max_leaf_len = 1;
        // let n = 1_000_000;
        //
        // let mut rng = SmallRng::seed_from_u64(0);
        // let mut frontier = NewFrontier::new(dimensions, max_leaf_len);
        //
        // let start = Instant::now();
        // let mut total_would_add = 0.0;
        // let mut total_add = 0.0;
        //
        // for i in 0..n {
        //     if i % 100_000 == 0 {
        //         println!("progress={}", i as f64 / n as f64);
        //
        //         println!("  took {}s (add {}s)", start.elapsed().as_secs_f64(), total_add);
        //         println!("  nodes={}, nodes/value={}", frontier.len(), frontier.len() as f64 / i as f64);
        //         println!("  max_depth={}", frontier.curr_max_depth);
        //
        //         let mut total_checked = 0;
        //         let tries = 10_000;
        //         for _ in 0..tries {
        //             let item = (0..dimensions).map(|_| {
        //                 // TODO test fixed-unlucky/bad axes
        //                 rng.gen_range(0..5) as f64
        //             }).collect_vec();
        //
        //             let start_would_add = Instant::now();
        //             let (_, checked) = frontier.is_dominated_by_any(&item);
        //             total_would_add += start_would_add.elapsed().as_secs_f64();
        //             total_checked += checked;
        //         }
        //
        //         let average_checked = total_checked as f64 / tries as f64;
        //         let fraction_checked = average_checked / i as f64;
        //         println!("  would_add total took {}s, average checks={}, fraction checked={}", total_would_add, average_checked, fraction_checked);
        //     }
        //
        //     let item = (0..dimensions).map(|_| rng.gen_range(0..1024) as f64).collect_vec();
        //
        //     let start_add = Instant::now();
        //     frontier.add(item);
        //     total_add += start_add.elapsed().as_secs_f64();
        //
        //     frontier.print(usize::MAX);
        //     frontier.assert_valid();
        // }

        // frontier.print(usize::MAX);
    }

    #[test]
    fn bug_from_solver() {
        let inf = f64::INFINITY;
        let data = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, inf, inf, inf, inf],
            [0.0, 2000.0, 1000.0, 0.0, 1000.0, 0.0, inf, 1000.0, inf, inf, inf],
            [1000.0, 2000.0, 1000.0, 1000.0, 1000.0, 0.0, inf, 0.0, inf, inf, inf],
            [1000.0, 2100.0, 5000.0, 5000.0, 1000.0, -inf, inf, -inf, 5000.0, inf, inf],
            [5000.0, 2100.0, 5000.0, 5000.0, 5000.0, -inf, inf, -inf, 0.0, inf, inf],
            [5000.0, 2200.0, 9000.0, 9000.0, 5000.0, -inf, -inf, -inf, -inf, 9000.0, inf],
            [9000.0, 2200.0, 9000.0, 9000.0, 9000.0, -inf, -inf, -inf, -inf, 0.0, inf],
        ];

        let mut frontier = LinearFrontier::new(data[0].len());
        for line in data {
            frontier.print(usize::MAX);

            frontier.add_if_not_dominated(SparseVec::from_iter(line.into_iter()));
            frontier.assert_valid();
        }

        frontier.print(usize::MAX);
    }
}
