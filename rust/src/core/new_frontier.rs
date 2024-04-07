use std::cmp::max;
use std::ops::RangeInclusive;

use itertools::Itertools;

use crate::core::frontier::{DomBuilder, DomDir, Dominance};
use crate::dom_early_check;
use crate::util::mini::{max_f64, min_f64};

pub struct NewFrontier {
    dimensions: usize,
    max_leaf_len: usize,

    // TODO can we do something faster than this?
    // TODO box?
    root_node: Option<Box<Node>>,

    len: usize,
    curr_max_depth: usize,
}

// TODO custom drop implementation that doesn't recurse as much?
enum Node {
    Branch {
        // TODO if we always cycle, just use depth
        axis: usize,
        key: f64,

        // TODO alternate between eq and neq depending on depth to prevent any bias?
        /// entries with `value <= key`
        node_left: Box<Node>,
        /// entries with `key < value`
        node_right: Box<Node>,
    },
    Leaf(Vec<Vec<f64>>),
}

// TODO use this instead of bool
// enum DropResult {
//     /// Implications:
//     /// * An old entry exists that dominates new.
//     /// * No old entries were dropped.
//     OldDomNew,
//     /// * No old entry dominates new.
//     /// * Some old entries may have been dropped.
//     NoOldDomNew,
// }

impl NewFrontier {
    pub fn new(dimensions: usize, max_leaf_len: usize) -> Self {
        Self {
            dimensions,
            max_leaf_len,
            root_node: None,
            len: 0,
            curr_max_depth: 1,
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

    pub fn for_each_entry<'s>(&'s self, mut f: impl FnMut(usize, &'s Vec<f64>)) {
        self.for_each_node(|depth, node| {
            match node {
                Node::Branch { .. } => {}
                Node::Leaf(entries) => {
                    entries.iter().for_each(|e| f(depth, e))
                }
            }
        })
    }

    fn recurse_for_each_entry<'s>(&self, node: &'s Node, depth: usize, mut f: impl FnMut(usize, &'s Vec<f64>)) {
        self.recurse_for_each_node(node, depth, &mut |_, node| {
            match node {
                Node::Branch { .. } => {}
                Node::Leaf(entries) => {
                    entries.iter().for_each(|e| f(depth, e))
                }
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
            Node::Branch { axis: _, key: _, ref node_left, ref node_right } => {
                self.recurse_for_each_node(node_left, depth + 1, f);
                self.recurse_for_each_node(node_right, depth + 1, f);
            }
            Node::Leaf(_) => {}
        }
    }

    // TODO explicit stack vs recursion?
    // TODO state ptr vs arg+return
    // TODO rename: add and drop dominated
    pub fn add_if_not_dominated(&mut self, new: Vec<f64>) -> bool {
        assert_eq!(new.len(), self.dimensions);

        // check if any old dominates new and remove dominated old entries
        if let Some(root_node) = std::mem::take(&mut self.root_node) {
            let (any_old_dom, new_root_node) = self.recurse_drop_and_check_dom(root_node, &new, false, false);
            self.root_node = new_root_node;
            if any_old_dom {
                assert!(self.root_node.is_some());
                return false;
            }
        }

        // insert the new entry
        // TODO move method to Node struct to make lifetimes easier
        match std::mem::take(&mut self.root_node) {
            None => {
                self.root_node = Some(Box::new(Node::Leaf(vec![new])));
            }
            Some(mut root_node) => {
                self.recurse_add(&mut root_node, new, 0, None);
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
    /// The return values are (old_dom, is_now_empty):
    /// * `true`: an `old` entry exists that dominates or is equal to `new`.
    ///     Through the transitive property this also implies that no old values were dropped,
    ///     and that we can stop recursing early.
    /// * `false`: no `old` entry dominates or is equal to `new`.
    ///     Some old entries that were dominated by `new` may have been dropped.
    fn recurse_drop_and_check_dom(&mut self, node: Box<Node>, new: &[f64], branch_new_any_better: bool, branch_new_any_worse: bool) -> (bool, Option<Box<Node>>) {
        // TODO move this up one level and turn this into an assert?
        if branch_new_any_better && branch_new_any_worse {
            // println!("add incomparable, give up");
            return (
                // no old dominating found
                false,
                // no entries were dropped, so we can't be empty
                Some(node)
            );
        }

        // to avoid re-boxing (which is allocation), we keep the old box alive with a dummy value
        let mut node_box = node;
        let dummy_node = Node::Leaf(vec![]);
        let node = std::mem::replace(&mut *node_box, dummy_node);

        match node {
            Node::Branch { axis, key, node_left, node_right } => {
                let new_key = new[axis];

                let (exit_left, node_left) = self.recurse_drop_and_check_dom(
                    node_left,
                    new,
                    branch_new_any_better,
                    branch_new_any_worse || key < new_key,
                );
                
                if exit_left {
                    let node_left = node_left.unwrap();
                    
                    // reuse the existing box
                    *node_box = Node::Branch { axis, key, node_left, node_right };
                    return (true, Some(node_box));
                }
                
                let (exit_right, node_right) = self.recurse_drop_and_check_dom(
                    node_right,
                    new,
                    branch_new_any_better || new_key < key,
                    branch_new_any_worse
                );

                match (node_left, node_right) {
                    (Some(node_left), Some(node_right)) => {
                        // reuse the existing box
                        *node_box = Node::Branch { axis, key, node_left, node_right };
                        (exit_right, Some(node_box))
                    },
                    (Some(node_left), None) => {
                        assert!(!exit_right);
                        (false, Some(node_left))
                    },
                    (None, Some(node_right)) => {
                        (exit_right, Some(node_right))
                    },
                    (None, None) => {
                        (false, None)
                    }
                }
            }
            Node::Leaf(mut entries) => {
                let mut any_old_dom_or_eq = false;
                let len_before = entries.len();

                entries.retain(|old| {
                    match vec_dominance(new, old) {
                        DomDir::Better => {
                            // println!("dropping {:?}", old);
                            false
                        },
                        DomDir::Worse | DomDir::Equal => {
                            // TODO early exit
                            // println!("not adding because of dominating {:?}", old);
                            any_old_dom_or_eq = true;
                            true
                        }
                        DomDir::Incomparable => true,
                    }
                });

                let len_after = entries.len();
                self.len -= len_before - len_after;

                let node = if len_after == 0 {
                    assert!(!any_old_dom_or_eq);
                    None
                } else {
                    // reuse the existing box
                    *node_box = Node::Leaf(entries);
                    Some(node_box)
                };

                (any_old_dom_or_eq, node)
            }
        }
    }

    // TODO remove self from some of these functions
    fn recurse_add(&mut self, node: &mut Box<Node>, item: Vec<f64>, depth: usize, prev_axis: Option<usize>) {
        match &mut **node {
            &mut Node::Branch { axis, key: value, ref mut node_left, ref mut node_right } => {
                if item[axis] <= value {
                    // println!("recurse_add left");
                    self.recurse_add(node_left, item, depth + 1, Some(axis))
                } else {
                    // println!("recurse_add right");
                    self.recurse_add(node_right, item, depth + 1, Some(axis))
                }
            }
            Node::Leaf(ref mut values) => {
                // println!("recurse_add leaf");
                values.push(item);

                if values.len() > self.max_leaf_len {
                    let values = std::mem::take(values);
                    let new_node = self.split(values, prev_axis);
                    **node = new_node;
                    self.curr_max_depth = max(self.curr_max_depth, depth + 1);
                }
            }
        }
    }

    fn split(&mut self, mut entries: Vec<Vec<f64>>, prev_axis: Option<usize>) -> Node {
        assert!(entries.len() > 1);

        let start_axis = prev_axis.map_or(0, |axis| (axis + 1) % entries[0].len());

        // TODO what if we fail to split? this can't happen in dominance luckily enough

        for axis in start_axis..self.dimensions {
            if let Some(pivot) = sort_pick_pivot(axis, &mut entries) {
                let key = entries[pivot - 1][axis];
                // println!("recurse_add split axis={axis}, value={key}, index={}, entries={:?}", pivot, entries);

                // TODO reclaim values capacity?
                let entries_right = entries.split_off(pivot);

                let node_left = Box::new(Node::Leaf(entries));
                let node_right = Box::new(Node::Leaf(entries_right));

                return Node::Branch {
                    axis,
                    key,
                    node_left,
                    node_right,
                };
            }
        }

        panic!("failed to find any split axis, are there identical tuples?")
    }

    pub fn print(&self, max_depth: usize) {
        println!("Tree:");

        match &self.root_node {
            None => println!("  empty"),
            Some(root_node) => {
                self.recurse_print(root_node, 0, max_depth);
            }
        }
    }

    fn recurse_print(&self, node: &Node, depth: usize, max_depth: usize) {
        let indent = (depth + 1) * 2;
        match *node {
            Node::Branch { axis, key: value, ref node_left, ref node_right } => {
                println!("{:indent$}axis={} value={}", "", axis, value);
                if depth < max_depth {
                    self.recurse_print(node_left, depth + 1, max_depth);
                    self.recurse_print(node_right, depth + 1, max_depth);
                }
            }
            Node::Leaf(ref entries) => {
                println!("{:indent$}leaf len={} entries={:?}", "", entries.len(), entries);
            }
        }
    }
}

// debug utilities
impl NewFrontier {
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
                let res = vec_dominance(entries[i], entries[j]);
                let exp = if i == j { DomDir::Equal } else { DomDir::Incomparable };
                assert_eq!(exp, res);
            }
        }
    }

    fn recurse_assert_valid(&self, node: &Node) {
        match *node {
            Node::Branch { axis, key, ref node_left, ref node_right } => {
                // both branches must be nonempty
                assert!(self.get_subtree_entry_count(node_left) > 0);
                assert!(self.get_subtree_entry_count(node_right) > 0);

                // ranges must respect key
                let range_left = self.get_subtree_axis_value_range(node_left, axis).unwrap();
                let range_right = self.get_subtree_axis_value_range(node_right, axis).unwrap();
                // println!("left range: {:?}, right range: {:?}", range_left, range_right);

                assert!(!range_left.is_empty());
                assert!(!range_right.is_empty());

                // TODO turn into equality again once we figure out key updating?
                // assert_eq!(range_left.end, key);
                assert!(*range_left.end() <= key);
                assert!(key < *range_right.start());

                // recurse
                self.recurse_assert_valid(node_left);
                self.recurse_assert_valid(node_right);
            }
            Node::Leaf(ref values) => {
                // leafs must be nonempty and can't be too large
                assert!(0 < values.len() && values.len() <= self.max_leaf_len, "Invalid leaf length {}", values.len());
            }
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
            let v = e[axis];
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

fn sort_pick_pivot(axis: usize, slice: &mut [Vec<f64>]) -> Option<usize> {
    let cmp = |a: &Vec<f64>, b: &Vec<f64>| a[axis].total_cmp(&b[axis]);
    slice.sort_unstable_by(cmp);

    let mut pivot_index = None;
    let mut best_dist = usize::MAX;

    // TODO iterate starting at center
    for i in 1..slice.len() {
        if cmp(&slice[i], &slice[i - 1]).is_eq() {
            continue;
        }

        let dist = ((slice.len() + 1) / 2).abs_diff(i);
        if dist < best_dist {
            best_dist = dist;
            pivot_index = Some(i);
        }
    }

    if let Some(pivot_index) = pivot_index {
        for i in 0..slice.len() {
            if i <= pivot_index {
                assert!(slice[i][axis] <= slice[pivot_index][axis]);
            } else {
                assert!(slice[i][axis] > slice[pivot_index][axis]);
            }
        }
    }

    pivot_index
}

fn vec_dominance(self_value: &[f64], other_value: &[f64]) -> DomDir {
    assert_eq!(self_value.len(), other_value.len());
    let len = self_value.len();

    let mut builder = DomBuilder::new(self_value, other_value);
    for i in 0..len {
        builder.minimize(|x| x[i]);
        dom_early_check!(builder);
    }
    builder.finish()
}


impl Dominance for Vec<f64> {
    type Aux = ();

    fn dominance(&self, other: &Self, _: &()) -> DomDir {
        assert_eq!(self.len(), other.len());

        let mut dom = DomBuilder::new(self, other);
        for i in 0..self.len() {
            dom.minimize(|v| v[i]);
            dom_early_check!(dom);
        }
        dom.finish()
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    use crate::core::frontier::Frontier;

    use crate::core::new_frontier::NewFrontier;

    #[test]
    fn pivot() {}

    #[test]
    fn correctness() {
        let dimensions = 256;
        let max_leaf_len = 1;
        let n = 1024*32;

        let mut rng = SmallRng::seed_from_u64(0);
        let mut frontier = NewFrontier::new(dimensions, max_leaf_len);

        let mut baseline  = Frontier::new();
        
        
        let mut total_gen = 0.0;
        let mut total_add_new = 0.0;
        let mut total_add_old = 0.0;

        for _ in 0..n {
            // frontier.print(usize::MAX);
            // frontier.assert_valid();

            println!("Frontier length={}", frontier.len);

            let start_gen = Instant::now();
            let value = (0..dimensions).map(|_| rng.gen_range(0.0..1.0)).collect_vec();
            total_gen += start_gen.elapsed().as_secs_f64();
            // println!("Adding {:?}", value);

            let start_old = Instant::now();
            let added_base = baseline.add(&value, &(), || ());
            total_add_old += start_old.elapsed().as_secs_f64();

            let start_new = Instant::now();
            let added = frontier.add_if_not_dominated(value);
            total_add_new += start_new.elapsed().as_secs_f64();
            
            assert_eq!(added, added_base);
            assert_eq!(frontier.len(), baseline.len());
        }
        
        println!("Times:");
        println!("  gen=     {}s", total_gen);
        println!("  add_old= {}s", total_add_old);
        println!("  add_new= {}s", total_add_new);

        let depths = format!("{:?}", frontier.collect_entry_depths());
        std::fs::write("ignored/depths.txt", &depths).unwrap();
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
}
