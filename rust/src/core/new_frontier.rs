use std::cmp::{max, Ordering};
use std::collections::HashSet;
use std::ops::Range;

use itertools::enumerate;

use crate::util::mini::{max_f64, min_f64};

pub struct NewFrontier {
    dimensions: usize,
    max_leaf_len: usize,

    root_node: usize,
    nodes: Vec<Node>,

    len: usize,
    curr_max_depth: usize,
}

enum Node {
    Branch {
        // TODO if we always cycle, just use depth
        axis: usize,
        key: f64,

        // TODO alternate between eq and neq depending on depth to prevent any bias?
        /// entries with `value <= key` 
        node_left: usize,
        /// entries with `key < value`
        node_right: usize,
    },
    Leaf(Vec<Vec<f64>>),
}

impl NewFrontier {
    pub fn new(dimensions: usize, max_leaf_len: usize) -> Self {
        Self {
            dimensions,
            max_leaf_len,
            root_node: 0,
            nodes: vec![Node::Leaf(Vec::new())],
            len: 0,
            curr_max_depth: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn add(&mut self, item: Vec<f64>) {
        assert_eq!(item.len(), self.dimensions);
        self.recurse_add(self.root_node, item, 0, None);
        self.len += 1;
    }

    // TODO explicit stack vs recursion?
    pub fn add_if_not_dominated(&mut self, new: Vec<f64>) -> bool {
        let mut new_dominates_any = false;
        
        
        todo!()
    }
    
    fn recurse_add_if_not_dominated(&mut self, node: usize, new: Vec<f64>, new_dominates_any: &bool) -> bool {
        
    }

    // mostly equivalent to `would_add` and `add`
    pub fn is_dominated_by_any(&self, item: &[f64]) -> (bool, usize) {
        self.recurse_is_dominated_by_any(item, self.root_node)
    }
    
    fn recurse_is_dominated_by_any(&self, item: &[f64], node_index: usize) -> (bool, usize) {
        match &self.nodes[node_index] {
            &Node::Branch { axis, key: value, node_left: node_index_lte, node_right: node_index_gt } => {
                // TODO store min and max too in the branch (or all) node, so we can skip even more?
                // new, old
                match item[axis].total_cmp(&value) {
                    Ordering::Less => {
                        // we only need to check the lower branch
                        self.recurse_is_dominated_by_any(item, node_index_lte)
                    },
                    Ordering::Greater | Ordering::Equal => {
                        // we need to check both branches
                        let (dom_lte, checked_lte) = self.recurse_is_dominated_by_any(item, node_index_lte);
                        // TODO enable or disable this? which approximates the real stats best?
                        // if dom_lte {
                        //     return (true, checked_lte);
                        // }
                        let (dom_gte, checked_gt) = self.recurse_is_dominated_by_any(item, node_index_gt);

                        return (dom_lte || dom_gte, checked_lte + checked_gt);
                    }
                }
            }
            Node::Leaf(values) => {
                for (i, old) in enumerate(values) {
                    if item.iter().zip(old).all(|(a, b)| a <= b) {
                        // old is >= new, dominated
                        return (true, i + 1);
                    }
                }

                (false, values.len())
            }
        }
    }

    fn recurse_add(&mut self, node_index: usize, item: Vec<f64>, depth: usize, prev_axis: Option<usize>) {
        match self.nodes[node_index] {
            Node::Branch { axis, key: value, node_left: node_index_lte, node_right: node_index_gt } => {
                if item[axis] <= value {
                    self.recurse_add(node_index_lte, item, depth + 1, Some(axis))
                } else {
                    self.recurse_add(node_index_gt, item, depth + 1, Some(axis))
                }
            }
            Node::Leaf(ref mut values) => {
                values.push(item);
                if values.len() > self.max_leaf_len {
                    let values = std::mem::take(values);
                    let node = self.split(values, prev_axis);
                    self.nodes[node_index] = node;

                    self.curr_max_depth = max(self.curr_max_depth, depth + 1);
                }
            }
        }
    }

    fn split(&mut self, mut values: Vec<Vec<f64>>, prev_axis: Option<usize>) -> Node {
        assert!(values.len() > 1);

        let start_axis = prev_axis.map_or(0, |axis| (axis + 1) % values[0].len());

        // TODO what if we fail to split? this can't happen in dominance luckily enough

        for axis in start_axis..self.dimensions {
            if let Some(pivot) = sort_pick_pivot(axis, &mut values) {
                let value = values[pivot][axis];
                // TODO reclaim values capacity?
                let rhs = values.split_off(pivot);
                let lhs_index = self.nodes.len();
                let rhs_index = self.nodes.len() + 1;
                self.nodes.push(Node::Leaf(values));
                self.nodes.push(Node::Leaf(rhs));

                return Node::Branch {
                    axis,
                    key: value,
                    node_left: lhs_index,
                    node_right: rhs_index,
                };
            }
        }

        panic!("failed to find any split axis, are there identical tuples?")
    }

    pub fn print(&self, max_depth: usize) {
        self.recurse_print(self.root_node, 0, max_depth);
    }

    fn recurse_print(&self, node_index: usize, depth: usize, max_depth: usize) {
        match &self.nodes[node_index] {
            Node::Branch { axis, key: value, node_left: node_index_lte, node_right: node_index_gt } => {
                println!("{:indent$}axis={} value={}", "", axis, value, indent = depth);
                if depth < max_depth {
                    self.recurse_print(*node_index_lte, depth + 1, max_depth);
                    self.recurse_print(*node_index_gt, depth + 1, max_depth);
                }
            }
            Node::Leaf(values) => {
                println!("{:indent$}leaf len={}", "", values.len(), indent = depth);
            }
        }
    }
}

// debug utilities
impl NewFrontier {
    pub fn assert_valid(&self) {
        // check len
        assert_eq!(self.len, self.get_subtree_entry_count(self.root_node));
        
        // check tree invariants
        let mut unseen_nodes = HashSet::from_iter(0..self.nodes.len());
        self.recurse_assert_valid(self.root_node, &mut unseen_nodes);
    }

    fn recurse_assert_valid(&self, node: usize, unseen_nodes: &mut HashSet<usize>) {
        assert!(unseen_nodes.remove(&node));

        match &self.nodes[node] {
            &Node::Branch { axis, key: value, node_left, node_right } => {
                // both branches must be nonempty
                assert!(self.get_subtree_entry_count(node_left) > 0);
                assert!(self.get_subtree_entry_count(node_right) > 0);
                
                // ranges must respect key
                let range_left = self.get_subtree_axis_value_range(node_left, axis);
                let range_right = self.get_subtree_axis_value_range(node_right, axis);
                assert_eq!(range_left.end, value);
                assert!(value < range_right.start);
                
                // recurse
                self.recurse_assert_valid(node_left, unseen_nodes);
                self.recurse_assert_valid(node_right, unseen_nodes);
            }
            Node::Leaf(values) => {
                // leafs must be nonempty and can't be too large
                assert!(0 < values.len() && values.len() <= self.max_leaf_len);
            }
        }
    }

    fn get_subtree_entry_count(&self, node: usize) -> usize {
        match self.nodes[node] {
            Node::Branch { axis: _, key: _, node_left, node_right } => {
                self.get_subtree_node_count(node_left) + self.get_subtree_node_count(node_right) + 1
            }
            Node::Leaf(ref entries) => entries.len(),
        }
    }

    fn get_subtree_axis_value_range(&self, node: usize, axis: usize) -> Range<f64> {
        match self.nodes[node] {
            Node::Branch { axis: _, key: _, node_left, node_right } => {
                let left = self.get_subtree_axis_value_range(node_left, axis);
                let right = self.get_subtree_axis_value_range(node_right, axis);
                min_f64(left.start, right.start)..max_f64(left.end, right.end)
            }
            Node::Leaf(ref entries) => {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for item in entries {
                    min = min_f64(min, item[axis]);
                    max = max_f64(max, item[axis]);
                }
                min..max
            }
        }
    }

    pub fn collect_entry_depths(&self) -> Vec<usize> {
        let mut result = Vec::new();
        self.recurse_collect_entry_depths(self.root_node, 0, &mut result);
        result
    }

    pub fn recurse_collect_entry_depths(&self, node: usize, depth: usize, result: &mut Vec<usize>) {
        match self.nodes[node] {
            Node::Branch { axis: _, key: _, node_left, node_right } => {
                self.recurse_collect_entry_depths(node_left, depth + 1, result);
                self.recurse_collect_entry_depths(node_right, depth + 1, result);
            }
            Node::Leaf(ref entries) => {
                for _ in 0..entries.len() {
                    result.push(depth);
                }
            }
        }
    }
}

fn sort_pick_pivot(axis: usize, slice: &mut [Vec<f64>]) -> Option<usize> {
    let cmp = |a: &Vec<f64>, b: &Vec<f64>| a[axis].total_cmp(&b[axis]);
    slice.sort_unstable_by(cmp);

    let mut pivot_index = None;
    let mut best_dist = usize::MAX;

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

    pivot_index
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    use crate::core::new_frontier::NewFrontier;

    #[test]
    fn random() {
        let dimensions = 512;
        let max_leaf_len = 1;
        let n = 1_000_000;

        let mut rng = SmallRng::seed_from_u64(0);
        let mut frontier = NewFrontier::new(dimensions, max_leaf_len);

        let start = Instant::now();
        let mut total_would_add = 0.0;
        let mut total_add = 0.0;
        
        for i in 0..n {
            if i % 100_000 == 0 {
                println!("progress={}", i as f64 / n as f64);

                println!("  took {}s (add {}s)", start.elapsed().as_secs_f64(), total_add);
                println!("  nodes={}, nodes/value={}", frontier.nodes.len(), frontier.nodes.len() as f64 / i as f64);
                println!("  max_depth={}", frontier.curr_max_depth);
                
                let mut total_checked = 0;
                let tries = 10_000;
                for _ in 0..tries {
                    let item = (0..dimensions).map(|_| {
                        // TODO test fixed-unlucky/bad axes
                        rng.gen_range(0..5) as f64
                    }).collect_vec();
                    
                    let start_would_add = Instant::now();
                    let (_, checked) = frontier.is_dominated_by_any(&item);
                    total_would_add += start_would_add.elapsed().as_secs_f64();
                    total_checked += checked;
                }

                let average_checked = total_checked as f64 / tries as f64;
                let fraction_checked = average_checked / i as f64;
                println!("  would_add total took {}s, average checks={}, fraction checked={}", total_would_add, average_checked, fraction_checked);
            }

            let item = (0..dimensions).map(|_| rng.gen_range(0..1024) as f64).collect_vec();
            
            let start_add = Instant::now();
            frontier.add(item);
            total_add += start_add.elapsed().as_secs_f64();
        }

        // frontier.print(usize::MAX);
    }
}