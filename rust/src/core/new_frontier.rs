use std::cmp::{max, Ordering};
use itertools::enumerate;

pub struct NewFrontier {
    dimensions: usize,
    max_leaf_len: usize,

    node_index_root: usize,
    nodes: Vec<Node>,

    curr_max_depth: usize,
}

enum Node {
    Branch {
        // TODO if we always cycle, just use depth
        axis: usize,
        value: f64,

        node_index_lte: usize,
        node_index_gt: usize,
    },
    Leaf(Vec<Vec<f64>>),
}

impl NewFrontier {
    pub fn new(dimensions: usize, max_leaf_len: usize) -> Self {
        Self {
            dimensions,
            max_leaf_len,
            node_index_root: 0,
            nodes: vec![Node::Leaf(Vec::new())],
            curr_max_depth: 1,
        }
    }

    pub fn add(&mut self, item: Vec<f64>) {
        assert_eq!(item.len(), self.dimensions);
        self.recurse_add(self.node_index_root, item, 0, None);
    }
    
    pub fn would_add(&self, item: &[f64]) -> (bool, usize) {
        self.recurse_would_add(item, self.node_index_root)
    }
    
    fn recurse_would_add(&self, item: &[f64], node_index: usize) -> (bool, usize) {
        match &self.nodes[node_index] {
            &Node::Branch { axis, value, node_index_lte, node_index_gt } => {
                // TODO store min and max too in the branch node (or in all of them?)
                // new, old
                match item[axis].total_cmp(&value) {
                    Ordering::Less => {
                        // we only need to check the upper branch
                        self.recurse_would_add(item, node_index_gt)
                    },
                    Ordering::Greater | Ordering::Equal => {
                        // we need to check both branches (with short circuiting for now,
                        // TODO this short-circuiting go away once we need to remove old nodes)
                        
                        
                        let (would_add, checked_lte) = self.recurse_would_add(item, node_index_lte);
                        if !would_add {
                            return (false, checked_lte);
                        }
                        
                        let (would_add, checked_gt) = self.recurse_would_add(item, node_index_gt);
                        return (would_add, checked_lte + checked_gt);
                    }
                }
            }
            Node::Leaf(values) => {
                for (i, old) in enumerate(values) {
                    if item.iter().zip(old).all(|(a, b)| a <= b) {
                        // old is >= new, don't add
                        return (false, i+1);
                    }
                }

                (true, values.len())
            }
        }
    }

    fn recurse_add(&mut self, node_index: usize, item: Vec<f64>, depth: usize, prev_axis: Option<usize>) {
        match self.nodes[node_index] {
            Node::Branch { axis, value, node_index_lte, node_index_gt } => {
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
                    value,
                    node_index_lte: lhs_index,
                    node_index_gt: rhs_index,
                };
            }
        }

        panic!("failed to find any split axis, are there identical tuples?")
    }

    pub fn print(&self, max_depth: usize) {
        self.recurse_print(self.node_index_root, 0, max_depth);
    }

    fn recurse_print(&self, node_index: usize, depth: usize, max_depth: usize) {
        match &self.nodes[node_index] {
            Node::Branch { axis, value, node_index_lte, node_index_gt } => {
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
        let max_leaf_len = 8;
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
                let start_would_add = Instant::now();
                let tries = 10_000;
                for _ in 0..tries {
                    let item = (0..dimensions).map(|_| rng.gen_range(0..1024) as f64).collect_vec();
                    let (_, checked) = frontier.would_add(&item);
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