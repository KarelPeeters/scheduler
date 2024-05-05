use std::collections::HashSet;

// TODO better name for this type and the trait
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DomDir {
    Better,
    Worse,
    Equal,
    Incomparable,
}

pub trait Dominance {
    type Aux;
    fn dominance(&self, other: &Self, aux: &Self::Aux) -> DomDir;
}

#[derive(Clone)]
pub struct Entry<K, V> {
    key: K,
    value: V,
    prev_index: Option<usize>,
    next_index: Option<usize>,
}

#[derive(Clone)]
pub struct Frontier<K, V> {
    entries: Vec<Entry<K, V>>,
    first_index: Option<usize>,
    last_index: Option<usize>,
    pub dominance_calculations: u64,

    pub count_add_try: u64,
    pub count_add_success: u64,
    pub count_add_removed: u64,
}

impl<K, V> Frontier<K, V> {
    pub fn empty() -> Self {
        Self { entries: vec![], first_index: None, last_index: None, dominance_calculations: 0, count_add_try: 0, count_add_success: 0, count_add_removed: 0 }
    }

    pub fn single(k: K, v: V) -> Self {
        let mut result = Self::empty();
        result.add_entry(k, v, true);
        result
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter_arbitrary(&self) -> impl Iterator<Item=(&K, &V)> {
        self.entries.iter().map(|e| (&e.key, &e.value))
    }

    pub fn into_iter_arbitrary(self) -> impl Iterator<Item=(K, V)> {
        self.entries.into_iter().map(|e| (e.key, e.value))
    }

    pub fn keys(&self) -> impl Iterator<Item=&K> {
        self.iter_arbitrary().map(|(k, _)| k)
    }

    pub fn iter_lru(&self) -> impl Iterator<Item=(&K, &V)> {
        let mut next_index = self.first_index;
        std::iter::from_fn(move || {
            let index = next_index?;
            let entry = &self.entries[index];
            next_index = entry.next_index;
            Some((&entry.key, &entry.value))
        })
    }

    fn assert_valid(&self) {
        let mut indices = HashSet::new();
        let mut prev_index = None;
        let mut next_index = self.first_index;
        while let Some(index) = next_index {
            indices.insert(index);

            let entry = &self.entries[index];
            assert_eq!(entry.prev_index, prev_index);

            prev_index = Some(index);
            next_index = entry.next_index;
        }
        assert_eq!(self.last_index, prev_index);
        assert_eq!(indices.len(), self.entries.len());
    }

    fn add_entry(&mut self, key: K, value: V, front: bool) {
        let new_index = self.entries.len();
        self.entries.push(Entry {
            key,
            value,
            prev_index: if front { None } else { self.last_index },
            next_index: if front { self.first_index } else { None },
        });

        if front {
            if let Some(first_index) = self.first_index {
                let slot = &mut self.entries[first_index].prev_index;
                debug_assert_eq!(*slot, None);
                *slot = Some(new_index);
            } else {
                self.last_index = Some(new_index);
            }
            self.first_index = Some(new_index);
        } else {
            if let Some(last_index) = self.last_index {
                let slot = &mut self.entries[last_index].next_index;
                debug_assert_eq!(*slot, None);
                *slot = Some(new_index);
            } else {
                self.first_index = Some(new_index);
            }
            self.last_index = Some(new_index)
        }
    }

    fn remove(&mut self, index: usize) -> Entry<K, V> {
        // remove entry from its chain
        {
            let entry = &mut self.entries[index];
            let prev_index = entry.prev_index;
            let next_index = entry.next_index;
            if let Some(prev_index) = prev_index {
                self.entries[prev_index].next_index = next_index
            } else {
                self.first_index = next_index;
            }
            if let Some(next_index) = next_index {
                self.entries[next_index].prev_index = prev_index;
            } else {
                self.last_index = prev_index;
            }
        }

        // remove entry from the vec
        let mut entry = self.entries.swap_remove(index);
        entry.prev_index = None;
        entry.next_index = None;

        // fix the moved (previously last in the vec) element
        if let Some(moved_entry) = self.entries.get(index) {
            let prev_index = moved_entry.prev_index;
            let next_index = moved_entry.next_index;
            if let Some(prev_index) = prev_index {
                self.entries[prev_index].next_index = Some(index);
            } else {
                self.first_index = Some(index);
            }
            if let Some(next_index) = next_index {
                self.entries[next_index].prev_index = Some(index);
            } else {
                self.last_index = Some(index);
            }
        }

        entry
    }

    fn move_to_front(&mut self, index: usize) {
        let entry = self.remove(index);
        self.add_entry(entry.key, entry.value, true);
    }

    fn retain(&mut self, mut f: impl FnMut(&K, &V) -> RetainAction) {
        let mut next_index = self.first_index;

        while let Some(index) = next_index {
            let entry = &self.entries[index];
            let action = f(&entry.key, &entry.value);

            next_index = match action {
                RetainAction::ContinueKeep => {
                    entry.next_index
                }
                RetainAction::ContinueRemove => {
                    let normal_next = entry.next_index;
                    self.remove(index);
                    if normal_next == Some(self.entries.len()) {
                        Some(index)
                    } else {
                        normal_next
                    }
                }
                RetainAction::BreakMoveToFront => {
                    self.move_to_front(index);
                    break
                }
            };
        }

        if cfg!(debug_assertions) {
            self.assert_valid();
        }
    }

    /// Mutate entries in-place.
    /// The function `f` should preserve dominance between keys, 
    /// otherwise the frontier guarantees might not be valid any more.
    pub fn mutate_preserve_dominance(&mut self, mut f: impl FnMut(&mut K, &mut V)) {
        for entry in &mut self.entries {
            f(&mut entry.key, &mut entry.value);
        }
    }
}

impl<K: Dominance, V> Frontier<K, V>  {
    #[inline(never)]
    pub fn would_add(&self, new: &K, aux: &K::Aux) -> bool {
        for (old, _) in self.iter_lru() {
            match new.dominance(old, aux) {
                // new is better, we should add it
                DomDir::Better => return true,
                // old is better or equal, new is useless
                // TODO move to front here as well?
                DomDir::Worse | DomDir::Equal => return false,
                // both are incomparable, keep looking
                DomDir::Incomparable => {}
            }
        }

        // no old was better or equal, we should add new
        true
    }
}

#[derive(Debug, Copy, Clone)]
enum RetainAction {
    ContinueKeep,
    ContinueRemove,
    BreakMoveToFront,
}

impl<K: Dominance + Clone, V> Frontier<K, V>  {
    // TODO cow or clarify in name that we might clone
    // TODO clean this up, this signature is annoying, maybe change to (with #must_use)
    //    if let Some(add) = self.prepare_add(new, aux) { add.finish(value) }
    #[inline(never)]
    pub fn add(&mut self, new: &K, aux: &K::Aux, new_value: impl FnOnce() -> V) -> bool {
        if cfg!(debug_assertions) {
            self.assert_valid();
        }

        self.count_add_try += 1;

        let mut new_better_than_any_old = false;
        let mut any_old_better_than_new = false;

        let mut samples = 0;
        let mut removed = 0;

        self.retain(|old, _| {
            samples += 1;
            match new.dominance(old, aux) {
                DomDir::Better => {
                    // new is better, drop old
                    new_better_than_any_old = true;
                    removed += 1;
                    RetainAction::ContinueRemove
                }
                DomDir::Worse | DomDir::Equal => {
                    // old is better or equal, new is useless
                    any_old_better_than_new = true;
                    RetainAction::BreakMoveToFront
                }
                DomDir::Incomparable => {
                    // both are incomparable, keep looking
                    RetainAction::ContinueKeep
                }
            }
        });
        
        self.count_add_removed += removed;

        // if any_old_better_than_new {
        //     println!("hit: {}", samples as f32 / self.len() as f32);
        // }
        self.dominance_calculations += samples;

        assert!(!(new_better_than_any_old && any_old_better_than_new), "Transitive property of PartialOrd was not respected");

        if !any_old_better_than_new {
            // no old was better or equal, we should add new
            self.add_entry(new.clone(), new_value(), false);
            self.count_add_success += 1;
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct DomBuilder<S> {
    pub any_better: bool,
    pub any_worse: bool,

    self_value: S,
    other_value: S,
}

impl<S: Copy> DomBuilder<S> {
    pub fn new(self_value: S, other_value: S) -> Self {
        DomBuilder {
            any_better: false,
            any_worse: false,
            self_value,
            other_value,
        }
    }
    
    pub fn minimize_custom<T: PartialOrd>(&mut self, self_value: T, other_value: T) {
        match self_value.partial_cmp(&other_value) {
            Some(std::cmp::Ordering::Less) => self.any_better = true,
            Some(std::cmp::Ordering::Greater) => self.any_worse = true,
            Some(std::cmp::Ordering::Equal) => {}
            // TODO require ord? but then comparing floats becomes annoying...
            None => panic!("Cannot compare values"),
        }
    }

    pub fn minimize<T: PartialOrd>(&mut self, f: impl Fn(S) -> T) {
        self.minimize_custom(f(self.self_value), f(self.other_value));
    }

    pub fn maximize<T: PartialOrd>(&mut self, f: impl Fn(S) -> T) {
        self.minimize(|s| std::cmp::Reverse(f(s)));
    }

    pub fn finish(&self) -> DomDir {
        match (self.any_better, self.any_worse) {
            (true, true) => DomDir::Incomparable,
            (true, false) => DomDir::Better,
            (false, true) => DomDir::Worse,
            (false, false) => DomDir::Equal,
        }
    }
}

#[macro_export]
macro_rules! dom_early_check {
    ($dom: expr) => {{
        let dom: &DomBuilder<_> = &$dom;
        if dom.any_better && dom.any_worse {
            return DomDir::Incomparable;
        }
    }}
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    use crate::core::frontier::{Frontier, RetainAction};

    // TODO fuzz testing
    #[test]
    fn basic() {
        let mut frontier = Frontier::empty();

        frontier.add_entry(0, 0, true);
        frontier.add_entry(1, 1, true);
        frontier.add_entry(2, 2, true);
        frontier.add_entry(3, 3, true);
        frontier.add_entry(4, 4, true);
        // frontier.remove(2);
        frontier.move_to_front(2);
        frontier.remove(2);
        frontier.add_entry(5, 5, true);

        println!("{:?}", frontier.iter_lru().collect_vec());
    }

    #[test]
    fn fuzz_test() {
        let mut frontier = Frontier::empty();
        let mut rng = SmallRng::seed_from_u64(0);

        let max_size = 10;

        for i in 0..10_000 {
            if frontier.len() >= max_size {
                // println!("removing");
                frontier.remove(rng.gen_range(0..frontier.len()));
                continue;
            }
            if frontier.len() == 0 {
                // println!("adding");
                frontier.add_entry(i, i, rng.gen());
                continue;
            }

            match rng.gen_range(0..4) {
                0 => {
                    // println!("rand removing");
                    frontier.remove(rng.gen_range(0..frontier.len()));
                }
                1 => {
                    // println!("rand adding");
                    frontier.add_entry(i, i, rng.gen());
                }
                2 => {
                    // println!("rand changing");
                    frontier.move_to_front(rng.gen_range(0..frontier.len()));
                }
                3 => {
                    // randomly drop values
                    let len_before = frontier.len();
                    let mut any_break = false;
                    let mut visit_count = 0;

                    frontier.retain(|_, _| {
                        visit_count += 1;
                        match rng.gen_range(0..2) {
                            0 => RetainAction::ContinueRemove,
                            1 => RetainAction::ContinueKeep,
                            2 => {
                                any_break = true;
                                RetainAction::BreakMoveToFront
                            },
                            _ => unreachable!(),
                        }
                    });

                    if !any_break {
                        assert_eq!(visit_count, len_before);
                    }
                }
                _ => unreachable!(),
            }
        }

        println!("{}", frontier.len());
    }
}