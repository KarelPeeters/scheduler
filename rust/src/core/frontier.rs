use std::cmp::Ordering;

struct Frontier<T> {
    values: Vec<T>,
}

impl<T> Frontier<T> {
    pub fn new() -> Self {
        Self { values: vec![] }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

impl<T: PartialOrd> Frontier<T> {
    pub fn add(&mut self, new: T) -> bool {
        let mut i = 0;
        let mut dropped_any_old = false;

        while i < self.values.len() {
            let old = &self.values[i];
            match old.partial_cmp(&new) {
                None => {
                    // both elements are incomparable, keep looking
                    i += 1;
                }
                Some(Ordering::Less | Ordering::Equal) => {
                    // old element is better or equal, the new one is useless
                    assert!(!dropped_any_old, "Transitive property of PartialOrd was not respected");
                    return false;
                }
                Some(Ordering::Greater) => {
                    // new is better, drop the old one (and don't increment i)
                    dropped_any_old = true;
                    self.values.swap_remove(i);
                }
            }
        }

        // no old value dominated new, add it
        self.values.push(new);
        return true;
    }
}