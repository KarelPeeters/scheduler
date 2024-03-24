// TODO better name for this type and the trait
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DomDir {
    Better,
    Worse,
    Equal,
    Incomparable,
}

pub trait Dominance {
    fn dominance(&self, other: &Self) -> DomDir;
}

pub struct Frontier<T> {
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

impl<T: Dominance> Frontier<T> {
    pub fn would_add(&self, new: T) -> bool {
        for old in &self.values {
            match new.dominance(old) {
                // new is better, we should add it
                DomDir::Better => return true,
                // old is better or equal, new is useless
                DomDir::Worse | DomDir::Equal => return false,
                // both are incomparable, keep looking
                DomDir::Incomparable => {}
            }
        }

        // no old was better or equal, we should add new
        return true;
    }
}

impl<T: Dominance + Clone> Frontier<T> {
    pub fn add(&mut self, new: &T) -> bool {
        let mut i = 0;
        let mut dropped_any_old = false;

        while i < self.values.len() {
            let old = &self.values[i];
            match new.dominance(old) {
                DomDir::Better => {
                    // new is better, drop the old one (and don't increment index)
                    dropped_any_old = true;
                    self.values.swap_remove(i);
                }
                DomDir::Worse | DomDir::Equal => {
                    // old is better or equal, new is useless
                    assert!(!dropped_any_old, "Transitive property of PartialOrd was not respected");
                    return false;
                }
                DomDir::Incomparable => {
                    // both are incomparable, keep looking
                    i += 1;
                }
            }
        }

        // no old was better or equal, we should add new
        self.values.push(new.clone());
        return true;
    }
}
