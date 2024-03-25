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

pub struct Frontier<K, V> {
    values: Vec<(K, V)>,
}

impl<K, V> Frontier<K, V> {
    pub fn new() -> Self {
        Self { values: vec![] }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.values.iter()
    }
}

impl<K: Dominance, V> Frontier<K, V>  {
    pub fn would_add(&self, new: &K, aux: &K::Aux) -> bool {
        for (old, _) in &self.values {
            match new.dominance(old, aux) {
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

impl<K: Dominance + Clone, V> Frontier<K, V>  {
    // TODO cow or clarify in name that we might clone
    // TODO clean this up, this signature is annoying, maybe change to (with #must_use)
    //    if let Some(add) = self.prepare_add(new, aux) { add.finish(value) }
    pub fn add(&mut self, new: &K, aux: &K::Aux, new_value: impl FnOnce() -> V) -> bool {
        let mut i = 0;
        let mut dropped_any_old = false;

        while i < self.values.len() {
            let (old, _) = &self.values[i];
            match new.dominance(old, aux) {
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
        self.values.push((new.clone(), new_value()));
        return true;
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

    pub fn minimize<T: PartialOrd>(&mut self, f: impl Fn(S) -> T) {
        match f(self.self_value).partial_cmp(&f(self.other_value)) {
            Some(std::cmp::Ordering::Less) => self.any_better = true,
            Some(std::cmp::Ordering::Greater) => self.any_worse = true,
            Some(std::cmp::Ordering::Equal) => {}
            // TODO require ord? but then comparing floats becomes annoying...
            None => panic!("Cannot compare values"),
        }
    }

    pub fn maximize<T: PartialOrd>(&mut self, f: impl Fn(S) -> T) {
        self.minimize(|s| std::cmp::Reverse(f(s)));
    }

    pub fn finish(&self) -> DomDir {
        match (self.any_better, self.any_worse) {
            (true, true) => panic!("This should have been caught by dom_early_check!"),
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
