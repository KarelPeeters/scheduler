use std::collections::HashMap;

// TODO better name for this type and the trait
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DomDir {
    Better,
    Worse,
    Equal,
    Incomparable,
}

pub trait DominanceItem {
    type Aux;
    fn dominance(&self, other: &Self, aux: &Self::Aux) -> DomDir;
}

pub struct TupleDominanceItem {
    // TODO use sorted vec instead? probably faster for low index counts
    values: HashMap<usize, f64>,
    default_value: f64,
}

#[derive(Debug)]
pub struct DomBuilder<S> {
    pub any_better: bool,
    pub any_worse: bool,

    self_value: S,
    other_value: S,
}

// TODO delete this now that we have tuples?
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

impl<T: TupleDominanceItem> DominanceItem for T {
    type Aux = T::Aux;

    fn dominance(&self, other: &Self, aux: &Self::Aux) -> DomDir {
        let len = self.dominance_tuple_len(aux);
        assert_eq!(len, other.dominance_tuple_len(aux));
        
        let mut dom = DomBuilder::new(self, other);
        for i in 0..len {
            dom.minimize(|s| s.get_axis_minimize(aux, i));
            dom_early_check!(dom);
        }
        dom.finish()
    }
} 
