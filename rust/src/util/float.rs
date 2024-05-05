use std::cmp::min;

pub trait IterFloatExt: Iterator {
    fn min_f64(self) -> Option<Self::Item>;
    fn max_f64(self) -> Option<Self::Item>;
}

impl<I: Iterator<Item = f64>> IterFloatExt for I {
    fn min_f64(mut self) -> Option<Self::Item> {
        let mut min = self.next()?;
        if min.is_nan() {
            return Some(min);
        }

        for x in self {
            if x.is_nan() {
                return Some(x);
            }
            if x < min {
                min = x;
            }
        }
        Some(min)
    }

    fn max_f64(self) -> Option<Self::Item> {
        self.map(|x| -x).min_f64().map(|x| -x)
    }
}

pub fn min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    f64::min(a, b)
}

pub fn max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    f64::max(a, b)
}

pub fn min_option<T: Ord>(a: Option<T>, b: Option<T>) -> Option<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(min(a, b)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}