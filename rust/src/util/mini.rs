pub trait IterFloatExt: Iterator {
    fn min_float(self) -> Option<Self::Item>;
}

impl<I: Iterator<Item = f64>> IterFloatExt for I {
    fn min_float(mut self) -> Option<Self::Item> {
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