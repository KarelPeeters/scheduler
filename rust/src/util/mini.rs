use std::cmp::min;

pub fn min_option<T: Ord + Copy>(left: Option<T>, right: Option<T>) -> Option<T> {
    match (left, right) {
        (None, None) => None,
        (None, Some(single)) | (Some(single), None) => Some(single),
        (Some(left), Some(right)) => Some(min(left, right)),
    }
}
