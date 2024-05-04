use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub trait TypedIndex: Copy + Clone + Eq + PartialEq + Ord + PartialOrd + Hash {
    fn to_index(self) -> usize;
    fn from_index(inner: usize) -> Self;
}

// TODO remove ord? it's a bit too implicit
#[macro_export]
macro_rules! define_typed_index {
    ($name:ident) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(usize);
        impl crate::core::wrapper::TypedIndex for $name {
            fn to_index(self) -> usize {
                self.0
            }
            fn from_index(inner: usize) -> Self {
                $name(inner)
            }
        }
    };
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TypedVec<K: TypedIndex, V> {
    pub vec: Vec<V>,
    phantom: PhantomData<K>,
}

impl<K: TypedIndex, V: Debug> Debug for TypedVec<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.vec.fmt(f)
    }
}

impl<K: TypedIndex, V> TypedVec<K, V> {
    pub fn new() -> Self {
        Self::wrap(vec![])
    }

    pub fn full_like<W>(value: V, other: &TypedVec<K, W>) -> Self where V: Clone {
        Self::wrap(vec![value; other.vec.len()])
    }

    pub fn wrap(vec: Vec<V>) -> Self {
        TypedVec {
            vec,
            phantom: PhantomData,
        }
    }

    pub fn fill(&mut self, value: V) where V: Clone {
        self.vec.fill(value)
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(K, &V)> {
        (&self).into_iter()
    }

    pub fn keys(&self) -> impl Iterator<Item=K> {
        (0..self.vec.len()).map(K::from_index)
    }

    pub fn values(&self) -> impl Iterator<Item=&V> {
        self.vec.iter()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item=&mut V> {
        self.vec.iter_mut()
    }

    pub fn has_key(&self, key: K) -> bool {
        key.to_index() < self.vec.len()
    }

    pub fn push(&mut self, value: V) -> K {
        let key = K::from_index(self.vec.len());
        self.vec.push(value);
        key
    }
}

impl<K: TypedIndex, V> Index<K> for TypedVec<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        &self.vec[index.to_index()]
    }
}

impl<K: TypedIndex, V> IndexMut<K> for TypedVec<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        &mut self.vec[index.to_index()]
    }
}

impl<'a, K: TypedIndex, V> IntoIterator for &'a TypedVec<K, V> {
    type Item = (K, &'a V);
    type IntoIter = std::iter::Map<std::iter::Enumerate<std::slice::Iter<'a, V>>, fn((usize, &V)) -> (K, &V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter().enumerate().map(|(i, v)| (K::from_index(i), v))
    }
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Time(pub i64);

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Energy(pub i64);

macro_rules! impl_arith {
    ($ty:ident) => {
        impl $ty {
            pub fn ceil_div(self, div: i64) -> Self {
                assert!(self.0 >= 0);
                assert!(div > 0);
                let added = self.0.checked_add(div - 1).unwrap();
                $ty(added / div)
            }
        }
        impl std::ops::Add for $ty {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                $ty(self.0 + rhs.0)
            }
        }
        impl std::ops::Sub for $ty {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                $ty(self.0 - rhs.0)
            }
        }
        impl std::ops::AddAssign for $ty {
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0 + rhs.0;
            }
        }
        impl std::ops::SubAssign for $ty {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0 + rhs.0;
            }
        }
        impl std::ops::Mul::<u64> for $ty {
            type Output = Self;
            fn mul(self, rhs: u64) -> Self::Output {
                $ty(self.0 * rhs as i64)
            }
        }
        impl std::iter::Sum for $ty {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($ty(0), |acc, x| acc + x)
            }
        }
    };
}

impl_arith!(Time);
impl_arith!(Energy);
