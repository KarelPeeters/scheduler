// we used signed for easier math and extra sentinels

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Time(pub i64);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Energy(pub i64);

macro_rules! impl_arith {
    ($ty:ident) => {
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
