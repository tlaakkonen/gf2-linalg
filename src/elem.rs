#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GF2(bool);

impl GF2 {
    pub const ZERO: GF2 = GF2(false);
    pub const ONE: GF2 = GF2(true);

    pub fn new(value: bool) -> GF2 {
        GF2(value)
    }
}

impl std::fmt::Debug for GF2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 {
            write!(f, "1")
        } else {
            write!(f, "0")
        }
    }
}

impl std::fmt::Display for GF2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<bool> for GF2 {
    fn from(value: bool) -> Self {
        GF2(value)
    }
}

impl From<GF2> for bool {
    fn from(value: GF2) -> Self {
        value.0
    }
}

pub trait ToGF2 {
    fn to_gf2(self) -> GF2;
}

impl ToGF2 for GF2 {
    fn to_gf2(self) -> GF2 {
        self
    }
}

impl ToGF2 for bool {
    fn to_gf2(self) -> GF2 {
        GF2(self)
    }
}

macro_rules! impl_to_gf2_numeric {
    ($($t:ty),*) => {$(impl ToGF2 for $t {
        fn to_gf2(self) -> GF2 {
            GF2((self % 2) != 0)
        }
    })*};
}

impl_to_gf2_numeric!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl<T: ToGF2> std::ops::Add<T> for GF2 {
    type Output = GF2;

    fn add(self, rhs: T) -> Self::Output {
        GF2(self.0 ^ rhs.to_gf2().0)
    }
}

impl<T: ToGF2> std::ops::AddAssign<T> for GF2 {
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs
    }
}

impl<T: ToGF2> std::ops::Sub<T> for GF2 {
    type Output = GF2;

    fn sub(self, rhs: T) -> Self::Output {
        self + rhs
    }
}

impl<T: ToGF2> std::ops::SubAssign<T> for GF2 {
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs
    }
}


impl<T: ToGF2> std::ops::Mul<T> for GF2 {
    type Output = GF2;

    fn mul(self, rhs: T) -> Self::Output {
        GF2(self.0 & rhs.to_gf2().0)
    }
}

impl<T: ToGF2> std::ops::MulAssign<T> for GF2 {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs
    }
}


impl std::ops::Neg for GF2 {
    type Output = GF2;

    fn neg(self) -> Self::Output {
        self
    }
}

impl<T: ToGF2> std::ops::Div<T> for GF2 {
    type Output = GF2;

    fn div(self, v: T) -> Self::Output {
        assert_ne!(v.to_gf2(), GF2::ZERO);
        self
    }
}


impl<T: ToGF2> std::ops::DivAssign<T> for GF2 {
    fn div_assign(&mut self, v: T) {
        assert_ne!(v.to_gf2(), GF2::ZERO);
    }
}

#[cfg(feature = "rand")]
impl rand::distr::Distribution<GF2> for rand::distr::StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> GF2 {
        rng.random::<bool>().into()
    }
}

impl std::iter::Sum for GF2 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(GF2::ZERO, std::ops::Add::add)
    }
}

impl std::iter::Product for GF2 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(GF2::ONE, std::ops::Mul::mul)
    }
}

