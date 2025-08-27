use crate::{ToGF2, GF2};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Poly {
    terms: Vec<usize>
}

impl Poly {
    pub fn new(coeffs: impl IntoIterator<Item=impl ToGF2>) -> Self {
        Poly {
            terms: coeffs.into_iter().enumerate()
                .flat_map(|(i, c)| (c.to_gf2() != GF2::ZERO).then_some(i))
                .collect()
        }
    }

    pub fn from_terms(terms: impl IntoIterator<Item=usize>) -> Self {
        let mut dterms = terms.into_iter().collect::<Vec<_>>();
        dterms.sort();
        let mut terms = Vec::new();
        for t in dterms {
            if terms.is_empty() {
                terms.push(t);
            } else if terms[terms.len() - 1] == t {
                terms.pop();
            } else {
                terms.push(t);
            }
        }
        Poly { terms }
    }
    
    pub const ZERO: Self = Poly { terms: Vec::new() };

    pub fn one() -> Self {
        Self::monom(0)
    }

    pub fn monom(i: usize) -> Self {
        Poly { terms: vec![i] }
    }

    pub fn terms(&self) -> &[usize] {
        &self.terms
    }

    pub fn coeffs(&self) -> impl Iterator<Item=GF2> {
        struct CoeffIterator<'r> {
            terms: &'r [usize],
            idx: usize,
            term: usize
        }

        impl<'r> Iterator for CoeffIterator<'r> {
            type Item = GF2;
            
            fn next(&mut self) -> Option<GF2> {
                if self.idx >= self.terms.len() {
                    None
                } else if self.terms[self.idx] == self.term {
                    self.idx += 1;
                    self.term += 1;
                    Some(GF2::ONE)
                } else {
                    self.term += 1;
                    Some(GF2::ZERO)
                }
            }
        }

        CoeffIterator { terms: &self.terms, idx: 0, term: 0 }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.terms.len() == 1 && self.terms[0] == 0
    }

    pub fn degree(&self) -> usize {
        self.terms.last().copied().unwrap_or(0)
    }

    pub fn eval(&self, x: GF2) -> GF2 {
        if self.is_zero() {
            GF2::ZERO
        } else if x == GF2::ZERO {
            GF2::new(self.terms[0] == 0)
        } else {
            GF2::new(self.terms.len() % 2 != 0)
        }
    }
}

impl std::fmt::Debug for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(&t) = self.terms.first() {
            match t {
                0 => write!(f, "1")?,
                1 => write!(f, "x")?,
                _ => write!(f, "x^{}", t)?
            }
        } else {
            write!(f, "0")?
        };

        for &t in &self.terms[1..] {
            match t {
                0 => unreachable!(),
                1 => write!(f, " + x")?,
                _ => write!(f, " + x^{}", t)?
            }
        }

        Ok(())
    }
}

impl std::fmt::Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn symmetric_diff_sorted<C: Ord>(a: impl Iterator<Item=C>, b: impl Iterator<Item=C>) -> impl Iterator<Item=C> {
    struct SymmetricDifferenceSorted<A: Iterator, B: Iterator> {
        a: std::iter::Peekable<A>,
        b: std::iter::Peekable<B>
    }
    
    impl<A: Iterator<Item=C>, B: Iterator<Item=C>, C: Ord> Iterator for SymmetricDifferenceSorted<A, B> {
        type Item = C;
    
        fn next(&mut self) -> Option<C> {
            use std::cmp::Ordering::{Less, Greater, Equal};
            loop {
                return match (self.a.peek(), self.b.peek()) {
                    (Some(aa), Some(bb)) => match aa.cmp(bb) {
                        Less => self.a.next(),
                        Greater => self.b.next(),
                        Equal => { self.a.next(); self.b.next(); continue }
                    },
                    (Some(_), None) => self.a.next(),
                    (None, Some(_)) => self.b.next(),
                    (None, None) => None
                }
            }
        }
    }

    SymmetricDifferenceSorted { a: a.peekable(), b: b.peekable() }
}

impl std::ops::Add<&Poly> for &Poly {
    type Output = Poly;

    fn add(self, rhs: &Poly) -> Poly {
        Poly { terms: symmetric_diff_sorted(self.terms.iter(),rhs.terms.iter()).copied().collect() }
    }
}

impl std::ops::Add<Poly> for &Poly {
    type Output = Poly;
    fn add(self, rhs: Poly) -> Self::Output { self + &rhs }
}

impl std::ops::Add<&Poly> for Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Self::Output { &self + rhs }
}

impl std::ops::Add<Poly> for Poly {
    type Output = Poly;
    fn add(self, rhs: Poly) -> Self::Output { &self + &rhs }
}

impl std::ops::AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        *self = &*self + rhs;
    }
}

impl std::ops::AddAssign<Poly> for Poly {
    fn add_assign(&mut self, rhs: Poly) { *self += &rhs }
}

impl std::ops::Sub<&Poly> for &Poly {
    type Output = Poly;

    fn sub(self, rhs: &Poly) -> Self::Output {
        self + rhs
    }
}

impl std::ops::Sub<Poly> for &Poly {
    type Output = Poly;
    fn sub(self, rhs: Poly) -> Self::Output { self - &rhs }
}

impl std::ops::Sub<&Poly> for Poly {
    type Output = Poly;
    fn sub(self, rhs: &Poly) -> Self::Output { &self - rhs }
}

impl std::ops::Sub<Poly> for Poly {
    type Output = Poly;
    fn sub(self, rhs: Poly) -> Self::Output { &self - &rhs }
}

impl std::ops::SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Poly) {
        *self += rhs;
    }
}

impl std::ops::SubAssign<Poly> for Poly {
    fn sub_assign(&mut self, rhs: Poly) { *self -= &rhs; }
}

impl std::ops::Mul<&Poly> for &Poly {
    type Output = Poly;

    fn mul(self, rhs: &Poly) -> Self::Output {
        let (a, b) = if self.terms.len() <= rhs.terms.len() {
            (&self.terms, &rhs.terms)
        } else {
            (&rhs.terms, &self.terms)
        };

        let mut terms = Vec::new();
        let mut buf = Vec::new();

        for &aa in a {
            buf.extend(symmetric_diff_sorted(b.iter().map(|&bb| aa + bb), terms.iter().copied()));
            std::mem::swap(&mut terms, &mut buf);
            buf.clear();
        }
        
        Poly { terms }
    }
}

impl std::ops::Mul<Poly> for &Poly {
    type Output = Poly;
    fn mul(self, rhs: Poly) -> Self::Output { self * &rhs }
}

impl std::ops::Mul<&Poly> for Poly {
    type Output = Poly;
    fn mul(self, rhs: &Poly) -> Self::Output { &self * rhs }
}

impl std::ops::Mul<Poly> for Poly {
    type Output = Poly;
    fn mul(self, rhs: Poly) -> Self::Output { &self * &rhs }
}

impl std::ops::MulAssign<&Poly> for Poly {
    fn mul_assign(&mut self, rhs: &Poly) {
        *self = &*self * rhs;
    }
}

impl std::ops::MulAssign<Poly> for Poly {
    fn mul_assign(&mut self, rhs: Poly) { *self *= &rhs }
}

impl std::ops::Neg for Poly {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

impl Poly {
    pub fn square(&mut self) {
        self.pow2k(1);
    }

    pub fn pow2k(&mut self, k: usize) {
        self.terms.iter_mut().for_each(|t| *t <<= k)
    }

    pub fn pow(&self, n: usize) -> Poly {
        match n {
            0 => Poly::one(),
            1 => self.clone(),
            n if n.is_power_of_two() => {
                let mut x = self.clone();
                x.pow2k(n.trailing_zeros() as usize);
                x
            },
            mut n => {
                let mut x = self.clone();
                let mut y = Poly::one();

                let k = n.trailing_zeros() as usize;
                x.pow2k(k);
                n >>= k;
                loop {
                    y *= &x;
                    n >>= 1;

                    if n == 0 {
                        return y
                    } else {
                        let k = n.trailing_zeros() as usize;
                        x.pow2k(k + 1);
                        n >>= k;
                    }
                }
            }
        }
    }

    pub fn quot_rem(&self, d: &Poly) -> (Poly, Poly) {
        assert!(!d.is_zero());

        let d = &d.terms;
        let mut q = Vec::new();
        let mut r = self.terms.clone();
        let mut buf = Vec::new();

        while r.len() > 0 && r[r.len() - 1] >= d[d.len() - 1] {
            let t = r[r.len() - 1] - d[d.len() - 1];
            q.push(t);
            buf.extend(symmetric_diff_sorted(r.iter().copied(), d.iter().map(|&dd| dd + t)));
            std::mem::swap(&mut r, &mut buf);
            buf.clear();
        }

        q.reverse();
        (Poly { terms: q }, Poly { terms: r })
    }

    pub fn quot(&self, d: &Poly) -> Poly {
        self.quot_rem(d).0
    }

    pub fn rem(&self, d: &Poly) -> Poly {
        self.quot_rem(d).1
    }

    pub fn gcd(&self, rhs: &Poly) -> Poly {
        if self.is_one() || rhs.is_one() {
            return Poly::one()
        } else if self == rhs {
            return self.clone()
        }

        let (mut a, mut b) = if self.degree() >= rhs.degree() {
            (self.clone(), rhs.clone())
        } else {
            (rhs.clone(), self.clone())
        };

        while !(b.is_zero() || b.is_one() || a.is_one()) {
            a = a.rem(&b);
            std::mem::swap(&mut a, &mut b);
        }

        if a.is_one() || b.is_one() {
            Poly::one()
        } else {
            a
        }
    }

    pub fn extended_gcd(&self, rhs: &Poly) -> (Poly, Poly, Poly) {
        let (mut r0, mut r1) = (self.clone(), rhs.clone());
        let (mut s0, mut s1) = (Poly::one(), Poly::ZERO);
        let (mut t0, mut t1) = (Poly::ZERO, Poly::one());

        while !r1.is_zero() {
            let (q, r) = r0.quot_rem(&r1);
            r0 = r;
            s0 += &q * &s1;
            t0 += &q * &t1;
            std::mem::swap(&mut r0, &mut r1);
            std::mem::swap(&mut s0, &mut s1);
            std::mem::swap(&mut t0, &mut t1);
        }

        (r0, s0, t0)
    }

    pub fn lcm(&self, rhs: &Poly) -> Poly {
        let d = self.gcd(rhs);
        self * rhs.quot(&d)
    }

    pub fn diff(&self) -> Poly {
        Poly { terms: self.terms.iter().flat_map(|&t| (t % 2 != 0).then_some(t - 1)).collect() }
    }
}

