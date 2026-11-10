use std::collections::HashMap;

use crate::{GF2, InvalidInputError, Matrix, ToGF2, error::validate_input};

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

    #[cfg(feature = "rand")]
    pub fn random(rng: &mut impl rand::Rng, max_degree: usize) -> Poly {
        use rand::RngExt;
        Poly::new((0..max_degree+1).map(|_| rng.random::<GF2>()))
    }

    pub fn coeff(&self, term: usize) -> GF2 {
        GF2::new(self.terms.binary_search(&term).is_ok())
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
            return write!(f, "0")
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
    pub fn reciprocal(&self) -> Poly {
        let n = self.degree();
        let mut terms: Vec<_> = self.terms.iter().map(|&t| n - t).collect();
        terms.reverse();
        Poly { terms }
    }

    /// For irreducible f, compute the trace of self in F2\[\[x\]\]/(f) over F2
    pub fn trace_mod_checked(&self, f: &Poly) -> Result<GF2, InvalidInputError> {
        let mut g = self.rem_checked(f)?;
        let mut tr = g.eval(GF2::ZERO);
        for _ in 0..f.degree().saturating_sub(1) {
            g.square();
            g = g.rem_checked(f)?;
            tr += g.eval(GF2::ZERO);
        }
        Ok(tr)
    }

    /// For irreducible f, compute the trace of self in F2\[\[x\]\]/(f) over F2
    pub fn trace_mod(&self, f: &Poly) -> GF2 {
        self.trace_mod_checked(f).unwrap()
    }

    pub fn square(&mut self) {
        self.pow2k(1);
    }

    pub fn square_root(&self) -> Option<Poly> {
        Some(Poly {
            terms: self.terms.iter().map(|&t| (t % 2 == 0).then_some(t / 2)).collect::<Option<Vec<_>>>()?
        })
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

    pub fn pow_mod_checked(&self, mut n: usize, m: &Poly) -> Result<Poly, InvalidInputError> {
        validate_input!(!m.is_zero(), "modulus should be non-zero" in pow_mod);

        let mut x = self.clone();
        let mut y = Poly::one();
        while n != 0 {
            if n % 2 == 1 {
                y *= &x;
                y = y.rem_checked(m)?;
            }
            x.square();
            x = x.rem_checked(m)?;
            n >>= 1;            
        }
        Ok(y)
    }

    pub fn pow_mod(&self, n: usize, m: &Poly) -> Poly {
        self.pow_mod_checked(n, m).unwrap()
    }

    pub fn pow_mod_signed_checked(&self, n: isize, m: &Poly) -> Result<Option<Poly>, InvalidInputError> {
        Ok(if n < 0 {
            let inv = self.inv_mod_checked(m)?;
            if let Some(inv) = inv {
                Some(inv.pow_mod_checked((-n) as usize, m)?)
            } else {
                None
            }
        } else {
            Some(self.pow_mod_checked(n as usize, m)?)
        })
    }

    pub fn pow_mod_signed(&self, n: isize, m: &Poly) -> Option<Poly> {
        self.pow_mod_signed_checked(n, m).unwrap()
    }

    pub fn quot_rem_checked(&self, d: &Poly) -> Result<(Poly, Poly), InvalidInputError> {
        validate_input!(!d.is_zero(), "divisor should be non-zero" in quot_rem);

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
        Ok((Poly { terms: q }, Poly { terms: r }))
    }

    pub fn quot_rem(&self, d: &Poly) -> (Poly, Poly) {
        self.quot_rem_checked(d).unwrap()
    }

    pub fn quot_checked(&self, d: &Poly) -> Result<Poly, InvalidInputError> {
        Ok(self.quot_rem_checked(d)?.0)
    }

    pub fn rem_checked(&self, d: &Poly) -> Result<Poly, InvalidInputError> {
        Ok(self.quot_rem_checked(d)?.1)
    }

    pub fn quot(&self, d: &Poly) -> Poly {
        self.quot_checked(d).unwrap()
    }

    pub fn rem(&self, d: &Poly) -> Poly {
        self.rem_checked(d).unwrap()
    }

    pub fn gcd(&self, rhs: &Poly) -> Poly {
        if self.is_one() || rhs.is_one() {
            return Poly::one()
        } else if self == rhs {
            return self.clone()
        } else if self.is_zero() {
            return rhs.clone()
        } else if rhs.is_zero() {
            return self.clone()
        }

        let (mut a, mut b) = if self.degree() >= rhs.degree() {
            (self.clone(), rhs.clone())
        } else {
            (rhs.clone(), self.clone())
        };

        while !b.is_zero() {
            a = a.rem(&b);
            std::mem::swap(&mut a, &mut b);
        }

        a
    }

    pub fn extended_gcd(&self, rhs: &Poly) -> (Poly, Poly, Poly) {
        if self.is_zero() {
            return (rhs.clone(), Poly::one(), Poly::one())
        } else if rhs.is_zero() {
            return (self.clone(), Poly::one(), Poly::one())
        }

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

    pub fn inv_mod_checked(&self, f: &Poly) -> Result<Option<Poly>, InvalidInputError> {
        validate_input!(!f.is_zero(), "modulus should not be zero" in inv_mod);

        if self.is_zero() {
            return Ok(None)
        }

        let (r, s, _) = self.extended_gcd(f);
        Ok(r.is_one().then_some(s))
    }

    pub fn inv_mod(&self, f: &Poly) -> Option<Poly> {
        self.inv_mod_checked(f).unwrap()
    }

    pub fn lcm(&self, rhs: &Poly) -> Poly {
        if self.is_zero() || rhs.is_zero() {
            return Poly::ZERO
        }

        let d = self.gcd(rhs);
        self * rhs.quot(&d)
    }

    pub fn diff(&self) -> Poly {
        Poly { terms: self.terms.iter().flat_map(|&t| (t % 2 != 0).then(|| t - 1)).collect() }
    }

    pub fn is_square_free(&self) -> bool {
        self.is_zero() || self.is_one() || self.gcd(&self.diff()).is_one()
    }

    pub fn square_free_factors(&self) -> Vec<(Poly, usize)> {
        if self.is_zero() || self.is_one() {
            return vec![(self.clone(), 1)]
        }

        if let Some(sqrt) = self.square_root() {
            let mut sff = sqrt.square_free_factors();
            sff.iter_mut().for_each(|(_, p)| *p *= 2);
            return sff;
        }

        let g = self.gcd(&self.diff());
        if g.is_one() {
            return vec![(self.clone(), 1)]
        }

        let sff_a = g.square_free_factors();
        let sff_b = self.quot(&g).square_free_factors();
        let mut output = Vec::new();
        let mut idx_a = 0;
        let mut idx_b = 0;
        while idx_a < sff_a.len() && idx_b < sff_b.len() {
            if sff_a[idx_a].1 < sff_b[idx_b].1 {
                output.push(sff_a[idx_a].clone());
                idx_a += 1;
            } else if sff_a[idx_a].1 > sff_b[idx_b].1 {
                output.push(sff_b[idx_b].clone());
                idx_b += 1;
            } else {
                output.push((&sff_a[idx_a].0 * &sff_b[idx_b].0, sff_a[idx_a].1));
                idx_a += 1;
                idx_b += 1;
            }
        }
        while idx_a < sff_a.len() {
            output.push(sff_a[idx_a].clone());
            idx_a += 1;
        }
        while idx_b < sff_b.len() {
            output.push(sff_b[idx_b].clone());
            idx_b += 1;
        }
        output
    }

    pub fn is_irreducible(&self) -> bool {
        self.is_square_free() && {
            let mut mat = self.berlekamp_matrix();
            mat += Matrix::eye(mat.shape.0);
            mat.rank() == mat.shape.0 - 1
        }
    }

    pub fn berlekamp_matrix(&self) -> Matrix {
        let n = self.degree();
        if n == 0 {
            return Matrix::zeros(0, 0)
        }

        let mut mat = Matrix::zeros(n, n);
        let mut xiq = Poly::one();
        for i in 0..n {
            for (j, c) in xiq.coeffs().enumerate() {
                mat[(j, i)] = c;
            }
            xiq = (xiq * Poly::monom(2)).rem(self);
        }
        mat
    }

    fn factor_for_square_free(&self) -> Vec<Poly> {
        if self.is_one() || self.is_zero() {
            return vec![self.clone()]
        }

        let mut mat = self.berlekamp_matrix();
        mat += Matrix::eye(mat.shape.0);
        let nsp = mat.null_space();

        let mut factors = vec![self.clone()];
        for i in 0..nsp.shape.1 {
            let g = Poly::new(nsp.col(i).iter());
            if g.is_one() { continue; }

            for i in 0..factors.len() {
                let q = &mut factors[i];
                let mut h = q.gcd(&g);
                if !h.is_one() && h.degree() < q.degree() { 
                    *q = q.quot(&h);
                    let h2 = h.gcd(&(&g - Poly::one()));
                    if !h2.is_one() && h2.degree() < h.degree() {
                        h = h.quot(&h2);
                        factors.push(h);
                        factors.push(h2);
                    } else {
                        factors.push(h);
                    }
                }
                
                let q = &mut factors[i];
                let h = q.gcd(&(&g - Poly::one()));
                if !h.is_one() && h.degree() < q.degree() {
                    *q = q.quot(&h);
                    factors.push(h);
                }
            }            
        }

        factors.sort_by_key(Poly::degree);
        factors
    }

    pub fn factor_with_multiplicities(&self) -> Vec<(Poly, usize)> {
        if self.is_one() || self.is_zero() {
            return vec![(self.clone(), 1)]
        }

        let sff = self.square_free_factors();
        let mut result = HashMap::new();
        for (mut term, reps) in sff {
            if term.terms().len() > 0 && term.terms()[0] > 0 {
                *result.entry(Poly::monom(1))
                    .or_insert(0) += term.terms()[0] * reps;
                term = term.quot(&Poly::monom(term.terms()[0]));
            }

            if term.is_one() {
                continue;
            }

            for factor in term.factor_for_square_free() {
                *result.entry(factor)
                    .or_insert(0) += reps;
            }
        }
        let mut result = result.into_iter().collect::<Vec<_>>();
        result.sort_by_key(|(p, m)| (p.degree(), *m));
        result
    }

    pub fn factor(&self) -> Vec<Poly> {
        let facts = self.factor_with_multiplicities();
        let mut out = Vec::new();
        for (fact, reps) in facts {
            for _ in 0..reps {
                out.push(fact.clone());
            }
        }
        out
    }
}

#[test]
fn pow_match_naive() {
    use rand::{SeedableRng, RngExt};
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let a = Poly::random(&mut rng, 10);
        let n = rng.random_range(0..64);
        let computed = a.pow(n);
        let naive = (0..n).map(|_| &a).fold(Poly::one(), |a, b| a * b);
        assert_eq!(computed, naive);
    }
}

#[test]
fn pow_mod_match_naive() {
    use rand::{SeedableRng, RngExt};
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let a = Poly::random(&mut rng, 10);
        let m = loop {
            let m = Poly::random(&mut rng, 20);
            if !m.is_zero() { break m }
        };
        let n = rng.random_range(0..64);
        let computed = a.pow_mod(n, &m);
        let naive = (0..n).map(|_| &a).fold(Poly::one(), |a, b| (a * b).rem(&m));
        assert_eq!(computed, naive);
    }
}

#[test]
fn inv_mod_mult_one() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let m = loop {
            let m = Poly::random(&mut rng, 20);
            if !m.is_zero() { break m }
        };
        let a = loop {
            let a = Poly::random(&mut rng, 20);
            if a.gcd(&m).is_one() { break a }
        };
        
        assert_eq!((a.inv_mod(&m).unwrap() * a).rem(&m), Poly::one());
    }
}

#[test]
fn egcd_bezout() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let a = Poly::random(&mut rng, 10);
        let b = Poly::random(&mut rng, 10);
        let (r, s, t) = a.extended_gcd(&b);
        let r2 = a.gcd(&b);
        assert_eq!(r, a*s + t*b);
        assert_eq!(r, r2);
    }
}

#[test]
fn sff_are_square_free() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let poly = Poly::random(&mut rng, 20);
        let factors = poly.square_free_factors();
        let mut rec = Poly::one();
        for (f, m) in factors {
            assert!(f.is_square_free());
            for _ in 0..m {
                rec = rec * &f;
            }
        }
        assert_eq!(rec, poly);
    }
}

#[test]
fn factor_to_irreducibles() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let poly = Poly::random(&mut rng, 20);
        let factors = poly.factor();
        let mut rec = Poly::one();
        for f in factors {
            assert!(f.is_irreducible());
            rec = rec * f;
        }
        assert_eq!(rec, poly);
    }
}
