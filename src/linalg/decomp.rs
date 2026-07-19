use crate::LinearSpace;

use super::*;

/// Decompose A = P.T @ L @ U
/// P is a permutation matrix, L is lower triangular, U is upper triangular
#[derive(Debug, Clone)]
pub struct PLUDecomposition {
    pub pt: Matrix,
    pub l: Matrix,
    pub u: Matrix
}

impl Matrix {
    pub fn plu_decomposition(&self) -> PLUDecomposition {        
        struct PLURec { pt: Matrix, l: Matrix }
        impl GaussRecorder for &mut PLURec {
            fn row_add(&mut self, source: usize, target: usize) {
                self.l[(target, source)] = GF2::ONE;
            }

            fn row_swap(&mut self, a: usize, b: usize) {
                self.pt.row_swap(a, b);
                self.l.row_swap(a, b);
            }
        }

        let mut mat = self.clone();
        let mut rec = PLURec { pt: Matrix::eye(mat.shape.0), l: Matrix::zeros(mat.shape.0, mat.shape.0) };
        mat.row_reduce_ext(false, &mut rec);
        rec.l += Matrix::eye(mat.shape.0);

        PLUDecomposition { pt: rec.pt, l: rec.l, u: mat }
    }
}

#[test]
fn plu_decomp_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let plu = mat.plu_decomposition();
        assert_eq!(plu.pt.transpose().dot(&plu.l).dot(&plu.u), mat);
    }
}

impl PLUDecomposition {
    pub fn solve(&self, rhs: &Matrix) -> Option<Matrix> {
        let pb = self.pt.dot(rhs);
        let y = self.l.solve_triangular(true, &pb)?;
        self.u.solve_triangular(false, &y)
    }
}

#[test]
fn plu_decomp_solve() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random_invertible(&mut rng, 10);
        let rhs = Matrix::random(&mut rng, 10, 5);
        let plu = mat.plu_decomposition();
        assert_eq!(plu.solve(&rhs), mat.solve(&rhs));
    }
}

/// Decompose A = CF
/// C is n x rank(A), F is rank(A) x m
#[derive(Debug, Clone)]
pub struct RankDecomposition {
    pub c: Matrix,
    pub f: Matrix
}

impl Matrix {
    pub fn rank_decomposition(&self) -> RankDecomposition {
        let mut rref = self.clone();
        rref.row_reduce();
        let (perm, rank) = rref.pivot_cols();

        let f = rref.slice(..rank, ..);
        let mut c = Matrix::zeros(self.shape.0, rank);
        for j in 0..rank {
            for i in 0..self.shape.0 {
                c[(i, j)] = self[(i, perm[j])];
            }
        }

        RankDecomposition { c, f }
    }
}

#[test]
fn rank_decomp_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let rd = mat.rank_decomposition();
        assert_eq!(rd.c.dot(&rd.f), mat);
    }
}

/// For symmetric A, decompose A = MM^T + lam, where lam is diagonal, M is invertible
#[derive(Debug, Clone)]
pub struct FullRankSymmetricDecomposition {
    pub lam: Matrix,
    pub m: Matrix
}

impl Matrix {
    pub fn full_rank_symmetric_decomposition(&self) -> FullRankSymmetricDecomposition {
        assert!(&self.transpose() == self, "matrix must be symmetric");
        let n = self.shape.0;

        let mut m = Matrix::eye(n);
        for j in 0..n {
            for i in j+1..n {
                m[(i, j)] = (0..j).map(|k| m[(i, k)] * m[(j, k)]).sum::<GF2>() + self[(i, j)];
            }
        }

        let mut lam = Matrix::zeros(n, n);
        for i in 0..n {
            lam[(i, i)] = (0..n).map(|k| m[(i, k)] * m[(i, k)]).sum::<GF2>() + self[(i, i)];
        }

        FullRankSymmetricDecomposition { lam, m }
    }
}

#[test]
fn frsd_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mut mat = Matrix::random(&mut rng, 10, 10);
        for i in 0..10 {
            for j in i+1..10 {
                mat[(j, i)] = mat[(i, j)]
            }
        }
        let frsd = mat.full_rank_symmetric_decomposition();
        assert_eq!(mat, frsd.m.dot(&frsd.m.transpose()) + frsd.lam)
    }
}

/// For symmetric A, decompose A = MRM^T where M is invertible and R is block-diagonal. 
/// Each block is either [1], [[0,1],[1,0]] or [0], organized in this order.
/// num_i and num_h, respectively, indicate that the first num_i and next num_h 
/// blocks are [1] or [[0,1],[1,0]], respectively.
pub struct WittDecomposition {
    pub r: Matrix,
    pub m: Matrix,
    pub num_i: usize,
    pub num_h: usize
}

impl Matrix {
    pub fn witt_decomposition(&self) -> WittDecomposition {
        assert!(&self.transpose() == self, "matrix must be symmetric");

        let mut num_i = 0;
        let mut num_h = 0;
        let mut r = self.clone();
        let mut m = Matrix::eye(self.num_rows());
        let mut idx = 0;
        while idx < r.num_rows() {
            if let Some(p) = (idx..r.num_rows()).find(|&i| r[(i, i)] == GF2::ONE) {
                // Found an 'anisotropic' part:
                r.row_swap(p, idx);
                r.col_swap(p, idx);
                m.col_swap(idx, p);

                for i in idx+1..r.num_rows() {
                    if r[(idx, i)] == GF2::ONE {
                        r.row_add(idx, i);
                        r.col_add(idx, i);
                        m.col_add(i, idx);
                    }
                }

                num_i += 1;
                idx += 1;
            } else if let Some((pi, pj)) = (idx..r.num_rows())
                .map(|j| (j+1..r.num_rows()).map(move |i| (i, j)))
                .flatten()
                .find(|&(i, j)| r[(i, j)] == GF2::ONE) {
                // Found a hyperbolic part:
                r.row_swap(pj, idx);
                r.col_swap(pj, idx);
                r.row_swap(pi, idx+1);
                r.col_swap(pi, idx+1);
                m.col_swap(idx, pj);
                m.col_swap(idx+1, pi);

                for i in idx+2..r.num_rows() {
                    if r[(idx, i)] == GF2::ONE {
                        r.row_add(idx+1, i);
                        r.col_add(idx+1, i);
                        m.col_add(i, idx+1);
                    }
                }

                r.row_swap(idx, idx+1);
                r.col_swap(idx, idx+1);
                m.col_swap(idx+1, idx);

                for i in idx+2..r.num_rows() {
                    if r[(idx, i)] == GF2::ONE {
                        r.row_add(idx+1, i);
                        r.col_add(idx+1, i);
                        m.col_add(i, idx+1);
                    }
                }

                num_h += 1;
                idx += 2;
            } else {
                // No more non-zero elements, can stop early
                break
            }
        }

        WittDecomposition { r, m, num_i, num_h }
    }
}

#[test]
fn witt_decomposition_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mut mat = Matrix::random(&mut rng, 10, 10);
        for i in 0..mat.num_rows() {
            for j in i+1..mat.num_rows() {
                mat[(j, i)] = mat[(i, j)]
            }
        }
        let witt = mat.witt_decomposition();
        assert_eq!(mat, witt.m.dot(&witt.r).dot(&witt.m.transpose()));
    }
}

/// For symmetric A, decompose A = MM^T
/// M is either n x rank(A) or n x (rank(A) + 1) if A is alternating
pub struct SymmetricRankDecomposition {
    pub m: Matrix 
}

impl Matrix {
    pub fn symmetric_rank_decomposition(&self) -> SymmetricRankDecomposition {
        if self.is_zeros() {
            return SymmetricRankDecomposition { m: Matrix::zeros(self.num_rows(), 0) }
        }

        let witt = self.witt_decomposition();
        if witt.num_i == 0 {
            let m = witt.m.slice(.., ..2*witt.num_h);
            let mut a = Matrix::zeros(2*witt.num_h, 2*witt.num_h + 1);
            for i in 0..a.num_rows() {
                for j in i..a.num_cols() {
                    a[(i, j)] = GF2::ONE;
                }
                if i % 2 == 0 {
                    a[(i, i + 1)] = GF2::ZERO;
                }
            }
            SymmetricRankDecomposition { m: m.dot(&a) }
        } else {
            let m1 = witt.m.slice(.., ..witt.num_i-1);
            let m2 = witt.m.slice(.., witt.num_i-1..witt.num_i+2*witt.num_h);
            let mut a = Matrix::zeros(1+2*witt.num_h, 1+2*witt.num_h);
            for i in 0..a.num_rows() {
                for j in i.saturating_sub(1)..a.num_cols() {
                    a[(i, j)] = GF2::ONE;
                }
                if i % 2 == 1 {
                    a[(i, i)] = GF2::ZERO;
                }
            }
            SymmetricRankDecomposition { m: m1.hconcat(&m2.dot(&a)) }
        }
    }
}

#[test]
fn symmetric_rank_decomposition_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([43; 32]);
    for _ in 0..1000 {
        let mut mat = Matrix::random(&mut rng, 10, 10);
        for i in 0..mat.num_rows() {
            for j in i+1..mat.num_rows() {
                mat[(j, i)] = mat[(i, j)]
            }
        }
        let srd = mat.symmetric_rank_decomposition();
        assert_eq!(mat, srd.m.dot(&srd.m.transpose()));
    }
}

/// For a matrix A, find invertible E, P, and nilpotent N such that A = P(E \oplus N)P^-1.
#[derive(Debug, Clone)]
pub struct FittingDecomposition {
    pub invertible_part: Matrix,
    pub nilpotent_part: Matrix,
    pub basis: Matrix,
    pub dual_basis: Matrix
}

impl Matrix {
    pub fn fitting_decomposition(&self) -> FittingDecomposition {
        let mut q = self.clone();
        let mut r = q.rank();
        loop {
            let nq = q.dot(self);
            let nr = nq.rank();
            if r == nr {
                break;
            } else {
                q = nq;
                r = nr;
            }
        }
        
        let nsp = q.null_space();
        q.col_reduce();
        let basis = q.slice(.., ..r).hconcat(&nsp);
        let dual_basis = basis.inverse().unwrap();
        let form = dual_basis.dot(self).dot(&basis);
        let invertible_part = form.slice(..r, ..r);
        let nilpotent_part = form.slice(r.., r..);

        FittingDecomposition { invertible_part, nilpotent_part, basis, dual_basis }
    }
}

#[test]
fn fitting_decomposition_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let fd = mat.fitting_decomposition();
        assert_eq!(fd.basis.dot(&Matrix::block_diagonal(&[&fd.invertible_part, &fd.nilpotent_part])).dot(&fd.dual_basis), mat);
        assert_eq!(fd.invertible_part.determinant(), GF2::ONE);
    }
}

#[derive(Debug, Clone)]
pub struct KrylovSubspace {
    pub action: Matrix,
    pub basis: Matrix,
    pub dual_basis: Matrix,
    pub rank: usize,
    pub min_poly: Poly
}

impl Matrix {
    fn krylov_subspace_ext(&self, vec: &Matrix) -> (Matrix, Poly, usize, LinearSpace)  {
        assert_eq!(self.shape.0, self.shape.1, "cannot find krylov subspace of non-square matrix");
        assert!(vec.shape.0 == self.shape.0 && vec.shape.1 == 1, "input to krylov subspace must be a compatible column vector");

        if vec.is_zeros() {
            return (Matrix::zeros(0, vec.num_rows()), Poly::one(), 0, LinearSpace::empty(vec.num_rows()))
        }

        let mut lsp = LinearSpace::new(vec.transpose());
        let mut vecs = vec![vec.clone()];
        let nv = loop {
            let nv = self.dot(&vecs[vecs.len() - 1]);
            if !lsp.push(&nv.transpose()) { break nv }
            vecs.push(nv);
        };
        let rank = vecs.len();

        for vec in &mut vecs { *vec = vec.clone().reshape(1, vec.shape.0); }
        let vecs =  Matrix::vstack(&vecs);
        let mut coeffs = Matrix::eye(vecs.num_rows())
            .vconcat(&Matrix::zeros(1, vecs.num_rows()));
        let mut ext = vecs.vconcat(&nv.reshape(1, vec.num_rows()));
        ext.row_reduce_ext(false, &mut coeffs);
        let min_poly = Poly::new(coeffs.row(coeffs.num_rows() - 1).iter().chain([GF2::ONE]));

        (vecs, min_poly, rank, lsp)
    }

    pub fn minimal_polynomial_of(&self, vec: &Matrix) -> Poly {
        self.krylov_subspace_ext(vec).1
    }

    pub fn krylov_subspace(&self, vec: &Matrix) -> KrylovSubspace {
        let (vecs, min_poly, rank, lsp) = self.krylov_subspace_ext(vec);
        let bch = vecs.vconcat(&lsp.complement().basis()).transpose();
        let ibch = bch.inverse().unwrap();
        let action = ibch.dot(self).dot(&bch);
        KrylovSubspace { action, basis: bch, dual_basis: ibch, rank, min_poly }
    }

    fn krylov_block_reduce(&self, vec: &Matrix) -> (Poly, Matrix) {
        let (_, min_poly, _, lsp) = self.krylov_subspace_ext(vec);
        let (mut pivots, rank) = lsp.basis().pivot_cols();
        pivots[rank..].reverse();
        let mut rest = self.slice(.., &pivots[rank..]).transpose();
        lsp.reduce(&mut rest);
        let reduced = rest.slice(.., &pivots[rank..]).transpose();
        (min_poly, reduced)
    }
}

#[test]
fn krylov_min_poly_vanish() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let vec = Matrix::random(&mut rng, 10, 1);
        let poly = mat.minimal_polynomial_of(&vec);
        assert!(mat.eval_poly(&poly, &vec).is_zeros())
    }
}

#[test]
fn krylov_min_poly_div_total() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let minpoly = mat.minimal_polynomial();
        let vec = Matrix::random(&mut rng, 10, 1);
        let minpoly_vec = mat.minimal_polynomial_of(&vec);
        assert!(minpoly.rem(&minpoly_vec).is_zero());
    }
}

#[test]
fn krylov_block_reduce_match() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let vec = Matrix::random(&mut rng, 10, 1);
        let ksp = mat.krylov_subspace(&vec);
        let (min_poly, reduced) = mat.krylov_block_reduce(&vec);
        assert_eq!(ksp.min_poly, min_poly);
        assert_eq!(ksp.action.slice(ksp.rank.., ksp.rank..), reduced);
    }
}

#[derive(Debug, Clone)]
pub struct RationalCanonicalForm {
    pub normal_form: Matrix,
    pub indices: Vec<usize>,
    pub basis: Matrix,
    pub dual_basis: Matrix,
    pub invariant_factors: Vec<Poly>
}

impl Matrix {
    pub fn eval_poly(&self, p: &Poly, x: &Matrix) -> Matrix {
        let mut total = Matrix::zeros_like(x);
        let mut z = x.clone();
        for coeff in p.coeffs() {
            if coeff == GF2::ONE {
                total += &z;
            }
            z = self.dot(&z);
        }
        total
    }

    fn lcm_poly_vector(&self, v: &Matrix, mu_v: &Poly, w: &Matrix, mu_w: &Poly) -> Matrix {
        let d = mu_v.gcd(mu_w);
        if d.degree() == 0 {
            return v + w
        }

        let qt = mu_w.quot(&d);
        let h = d.gcd(&qt.pow_mod(d.degree(), &d));
        let k = d.quot(&h);
        let v2 = self.eval_poly(&h, v);
        let w2 = self.eval_poly(&k, w);
        v2 + w2
    }

    pub fn maximal_vector(&self) -> Matrix {
        assert_eq!(self.shape.0, self.shape.1, "cannot find maximal vector of non-square matrix");
        let n = self.shape.0;

        let mut e = Matrix::basis_vector(n, 0);
        let mut p = self.minimal_polynomial_of(&e);
        let mut base_idx = 0;
        let mut l = Vec::new();
        for i in 1..n {
            if p.degree() == n { break; }
            e = Matrix::basis_vector(n, i);
            let pe = self.minimal_polynomial_of(&e);
            let pp = p.lcm(&pe);
            if pp == pe {
                base_idx = i;
                l.clear();
            } else if pp == p {
                continue;
            } else {
                l.push((i, p, pe));
            }
            p = pp;
        }

        let mut z = Matrix::basis_vector(n, base_idx);
        for (i, mu_z, mu_e) in l {
            let e = Matrix::basis_vector(n, i);
            z = self.lcm_poly_vector(&z, &mu_z, &e, &mu_e);
        }
        z
    }

    pub fn minimal_polynomial(&self) -> Poly {
        self.minimal_polynomial_of(&self.maximal_vector())
    }

    fn jacob_complement(&self, d: usize) -> (Matrix, Matrix) {
        let n = self.shape.0;
        let mut rows = vec![Matrix::basis_vector(n, d - 1).transpose()];
        for _ in 1..d {
            rows.push(rows[rows.len() - 1].dot(self));
        }
        let rows =  Matrix::vstack(&rows);
        let extras = rows.null_space();
        let bch = Matrix::eye(n).slice(.., ..d).hconcat(&extras);
        let ibch = bch.inverse().unwrap();
        (bch, ibch)
    }

    fn transform_maximal_block(&self) -> (Matrix, Matrix, usize, Poly) {
        let v = self.maximal_vector();
        let sp = self.krylov_subspace(&v);
        let (bch2, ibch2) = sp.action.jacob_complement(sp.rank);
        let bch_total = sp.basis.dot(&bch2);
        let ibch_total = ibch2.dot(&sp.dual_basis);
        (bch_total, ibch_total, sp.rank, sp.min_poly)
    }

    pub fn rational_canonical_form(&self) -> RationalCanonicalForm {
        assert_eq!(self.shape.0, self.shape.1, "cannot find rational canonical form of non-square matrix");
        let n = self.shape.0;
        let mut t = self.clone();
        let mut total_k = 0;
        let mut indices = vec![0];
        let mut bch = Matrix::eye(n);
        let mut ibch = Matrix::eye(n);
        let mut invariant_factors = Vec::new();
        while total_k < n {
            if t.is_identity() {
                for i in total_k+1..n+1 {
                    invariant_factors.push(Poly::from_terms([0, 1]));
                    indices.push(i);
                }
                break;
            }

            let (bchb, ibchb, k, factor) = t.transform_maximal_block();
            invariant_factors.push(factor);
            t = ibchb.dot(&t).dot(&bchb).slice(k.., k..);
            bch = bch.dot(&Matrix::block_diagonal(&[Matrix::eye(total_k), bchb]));
            ibch = Matrix::block_diagonal(&[Matrix::eye(total_k), ibchb]).dot(&ibch);
            total_k += k;
            indices.push(total_k);
        }

        let normal_form = ibch.dot(self).dot(&bch);

        assert!(bch.dot(&ibch).is_identity() && ibch.dot(&bch).is_identity());
        let mask = Matrix::block_diagonal(&indices.iter().zip(&indices[1..])
            .map(|(&a, &b)| Matrix::ones(b - a, b - a))
            .collect::<Vec<_>>());
        assert!((&normal_form * &mask) == normal_form);
     
        RationalCanonicalForm { normal_form, basis: bch, dual_basis: ibch, indices, invariant_factors }
    }

    pub fn characteristic_polynomial(&self) -> Poly {
        assert!(self.is_square(), "cannot calculate characteristic polynomial of non-square matrix");

        if self.num_rows() == 0 {
            return Poly::one()
        } else if self.is_zeros() {
            return Poly::monom(self.num_rows())
        } else if self.is_identity() {
            return Poly::from_terms([0, 1]).pow(self.num_rows())
        }

        let vec = Matrix::col_vector(0, self.num_rows());
        let (poly, reduced) = self.krylov_block_reduce(&vec);
        return poly * reduced.characteristic_polynomial()
    }
}

impl RationalCanonicalForm {
    pub fn characteristic_polynomial(&self) -> Poly {
        self.invariant_factors.iter()
            .fold(Poly::one(), |a, b| a * b)
    }
}

#[test]
fn min_poly_vanish() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let minpoly = mat.minimal_polynomial();
        assert!(mat.eval_poly(&minpoly, &Matrix::eye(10)).is_zeros());
    }
}

#[test]
fn min_poly_divide_char_poly() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let minpoly = mat.minimal_polynomial();
        let charpoly = mat.characteristic_polynomial();
        assert!(charpoly.rem(&minpoly).is_zero());
    }
}

#[test]
fn rational_canonical_form_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let rcf = mat.rational_canonical_form();
        assert_eq!(rcf.basis.dot(&rcf.normal_form).dot(&rcf.dual_basis), mat);
        assert_eq!(rcf.indices.len(), rcf.invariant_factors.len() + 1);
    }
}

#[test]
fn char_poly_vanish() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let char_poly = mat.characteristic_polynomial();
        assert!(mat.eval_poly(&char_poly, &Matrix::eye(10)).is_zeros());
    }
}

#[test]
fn char_poly_match_rcf() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let char_poly = mat.characteristic_polynomial();
        let rcf = mat.rational_canonical_form();
        let rcf_char_poly = rcf.characteristic_polynomial();
        assert_eq!(rcf_char_poly.degree(), mat.num_rows());
        assert_eq!(char_poly.degree(), mat.num_rows());
        assert_eq!(rcf_char_poly, char_poly);
    }
}

#[derive(Debug, Clone)]
pub struct GeneralizedJordanForm {
    pub normal_form: Matrix,
    pub indices: Vec<usize>,
    pub basis: Matrix,
    pub dual_basis: Matrix,
    pub irreducible_factors: Vec<(Poly, usize)>
}

impl RationalCanonicalForm {
    pub fn generalized_jordan_form(&self) -> GeneralizedJordanForm {
        let mut block_bchs = Vec::new();
        let mut block_ibchs = Vec::new();
        let mut irreducible_factors = Vec::new();
        let mut indices = vec![0];
        for (i, f) in self.invariant_factors.iter().enumerate() {
            let block = self.normal_form.slice(
                self.indices[i]..self.indices[i+1], self.indices[i]..self.indices[i+1]
            );

            let factors = f.factor_with_multiplicities();
            if factors.len() == 1 && factors[0].1 == 1 {
                irreducible_factors.push((f.clone(), 1));
                indices.push(self.indices[i + 1]);
                block_bchs.push(Matrix::eye(block.num_rows()));
                block_ibchs.push(Matrix::eye(block.num_rows()));
                continue
            }

            let mut bvecs = Vec::new();
            for (irred, pow) in factors {
                irreducible_factors.push((irred.clone(), pow));
                indices.push(indices[indices.len() - 1] + irred.degree() * pow);

                let compl = f.quot(&irred.pow(pow));
                let v = block.eval_poly(&compl, &Matrix::basis_vector(block.num_rows(), 0));
                for j in 0..pow {
                    let mut u = block.eval_poly(&irred.pow(j), &v);
                    for _ in 0..irred.degree() {
                        bvecs.push(u.clone());
                        u = block.dot(&u);
                    }
                }
            }

            let basis = Matrix::hstack(&bvecs);
            let mut t = basis.clone();
            t.row_reduce();
            let ibasis = basis.inverse().unwrap();
            block_bchs.push(basis);
            block_ibchs.push(ibasis);
        }

        let bch = Matrix::block_diagonal(&block_bchs);
        let ibch = Matrix::block_diagonal(&block_ibchs);
        let normal_form = ibch.dot(&self.normal_form).dot(&bch);
        let bch = self.basis.dot(&bch);
        let ibch = ibch.dot(&self.dual_basis);

        GeneralizedJordanForm { 
            normal_form, indices, irreducible_factors,
            basis: bch, dual_basis: ibch
        }
    }
}

impl Matrix {
    pub fn generalized_jordan_form(&self) -> GeneralizedJordanForm {
        self.rational_canonical_form().generalized_jordan_form()
    }
}

#[test]
fn nilpotent_gjf_mask() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mut mat = Matrix::random(&mut rng, 10, 10);
        for i in 0..10 {
            for j in 0..=i {
                mat[(i, j)] = GF2::ZERO;
            }
        }
        let gjf= mat.generalized_jordan_form();
        let nf = gjf.normal_form;
        for i in 0..10 {
            for j in 0..10 {
                assert!(nf[(i, j)] == GF2::ZERO || (i > 0 && j == i - 1));
            }
        }
    }
}

#[test]
fn generalized_jordan_form_roundtrip() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 10, 10);
        let gjf = mat.generalized_jordan_form();
        assert_eq!(gjf.basis.dot(&gjf.normal_form).dot(&gjf.dual_basis), mat);
        let mask = Matrix::block_diagonal(&gjf.indices.iter().zip(&gjf.indices[1..])
            .map(|(&a, &b)| Matrix::ones(b - a, b - a))
            .collect::<Vec<_>>());
        assert_eq!(&gjf.normal_form * &mask, gjf.normal_form);
    }
}

// import numpy as np
// from galois import GF2, FieldArray

// def transvection_decomp(A: FieldArray) -> tuple[FieldArray, FieldArray]:
// 		I = GF2.Identity(A.shape[0])    
// 		if not np.any(A + I):
// 				return GF2.Zeros((0,A.shape[0])), GF2.Zeros((0,A.shape[0]))
			
//     Ap = GF2.Identity(A.shape[0])
//     Aw = A.copy()
//     U = []
//     V = []    
//     while np.any(Aw + I):
// 		    # Pick v != 0 with Nv = 0:
// 		    N = (Aw + I).null_space()
// 		    v = N.null_space()[0, :]
		    
// 		    # Pick x so that v^TAx = v^Tx = 1:
// 		    vA = np.dot(Aw.T, v)
// 		    if np.any(vA * v):
// 				    i = np.argmax(vA * v)
// 				    x = GF2.Zeros(A.shape[0])
// 					  x[i] = 1
// 			  else:
// 					  i = np.argmax(vA * (vA + v))
// 					  j = np.argmax(v * (vA + v))
// 					  x = GF2.Zeros(A.shape[0])
// 					  x[i] = 1
// 					  x[j] = 1
				
// 				u = np.dot(Aw, x) + x
				
//         assert np.dot(vA, x) == 1
//         assert np.dot(v, x) == 1
//         assert not np.any(np.dot(N, v))
//         assert np.linalg.det(I + np.outer(u, v)) != 0
        
//         Aw = (I + np.outer(u, v)) @ Aw
//         Ap = Ap @ (I + np.outer(u, v))
//         U.append(u)
//         V.append(v)
        
//     assert not np.any(A + Ap)
//     return np.stack(U, axis=0), np.stack(V, axis=0)
