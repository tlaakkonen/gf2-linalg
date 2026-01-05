use super::*;

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

impl PLUDecomposition {
    pub fn solve(&self, rhs: &Matrix) -> Option<Matrix> {
        let pb = self.pt.dot(rhs);
        let y = self.l.solve_triangular(true, &pb)?;
        self.u.solve_triangular(false, &y)
    }
}

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


#[derive(Debug, Clone)]
pub struct KrylovSubspace {
    pub action: Matrix,
    pub basis: Matrix,
    pub dual_basis: Matrix,
    pub rank: usize,
    pub min_poly: Poly
}

impl Matrix {
    fn krylov_subspace_ext(&self, vec: &Matrix, rec: impl GaussRecorder) -> Matrix {
        assert_eq!(self.shape.0, self.shape.1, "cannot find krylov subspace of non-square matrix");
        assert!(vec.shape.0 == self.shape.0 && vec.shape.1 == 1, "input to krylov subspace must be a compatible column vector");
        let n = self.shape.0;
    
        let mut vecs = vec![vec.clone()];
        for i in 0..n {
            vecs.push(self.dot(&vecs[i]));
        }
        let mut augmented = Matrix::hstack(&vecs);
        augmented.row_reduce_ext(true, rec);
        augmented
    }

    pub fn minimal_polynomial_of(&self, vec: &Matrix) -> Poly {
        let n = self.shape.0;
        let augmented = self.krylov_subspace_ext(vec, ());
        let rank = (0..n).find(|&i| augmented[(i, i)] == GF2::ZERO).unwrap_or(n);
        Poly::new((0..rank).map(|i| augmented[(i, rank)]).chain([GF2::ONE]))
    }

    pub fn krylov_subspace(&self, vec: &Matrix) -> KrylovSubspace {
        let n = self.shape.0;
        let mut ibch = Matrix::eye(n);
        let augmented = self.krylov_subspace_ext(vec, &mut ibch);
        let rank = (0..n).find(|&i| augmented[(i, i)] == GF2::ZERO).unwrap_or(n);
        let min_poly = Poly::new((0..rank).map(|i| augmented[(i, rank)]).chain([GF2::ONE]));

        let bch = ibch.inverse().unwrap();
        let action = ibch.dot(self).dot(&bch);
        KrylovSubspace { action, basis: bch, dual_basis: ibch, rank, min_poly }
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
        let h = qt.pow(d.degree()).gcd(&d);
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
                indices.extend(total_k+1..n+1);
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
        self.rational_canonical_form()
            .invariant_factors
            .into_iter()
            .fold(Poly::one(), |a, b| a * b)
    }
}

