use std::fmt::{Debug, Display};

use crate::{GF2, Matrix};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LinearSpace {
    basis: Matrix,
    pivots: Vec<usize>
}

impl Debug for LinearSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LinearSpace({} x {}):\n{:?}", self.dim(), self.ambient_dim(), self.basis)
    }
}

impl Display for LinearSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl LinearSpace {
    pub fn empty(n: usize) -> LinearSpace {
        LinearSpace {
            basis: Matrix::zeros(0, n),
            pivots: Vec::new()
        }
    }

    pub fn full(n: usize) -> LinearSpace {
        LinearSpace {
            basis: Matrix::eye(n),
            pivots: (0..n).collect()
        }
    }

    pub fn new(mut basis: Matrix) -> LinearSpace {
        basis.row_reduce_ext(false, ());
        let (mut pivots, rank) = basis.pivot_cols();
        pivots.truncate(rank);
        LinearSpace {
            basis: basis.slice(..rank, ..), 
            pivots
        }
    }

    pub fn basis(&self) -> &Matrix {
        &self.basis
    }

    pub fn dim(&self) -> usize {
        self.basis.num_rows()
    }

    pub fn ambient_dim(&self) -> usize {
        self.basis.num_cols()
    }

    fn reduce_using(&self, target: &mut Matrix) {
        assert_eq!(target.num_cols(), self.basis.num_cols());
        for r in 0..target.num_rows() {
            for (idx, &c) in self.pivots.iter().enumerate() {
                if target[(r, c)] == GF2::ZERO { continue }
                for j in 0..target.num_cols() {
                    target[(r, j)] += self.basis[(idx, j)];
                }
            }
        }
    }

    pub fn contains(&self, v: &Matrix) -> bool {
        let mut v = v.clone();
        self.reduce_using(&mut v);
        v.is_zeros()
    }

    pub fn is_subset_of(&self, parent: &LinearSpace) -> bool {
        parent.contains(&self.basis)
    }

    pub fn is_equal(&self, other: &LinearSpace) -> bool {
        self.dim() == other.dim() && self.is_subset_of(other)
    }

    pub fn is_empty(&self) -> bool {
        self.dim() == 0
    }

    pub fn is_full(&self) -> bool {
        self.dim() == self.ambient_dim()
    }

    fn cycle_sort(&mut self) {
        self.pivots.sort_unstable();
        for i in 0..self.pivots.len()-1 {
            let pivot = (0..self.ambient_dim()).find(|&j| self.basis[(i, j)] == GF2::ONE).unwrap();
            if pivot == self.pivots[i] {
                continue
            }

            let mut pos = self.pivots.binary_search(&pivot).unwrap();
            self.basis.row_swap(pos, i);
            while pos != i {
                let pivot = (0..self.ambient_dim()).find(|&j| self.basis[(i, j)] == GF2::ONE).unwrap();
                pos = self.pivots.binary_search(&pivot).unwrap();
                self.basis.row_swap(pos, i);
            }
        }
    }

    pub fn push(&mut self, v: &Matrix) {
        let mut v = v.clone();
        self.reduce_using(&mut v);
        if v.is_zeros() { return }

        v.row_reduce_ext(false, ());
        let (mut vpivots, vrank) = v.pivot_cols();
        vpivots.truncate(vrank);
        self.basis = self.basis.vconcat(&v.slice(..vrank, ..));
        self.pivots.append(&mut vpivots);
        self.cycle_sort();
    }

    pub fn union(&self, other: &LinearSpace) -> LinearSpace {
        let mut new = self.clone();
        new.push(&other.basis);
        new
    } 

    pub fn intersect(&self, other: &LinearSpace) -> LinearSpace {
        let nsp = self.basis.vconcat(&other.basis).left_null_space();
        let basis = nsp.slice(.., ..self.dim()).dot(&self.basis);
        LinearSpace::new(basis)
    }

    pub fn complement(&self) -> LinearSpace {
        let (mut pivots, rank) = self.basis.pivot_cols();
        let mut cols = pivots.split_off(rank);
        cols.reverse();
        let basis = Matrix::eye(self.ambient_dim()).slice(cols.as_slice(), ..);
        LinearSpace { basis, pivots: cols }
    }
}