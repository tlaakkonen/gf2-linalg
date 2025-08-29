use crate::{Matrix, Poly, GF2};

mod decomp;
pub use decomp::*;

pub trait GaussRecorder {
    #[allow(unused_variables)]
    fn row_add(&mut self, source: usize, target: usize) {}

    fn row_swap(&mut self, a: usize, b: usize) {
        self.row_add(a, b);
        self.row_add(b, a);
        self.row_add(a, b);
    }

    #[allow(unused_variables)]
    fn col_add(&mut self, source: usize, target: usize) {}

    fn col_swap(&mut self, a: usize, b: usize) {
        self.col_add(a, b);
        self.col_add(b, a);
        self.col_add(a, b);
    }
}

impl GaussRecorder for () {}

impl GaussRecorder for &mut Matrix {
    fn row_add(&mut self, source: usize, target: usize) {
        Matrix::row_add(self, source, target);
    }

    fn row_swap(&mut self, a: usize, b: usize) {
        Matrix::row_swap(self, a, b);
    }

    fn col_add(&mut self, source: usize, target: usize) {
        Matrix::col_add(self, source, target);
    }

    fn col_swap(&mut self, a: usize, b: usize) {
        Matrix::col_swap(self, a, b);
    }
}

impl<const N: usize> GaussRecorder for [&mut Matrix; N] {
    fn row_add(&mut self, source: usize, target: usize) {
        for mat in self.iter_mut() {
            mat.row_add(source, target);
        }
    }

    fn col_add(&mut self, source: usize, target: usize) {
        for mat in self.iter_mut() {
            mat.col_add(source, target);
        }
    }

    fn row_swap(&mut self, a: usize, b: usize) {
        for mat in self.iter_mut() {
            mat.row_swap(a, b);
        }
    }

    fn col_swap(&mut self, a: usize, b: usize) {
        for mat in self.iter_mut() {
            mat.col_swap(a, b);
        }
    }
}

impl Matrix {
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.shape.1, other.shape.0,
            "cannot multiply matrix of shape {:?} with matrix of shape {:?}", self.shape, other.shape
        );

        let mut target = Matrix::zeros(self.shape.0, other.shape.1);
        for i in 0..self.shape.0 {
            for j in 0..other.shape.1 {
                for k in 0..self.shape.1 {
                    target[(i, j)] += self[(i, k)] * other[(k, j)];
                }
            }
        }
        target
    }

    pub fn row_add(&mut self, source: usize, target: usize) {
        for i in 0..self.shape.1 {
            let value = self[(source, i)];
            self[(target, i)] += value;
        }
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        for i in 0..self.shape.1 {
            let a_val = self[(a, i)];
            let b_val = self[(b, i)];
            self[(a, i)] = b_val;
            self[(b, i)] = a_val;
        }
    }

    pub fn col_add(&mut self, source: usize, target: usize) {
        for j in 0..self.shape.0 {
            let value = self[(j, source)];
            self[(j, target)] += value;
        }
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        for j in 0..self.shape.0 {
            let a_val = self[(j, a)];
            let b_val = self[(j, b)];
            self[(j, a)] = b_val;
            self[(j, b)] = a_val;
        }
    }

    pub fn row_reduce(&mut self) {
        self.row_reduce_ext(true, ());
    }

    pub fn row_reduce_ext(&mut self, full: bool, mut rec: impl GaussRecorder) {
        let mut col = 0;
        let mut row = 0;
        while row < self.shape.0 && col < self.shape.1 {
            let Some(pivot) = (row..self.shape.0).find(|&i| self[(i, col)] == GF2::ONE) else {
                col += 1;
                continue
            };

            if pivot > row {
                self.row_swap(pivot, row);
                rec.row_swap(pivot, row);
            }

            let range = if full { 0..self.shape.0 } else { row+1..self.shape.0 };
            for i in range {
                if i != row && self[(i, col)] == GF2::ONE {
                    self.row_add(row, i);
                    rec.row_add(row, i);
                }
            }

            row += 1;
            col += 1;
        }
    }

    pub fn pivot_cols(&self) -> (Vec<usize>, usize) {
        let mut perm = vec![0; self.shape.1];
        let mut idx = 0;
        let mut nidx = self.shape.1;

        let mut row = 0;
        let mut col = 0;
        while row < self.shape.0 && col < self.shape.1 {
            if self[(row, col)] == GF2::ONE {
                perm[idx] = col;
                idx += 1;

                row += 1;
                col += 1;
            } else {
                perm[nidx - 1] = col;
                nidx -= 1;

                col += 1;
            }
        }

        while nidx > idx {
            perm[nidx - 1] = col;
            nidx -= 1;
            col += 1;
        }

        (perm, idx)
    }

    pub fn col_reduce(&mut self) {
        self.col_reduce_ext(true, ());
    }

    pub fn col_reduce_ext(&mut self, full: bool, mut rec: impl GaussRecorder) {
        let mut col = 0;
        let mut row = 0;
        while row < self.shape.0 && col < self.shape.1 {
            let Some(pivot) = (col..self.shape.1).find(|&i| self[(row, i)] == GF2::ONE) else {
                row += 1;
                continue
            };

            if pivot > col {
                self.col_swap(pivot, col);
                rec.col_swap(pivot, col);
            }

            let range = if full { 0..self.shape.1 } else { col+1..self.shape.1 };
            for i in range {
                if i != col && self[(row, i)] == GF2::ONE {
                    self.col_add(col, i);
                    rec.col_add(col, i);
                }
            }

            row += 1;
            col += 1;
        }
    }

    pub fn pivot_rows(&self) -> (Vec<usize>, usize) {
        let mut perm = vec![0; self.shape.0];
        let mut idx = 0;
        let mut nidx = self.shape.0;

        let mut row = 0;
        let mut col = 0;
        while row < self.shape.0 && col < self.shape.1 {
            if self[(row, col)] == GF2::ONE {
                perm[idx] = row;
                idx += 1;

                row += 1;
                col += 1;
            } else {
                perm[nidx - 1] = row;
                nidx -= 1;

                row += 1;
            }
        }

        while nidx > idx {
            perm[nidx - 1] = row;
            nidx -= 1;
            row += 1;
        }

        (perm, idx)
    }

    pub fn rank(&self) -> usize {
        let mut mat = if self.shape.0 <= self.shape.1 {
            self.clone()
        } else {
            self.transpose()
        };
        mat.row_reduce_ext(false, ());
        mat.pivot_cols().1
    }

    pub fn determinant(&self) -> GF2 {
        assert_eq!(
            self.shape.0, self.shape.1,
            "cannot find determinant of non-square matrix with shape {:?}", self.shape
        );

        (self.rank() == self.shape.0).into()
    }

    pub fn is_invertible(&self) -> bool {
        (self.shape.0 == self.shape.1) && self.determinant().into()
    }

    pub fn null_space(&self) -> Matrix {
        let mut mat = self.clone();
        mat.row_reduce();
        let (perm, rank) = mat.pivot_cols();

        let mut output = Matrix::zeros(self.shape.1, self.shape.1 - rank);
        for i in 0..output.shape.1 {
            output[(perm[rank + i], i)] = GF2::ONE;
            for j in 0..rank {
                output[(perm[j], i)] = mat[(j, perm[rank + i])];
            }
        }
        
        output
    }

    pub fn left_null_space(&self) -> Matrix {
        let mut mat = self.clone();
        mat.col_reduce();
        let (perm, rank) = mat.pivot_rows();

        let mut output = Matrix::zeros(self.shape.0 - rank, self.shape.0);
        for i in 0..output.shape.0 {
            output[(i, perm[rank + i])] = GF2::ONE;
            for j in 0..rank {
                output[(i, perm[j])] = mat[(perm[rank + i], j)];
            }
        }

        output
    }

    pub fn solve(&self, rhs: &Matrix) -> Option<Matrix> {
        assert_eq!(
            self.shape.0, rhs.shape.0,
            "cannot solve linear system with shapes {:?} and {:?}", self.shape, rhs.shape
        );

        let mut target = rhs.clone();
        let mut mat = self.clone();
        mat.row_reduce_ext(true, &mut target);
        let (perm, rank) = mat.pivot_cols();

        if !target.slice(rank.., ..).is_zeros() {
            return None
        }

        let mut sol = Matrix::zeros(self.shape.1, rhs.shape.1);
        for i in 0..rank {
            for j in 0..rhs.shape.1 {
                sol[(perm[i], j)] = target[(i, j)];
            }
        }
        Some(sol)
    }

    pub fn inverse(&self) -> Option<Matrix> {
        assert_eq!(
            self.shape.0, self.shape.1,
            "cannot invert non-square matrix with shape {:?}", self.shape
        );

        let mut mat = self.clone();
        let mut output = Matrix::eye(self.shape.0);
        mat.row_reduce_ext(true, &mut output);
        
        if (0..self.shape.0).all(|i| mat[(i, i)] == GF2::ONE) {
            Some(output)
        } else {
            None
        }
    }

    pub fn solve_triangular(&self, lower: bool, rhs: &Matrix) -> Option<Matrix> {
        let mut sol = Matrix::zeros(self.shape.1, rhs.shape.1);
        let mut i = if lower { 0 } else { self.shape.0 - 1 };
        loop {
            let pivot = if lower {
                (0..(i + 1).min(self.shape.1)).rfind(|&j| self[(i, j)] == GF2::ONE)
            } else {
                (i.min(self.shape.1 - 1)..self.shape.1).find(|&j| self[(i, j)] == GF2::ONE)
            };

            if let Some(pivot) = pivot {
                for k in 0..rhs.shape.1 {
                    sol[(i, k)] += rhs[(i, k)];
                }
                for j in 0..pivot {
                    if self[(i, j)] == GF2::ONE {
                        for k in 0..rhs.shape.1 {
                            let value = sol[(j, k)];
                            sol[(i, k)] += value;
                        }
                    }
                }
            } else {
                if !(0..rhs.shape.1).all(|k| rhs[(i, k)] == GF2::ZERO) {
                    return None
                }
            }

            if lower {
                i += 1;
                if i == self.shape.0 { break; }
            } else {
                if i == 0 { break; }
                i -= 1;
            }
        }

        Some(sol)
    }
}

