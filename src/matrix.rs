use crate::GF2;

use std::{borrow::Borrow, fmt::{Debug, Display}, ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign}};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Matrix {
    pub shape: (usize, usize),
    data: Vec<GF2>
}

impl Matrix {
    pub fn num_rows(&self) -> usize {
        self.shape.0
    }

    pub fn num_cols(&self) -> usize {
        self.shape.1
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![GF2::ZERO; rows * cols],
            shape: (rows, cols)
        }
    }

    pub fn zeros_like(mat: &Matrix) -> Matrix {
        Matrix::zeros(mat.shape.0, mat.shape.1)
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: vec![GF2::ONE; rows * cols],
            shape: (rows, cols)
        }
    }

    pub fn ones_like(mat: &Matrix) -> Matrix {
        Matrix::ones(mat.shape.0, mat.shape.1)
    }

    pub fn row_vector(i: usize, n: usize) -> Matrix {
        let mut mat = Matrix::zeros(1, n);
        mat[(0, i)] = GF2::ONE;
        mat
    }

    pub fn col_vector(i: usize, n: usize) -> Matrix {
        let mut mat = Matrix::zeros(n, 1);
        mat[(i, 0)] = GF2::ONE;
        mat
    }

    pub fn eye(n: usize) -> Matrix {
        let mut mat = Matrix::zeros(n, n);
        for i in 0..n {
            mat.data[n * i + i] = GF2::ONE;
        }
        mat
    }

    pub fn from_scalar(elem: GF2) -> Matrix {
        Matrix { data: vec![elem], shape: (1, 1) }
    }

    pub fn basis_vector(n: usize, i: usize) -> Matrix {
        let mut data = vec![GF2::ZERO; n];
        data[i] = GF2::ONE;
        Matrix { data, shape: (n, 1) }
    }

    pub fn from_data(data: Vec<GF2>, shape: (usize, usize)) -> Matrix {
        assert_eq!(data.len(), shape.0 * shape.1);
        Matrix { data, shape }
    }

    pub fn block_diagonal(blocks: &[impl Borrow<Matrix>]) -> Matrix {
        assert!(
            blocks.iter().all(|b|  b.borrow().is_square()),
            "cannot assemble block-diagonal matrix from non-square blocks of shape {:?}",
            blocks.iter().map(|b| b.borrow().shape).collect::<Vec<_>>()
        );
        let size = blocks.iter().map(|b| b.borrow().shape.1).sum::<usize>();
        let mut data = Vec::with_capacity(size * size);
        let mut offset = 0;
        for block in blocks {
            let block: &Matrix = block.borrow();
            for i in 0..block.shape.0 {
                data.resize(data.len() + offset, GF2::ZERO);
                data.extend_from_slice(&block.data[i*block.shape.1..(i+1)*block.shape.1]);
                data.resize(data.len() + size - block.shape.1 - offset, GF2::ZERO);
            }
            offset += block.shape.1;
        }
        Matrix { data, shape: (size, size) }
    }

    pub fn is_zeros(&self) -> bool {
        self.data.iter().all(|&elem| elem == GF2::ZERO)
    }

    pub fn is_identity(&self) -> bool {
        self.data.iter().enumerate().all(|(i, &elem)| elem == (i / self.shape.1 == i % self.shape.1).into())
    }

    pub fn is_ones(&self) -> bool {
        self.data.iter().all(|&elem| elem == GF2::ONE)
    }

    pub fn is_square(&self) -> bool {
        self.shape.0 == self.shape.1
    }

    pub fn is_symmetric(&self) -> bool {
        self.is_square() && (0..self.num_rows()).all(|i| (0..self.num_rows()).all(|j| self[(i, j)] == self[(j, i)]))
    }

    pub fn hamming_weight(&self) -> usize {
        self.data.iter().copied().map(bool::from).map(usize::from).sum::<usize>()
    }

    pub fn fill(&mut self, value: GF2) {
        self.data.fill(value)
    }

    #[cfg(feature = "rand")]
    pub fn random(rng: &mut impl rand::Rng, rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: (0..rows * cols).map(|_| rng.random()).collect(),
            shape: (rows, cols)
        }
    }

    #[cfg(feature = "rand")]
    pub fn random_invertible(rng: &mut impl rand::Rng, n: usize) -> Matrix {
        loop {
            let mat = Matrix::random(rng, n, n);
            if mat.is_invertible() {
                return mat
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=GF2> {
        self.data.iter().copied()
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for row in 0..self.shape.0 {
            for col in 0..self.shape.1 {
                if col < self.shape.1 - 1 {
                    write!(f, "{} ", self[(row, col)])?
                } else {
                    write!(f, "{}", self[(row, col)])?
                }
            }

            if row < self.shape.0 - 1 {
                write!(f, "\n ")?
            }
        }
        write!(f, "]")
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Matrix {
    pub fn to_latex(&self) -> String {
        use std::fmt::Write;
        let mut f = String::new();
        write!(f, "\\begin{{pmatrix}}\n  ").unwrap();
        for row in 0..self.shape.0 {
            for col in 0..self.shape.1 {
                if col < self.shape.1 - 1 {
                    write!(f, "{} & ", self[(row, col)]).unwrap();
                } else {
                    write!(f, "{} \\\\", self[(row, col)]).unwrap();
                }
            }

            if row < self.shape.0 - 1 {
                write!(f, "\n  ").unwrap();
            }
        }
        write!(f, "\n\\end{{pmatrix}}").unwrap();
        f
    }
}

pub trait Slice<'r>: private::Slice<'r> {}

use private::SliceIndices;
mod private {
    #[derive(Clone)]
    pub enum SliceIndices<'r> {
        Range(std::ops::Range<usize>),
        Slice(&'r [usize])
    }

    impl<'r> SliceIndices<'r> {
        pub fn len(&self) -> usize {
            match self {
                SliceIndices::Range(r) => r.end - r.start,
                SliceIndices::Slice(s) => s.len()
            }
        }
    }

    impl<'r> std::fmt::Display for SliceIndices<'r> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SliceIndices::Range(r) => write!(f, "{}..{}", r.start, r.end),
                SliceIndices::Slice(s) => write!(f, "{:?}", s)
            }
        }
    }

    impl<'r> Iterator for SliceIndices<'r> {
        type Item = usize;
        fn next(&mut self) -> Option<Self::Item> {
            match self {
                SliceIndices::Range(r) => (r.start < r.end).then(|| { r.start += 1; r.start - 1 }),
                SliceIndices::Slice(s) => s.split_first().map(|(&val, ns)| { *s = ns; val })
            }
        }
    }

    pub trait Slice<'r> {
        fn to_slice_indices(self, len: usize) -> (SliceIndices<'r>, bool);
    }
}

impl<'r> Slice<'r> for usize {}
impl<'r> private::Slice<'r> for usize {
    fn to_slice_indices(self, len: usize) -> (SliceIndices<'r>, bool) { 
        (SliceIndices::Range(self..self+1), self < len)
    }
}

impl<'r> Slice<'r> for &'r [usize] {}
impl<'r> private::Slice<'r> for &'r [usize] {
    fn to_slice_indices(self, len: usize) -> (SliceIndices<'r>, bool) { 
        (SliceIndices::Slice(self), self.iter().all(|&i| i < len))
    }
}

impl<'r, const N: usize> Slice<'r> for &'r [usize; N] {}
impl<'r, const N: usize> private::Slice<'r> for &'r [usize; N] {
    fn to_slice_indices(self, len: usize) -> (SliceIndices<'r>, bool) { self.as_slice().to_slice_indices(len) }
}

macro_rules! impl_slice_bounds {
    ($($t:ty),*) => {$(
        impl<'r> Slice<'r> for $t {}
        impl<'r> private::Slice<'r> for $t {
            fn to_slice_indices(self, len: usize) -> (SliceIndices<'r>, bool) {
                use std::ops::{RangeBounds, Bound};

                let start = match self.start_bound() {
                    Bound::Unbounded => 0,
                    Bound::Included(&v) => v,
                    Bound::Excluded(&v) => v + 1
                };
                let end = match self.end_bound() {
                    Bound::Unbounded => len,
                    Bound::Included(&v) => v + 1,
                    Bound::Excluded(&v) => v
                };

                (SliceIndices::Range(start..end), start <= end && end <= len)
            }
        }
    )*};
}

impl_slice_bounds!(
    std::ops::Range<usize>, std::ops::RangeInclusive<usize>, std::ops::RangeFrom<usize>, 
    std::ops::RangeTo<usize>, std::ops::RangeToInclusive<usize>, std::ops::RangeFull
);

impl Matrix {
    pub fn slice<'m>(&'m self, rows: impl Slice<'m>, cols: impl Slice<'m>) -> Matrix {
        let (rows, rvalid) = rows.to_slice_indices(self.shape.0);
        let (cols, cvalid) = cols.to_slice_indices(self.shape.1);
        assert!(
            rvalid && cvalid, 
            "slice ({}, {}) is out of bounds for matrix of size {:?}", 
            rows, cols, self.shape
        );

        let mut data = Vec::with_capacity(rows.len() * cols.len());
        for row in rows.clone() {
            match &cols {
                SliceIndices::Range(cols) => data.extend_from_slice(&self.data[self.shape.1 * row + cols.start .. self.shape.1 * row + cols.end]),
                SliceIndices::Slice(cols) => data.extend(cols.iter().map(|&col| self.data[self.shape.1 * row + col]))
            }
            
        }
        Matrix { data, shape: (rows.len(), cols.len()) }
    }

    pub fn row(&self, row: usize) -> Matrix {
        self.slice(row, ..)
    }

    pub fn col(&self, col: usize) -> Matrix {
        self.slice(.., col)
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![GF2::ZERO; self.shape.0 * self.shape.1];
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                data[j * self.shape.0 + i] = self.data[i * self.shape.1 + j];
            }
        }
        Matrix { data, shape: (self.shape.1, self.shape.0) }
    }

    pub fn broadcast_to(&self, shape: (usize, usize)) -> Matrix {
        assert!(
            self.shape == shape || self.shape == (1, 1) || 
            (self.shape.0 == shape.0 && self.shape.1 == 1) || (self.shape.1 == shape.1 && self.shape.0 == 1),
            "shape {:?} cannot be broadcast to shape {:?}", self.shape, shape
        );

        if self.shape == shape {
            self.clone()
        } else if self.shape == (1, 1) {
            Matrix { data: vec![self.data[0]; shape.0 * shape.1], shape }
        } else if self.shape.0 == 1 {
            let mut data = Vec::with_capacity(shape.0 * shape.1);
            for _ in 0..shape.0 {
                data.extend_from_slice(&self.data);
            }
            Matrix { data, shape }
        } else {
            let mut data = vec![GF2::ZERO; shape.0 * shape.1];
            for i in 0..shape.0 {
                data[i*shape.1..(i+1)*shape.1].fill(self.data[i]);
            }
            Matrix { data, shape }
        }
    }

    pub fn reshape(mut self, rows: usize, cols: usize) -> Matrix {
        self.reshape_in_place(rows, cols);
        self
    }

    pub fn reshape_in_place(&mut self, rows: usize, cols: usize) {
        assert_eq!(
            self.shape.0 * self.shape.1, rows * cols, 
            "shape {:?} cannot be reshaped to {:?}", self.shape, (rows, cols)
        );
        self.shape = (rows, cols)
    }

    pub fn ravel(self) -> Matrix {
        let num_elements = self.shape.0 * self.shape.1;
        self.reshape(num_elements, 1)
    }

    pub fn hconcat(&self, other: &Matrix) -> Matrix {
        Matrix::hstack(&[self, other])
    }

    pub fn hstack(mats: &[impl Borrow<Matrix>]) -> Matrix {
        assert!(mats.len() > 0, "cannot stack list of empty matrices");
        let height = mats[0].borrow().shape.0;
        assert!(
            mats.iter().all(|m| m.borrow().shape.0 == height), 
            "cannot hstack matrices of shapes {:?}", mats.iter().map(|m| m.borrow().shape).collect::<Vec<_>>()
        );
        let width = mats.iter().map(|m| m.borrow().shape.1).sum::<usize>();
        let mut data = Vec::with_capacity(height * width);
        for i in 0..height {
            for mat in mats {
                let mat = mat.borrow();
                data.extend_from_slice(&mat.data[i*mat.shape.1..(i+1)*mat.shape.1]);
            }
        }

        Matrix { data, shape: (height, width) }
    }

    pub fn vconcat(&self, other: &Matrix) -> Matrix {
        Matrix::vstack(&[self, other])
    }

    pub fn vstack(mats: &[impl Borrow<Matrix>]) -> Matrix {
        assert!(mats.len() > 0, "cannot stack list of empty matrices");
        let width = mats[0].borrow().shape.1;
        assert!(
            mats.iter().all(|m| m.borrow().shape.1 == width),
            "cannot vstack matrices of shapes {:?}", mats.iter().map(|m| m.borrow().shape).collect::<Vec<_>>()
        );
        let height = mats.iter().map(|m| m.borrow().shape.0).sum::<usize>();
        let mut data = Vec::with_capacity(height * width);
        for mat in mats {
            data.extend_from_slice(&mat.borrow().data);
        }
        Matrix { data, shape: (height, width) }
    }

    pub fn vappend(&mut self, other: &Matrix) {
        assert_eq!(other.shape.1, self.shape.1, 
            "cannot vappend matrix of shape {:?} to matrix of shape {:?}", other.shape, self.shape
        );
        self.data.extend_from_slice(&other.data);
        self.shape.0 += other.shape.0;
    }

    pub fn swap_remove_row(&mut self, i: usize) {
        self.row_swap(i, self.num_rows() - 1);
        self.shape.0 -= 1;
        self.data.truncate(self.shape.0 * self.shape.1);
    }

    pub fn remove_col(&mut self, i: usize) {
        let mut idx = 0;
        self.data.retain(|_| {
            let ret = idx != i;
            idx += 1;
            if idx >= self.shape.1 { idx -= self.shape.1; }
            ret
        });
        self.shape.1 -= 1;
    }

    pub fn triu(&self) -> Matrix {
        let mut out = self.clone();
        for i in 0..self.num_rows() {
            for j in 0..i.min(self.num_cols()) {
                out[(i, j)] = GF2::ZERO;
            }
        }
        out
    }

    pub fn tril(&self) -> Matrix {
        let mut out = self.clone();
        for i in 0..self.num_rows() {
            for j in i+1..self.num_cols() {
                out[(i, j)] = GF2::ZERO;
            }
        }
        out
    }

    pub fn diag(&self) -> Matrix {
        let mut out = Matrix::zeros(1, self.num_rows().min(self.num_rows()));
        for i in 0..self.num_rows().min(self.num_rows()) {
            out[(0, i)] = self[(i, i)];
        }
        out
    }

    pub fn from_diag(diag: &Matrix) -> Matrix {
        assert_eq!(diag.num_rows(), 1, "input to from_diag must be a row vector");
        let mut out = Matrix::zeros(diag.num_cols(), diag.num_cols());
        for i in 0..diag.num_cols() {
            out[(i, i)] = diag[(0, i)];
        }
        out
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = GF2;

    fn index(&self, (row, col): (usize, usize)) -> &GF2 {
        assert!(
            row < self.shape.0 && col < self.shape.1,
            "index {:?} is out of bounds for matrix of size {:?}", (row, col), self.shape
        );

        &self.data[self.shape.1 * row + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut GF2 {
        assert!(
            row < self.shape.0 && col < self.shape.1,
            "index {:?} is out of bounds for matrix of size {:?}", (row, col), self.shape
        );

        &mut self.data[self.shape.1 * row + col]
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.shape, rhs.shape, "cannot add matrices of differing shapes");
        Matrix {
            data: self.data.iter().zip(&rhs.data).map(|(&a, &b)| a + b).collect(),
            shape: self.shape
        }
    }
}
impl Add<Matrix> for &Matrix { type Output = Matrix; fn add(self, rhs: Matrix) -> Matrix { self + &rhs } }
impl Add<&Matrix> for Matrix { type Output = Matrix; fn add(self, rhs: &Matrix) -> Matrix { &self + rhs } }
impl Add<Matrix> for Matrix { type Output = Matrix; fn add(self, rhs: Matrix) -> Matrix { &self + &rhs } }
impl Sub<&Matrix> for &Matrix { type Output = Matrix; fn sub(self, rhs: &Matrix) -> Matrix { self + rhs } }
impl Sub<Matrix> for &Matrix { type Output = Matrix; fn sub(self, rhs: Matrix) -> Matrix { self + &rhs } }
impl Sub<&Matrix> for Matrix { type Output = Matrix; fn sub(self, rhs: &Matrix) -> Matrix { &self + rhs } }
impl Sub<Matrix> for Matrix { type Output = Matrix; fn sub(self, rhs: Matrix) -> Matrix { &self + &rhs } }

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        assert_eq!(self.shape, rhs.shape, "cannot add matrices of differing shapes");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, &b)| *a += b);
    }
}
impl AddAssign<Matrix> for Matrix { fn add_assign(&mut self, rhs: Matrix) { *self += &rhs; } }
impl SubAssign<&Matrix> for Matrix { fn sub_assign(&mut self, rhs: &Matrix) { *self += rhs; } }
impl SubAssign<Matrix> for Matrix { fn sub_assign(&mut self, rhs: Matrix) { *self += &rhs; } }

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.shape, rhs.shape, "cannot element-wise multiply matrices of differing shapes");
        Matrix {
            data: self.data.iter().zip(&rhs.data).map(|(&a, &b)| a * b).collect(),
            shape: self.shape
        }
    }
}
impl Mul<Matrix> for &Matrix { type Output = Matrix; fn mul(self, rhs: Matrix) -> Matrix { self * &rhs } }
impl Mul<&Matrix> for Matrix { type Output = Matrix; fn mul(self, rhs: &Matrix) -> Matrix { &self * rhs } }
impl Mul<Matrix> for Matrix { type Output = Matrix; fn mul(self, rhs: Matrix) -> Matrix { &self * &rhs } }

impl MulAssign<&Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &Matrix) {
        assert_eq!(self.shape, rhs.shape, "cannot element-wise multiply matrices of differing shapes");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, &b)| *a *= b);
    }
}
impl MulAssign<Matrix> for Matrix { fn mul_assign(&mut self, rhs: Matrix) { *self *= &rhs; } }
