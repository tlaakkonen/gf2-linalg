use crate::{
    GF2, LinearSpace, ToGF2, 
    error::{
        InvalidInputError, SliceError, IndexError, ShapeMismatchError,
        validate_input, validate_shape, validate_index
    }
};

use std::{borrow::Borrow, fmt::{Debug, Display}, ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign}};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Matrix {
    pub shape: (usize, usize),
    inner: matrix_impl::MatrixInner
}

mod matrix_impl {
    use super::GF2;
    use std::ops::Range;

    #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct MatrixInner {
        data: Vec<GF2>
    }

    impl MatrixInner {
        #[inline]
        #[allow(unused)]
        pub fn from_data(data: Vec<GF2>, pitch_hint: usize) -> Self {
            Self { data }
        }

        #[inline]
        pub fn set(&mut self, idx: usize, v: GF2) {
            self.data[idx] = v;
        }

        #[inline]
        pub fn get(&self, idx: usize) -> GF2 {
            self.data[idx]
        }

        pub fn fill(&mut self, v: GF2) {
            self.data.fill(v);
        }

        pub fn iter_range(&self, range: Range<usize>) -> impl Iterator<Item=GF2> {
            self.data[range].iter().copied()
        }

        pub fn iter(&self) -> impl Iterator<Item=GF2> {
            self.data.iter().copied()
        }

        pub fn extend(&mut self, other: &Self) {
            self.data.extend_from_slice(&other.data)
        }

        pub fn extend_range(&mut self, other: &Self, range: Range<usize>) {
            self.data.extend_from_slice(&other.data[range])
        }

        pub fn extend_iter(&mut self, iter: impl Iterator<Item=GF2>) {
            self.data.extend(iter)
        }

        pub fn truncate(&mut self, len: usize) {
            self.data.truncate(len);
        }

        pub fn index_mut(&mut self, idx: usize) -> &mut GF2 {
            &mut self.data[idx]
        }

        pub fn add(&self, other: &Self) -> Self {
            Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect() }
        }

        pub fn add_assign(&mut self, other: &Self) {
            self.data.iter_mut().zip(&other.data).for_each(|(a, &b)| *a += b);
        }

        pub fn mul(&self, other: &Self) -> Self {
            Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect() }
        }

        pub fn mul_assign(&mut self, other: &Self) {
            self.data.iter_mut().zip(&other.data).for_each(|(a, &b)| *a *= b);
        }

        pub fn transpose(&self, rows: usize, cols: usize) -> Self {
            let mut data = vec![GF2::ZERO; self.data.len()];
            for i in 0..rows {
                for j in 0..cols {
                    data[j * rows + i] = self.data[i * cols + j];
                }
            }
            Self { data }
        }

        pub fn swap_range_within(&mut self, a: Range<usize>, b: Range<usize>) {
            if a == b { return }
            let [a, b] = self.data.get_disjoint_mut([a, b])
                .expect("overlapping ranges passed to swap_range_within");
            a.swap_with_slice(b);
        }

        pub fn add_range_within(&mut self, source: Range<usize>, target: Range<usize>) {
            if source == target {
                self.data[target].fill(GF2::ZERO);
            } else {
                let [source, target] = self.data.get_disjoint_mut([source, target])
                    .expect("overlapping ranges passed to add_range_within");
                target.iter_mut().zip(source).for_each(|(a, &mut b)| *a += b);
            }
        }

        pub fn add_range_assign(&mut self, target: Range<usize>, rhs: &Self, source: Range<usize>) {
            self.data[target].iter_mut().zip(&rhs.data[source]).for_each(|(a, &b)| *a += b);
        }

        pub fn sum_range(&self, r: Range<usize>) -> GF2 {
            self.data[r].iter().copied().sum()
        }

        pub fn weight_range(&self, r: Range<usize>) -> usize {
            self.data[r].iter().copied().sum()
        }

        pub fn vector_dot_range(lhs: &Self, l: Range<usize>, rhs: &Self, r: Range<usize>) -> GF2 {
            lhs.data[l].iter().zip(&rhs.data[r]).map(|(&a, &b)| a * b).sum()
        }
    }
}

impl Matrix {
    pub fn num_rows(&self) -> usize {
        self.shape.0
    }

    pub fn num_cols(&self) -> usize {
        self.shape.1
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix::from_data(vec![GF2::ZERO; rows*cols], (rows, cols))
    }

    pub fn zeros_like(mat: &Matrix) -> Matrix {
        Matrix::zeros(mat.shape.0, mat.shape.1)
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix::from_data(vec![GF2::ONE; rows*cols], (rows, cols))
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
            mat.inner.set(n * i + i, GF2::ONE);
        }
        mat
    }

    pub fn from_scalar(elem: GF2) -> Matrix {
        Matrix::from_data(vec![elem], (1, 1))
    }

    #[inline]
    pub fn from_data_checked(data: Vec<GF2>, shape: (usize, usize)) -> Result<Matrix, InvalidInputError> {
        validate_input!(
            data.len() == shape.0 * shape.1,
            "data of length {:?} cannot be interpreted as a matrix of shape {:?}" in from_data, data.len(), shape
        );
        Ok(Matrix { inner: matrix_impl::MatrixInner::from_data(data, shape.1), shape })
    }

    #[inline]
    pub fn from_data(data: Vec<GF2>, shape: (usize, usize)) -> Matrix {
        Self::from_data_checked(data, shape).unwrap()
    }

    pub fn from_rows<E: ToGF2, const N: usize>(rows: impl AsRef<[[E; N]]>) -> Matrix {
        let data = rows.as_ref().iter().flatten().map(ToGF2::to_gf2).collect::<Vec<_>>();
        let shape = (rows.as_ref().len(), N);
        Matrix::from_data(data, shape)
    }

    pub fn block_diagonal_checked(blocks: &[impl Borrow<Matrix>]) -> Result<Matrix, InvalidInputError> {
        validate_input!(
            blocks.iter().all(|b|  b.borrow().is_square()),
            "cannot assemble block-diagonal matrix from non-square blocks of shape {:?}" in block_diagonal,
            blocks.iter().map(|b| b.borrow().shape).collect::<Vec<_>>()
        );
        
        let size = blocks.iter().map(|b| b.borrow().shape.1).sum::<usize>();
        let mut data = Vec::with_capacity(size * size);
        let mut offset = 0;
        for block in blocks {
            let block: &Matrix = block.borrow();
            for i in 0..block.shape.0 {
                data.resize(data.len() + offset, GF2::ZERO);
                data.extend(block.inner.iter_range(i*block.shape.1..(i+1)*block.shape.1));
                data.resize(data.len() + size - block.shape.1 - offset, GF2::ZERO);
            }
            offset += block.shape.1;
        }
        Ok(Matrix::from_data(data, (size, size)))
    }

    pub fn block_diagonal(blocks: &[impl Borrow<Matrix>]) -> Matrix {
        Self::block_diagonal_checked(blocks).unwrap()
    }

    pub fn is_zeros(&self) -> bool {
        self.inner.iter().all(|elem| elem == GF2::ZERO)
    }

    pub fn is_identity(&self) -> bool {
        self.inner.iter().enumerate().all(|(i, elem)| elem == (i / self.shape.1 == i % self.shape.1).into())
    }

    pub fn is_ones(&self) -> bool {
        self.inner.iter().all(|elem| elem == GF2::ONE)
    }

    pub fn is_square(&self) -> bool {
        self.shape.0 == self.shape.1
    }

    pub fn is_symmetric(&self) -> bool {
        self.is_square() && (0..self.num_rows()).all(|i| (0..self.num_rows()).all(|j| self[(i, j)] == self[(j, i)]))
    }

    pub fn hamming_weight(&self) -> usize {
        self.inner.iter().map(bool::from).map(usize::from).sum::<usize>()
    }

    pub fn fill(&mut self, value: GF2) {
        self.inner.fill(value)
    }

    #[cfg(feature = "rand")]
    pub fn random(rng: &mut impl rand::Rng, rows: usize, cols: usize) -> Matrix {
        use rand::RngExt;
        Matrix::from_data((0..rows * cols).map(|_| rng.random()).collect(), (rows, cols))
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
        self.inner.iter()
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
        let _ = write!(f, "\\begin{{pmatrix}}\n  ");
        for row in 0..self.shape.0 {
            for col in 0..self.shape.1 {
                if col < self.shape.1 - 1 {
                    let _ = write!(f, "{} & ", self[(row, col)]);
                } else {
                    let _ = write!(f, "{} \\\\", self[(row, col)]);
                }
            }

            if row < self.shape.0 - 1 {
                let _ = write!(f, "\n  ");
            }
        }
        let _ = write!(f, "\n\\end{{pmatrix}}");
        f
    }
}

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, Ix0, Ix1, Ix2, Dimension, DimMax, Array2, Data};

#[cfg(feature = "ndarray")]
impl Matrix {
    pub fn from_ndarray<E: ToGF2, S: Data<Elem=E>, D: Dimension + DimMax<Ix2,Output=Ix2>>(arr: ArrayBase<S, D>) -> Matrix {
        if let Ok(arr) = arr.view().into_dimensionality::<Ix0>() {
            Matrix::from_scalar(arr[()].to_gf2())
        } else if let Ok(arr) = arr.view().into_dimensionality::<Ix1>() {
            Matrix::from_data((0..arr.dim()).map(|i| arr[i].to_gf2()).collect(), (arr.dim(), 1))
        } else if let Ok(arr) = arr.view().into_dimensionality::<Ix2>() {
            Matrix::from_data((0..arr.dim().0).map(|i| {
                (0..arr.dim().1).map(move |j| (i, j))
            }).flatten().map(|(i, j)| {
                arr[(i, j)].to_gf2()
            }).collect(), arr.dim())
        } else {
            unreachable!()
        }
    }

    pub fn to_ndarray(&self) -> Array2<bool> {
        Array2::from_shape_fn(self.shape, |(i, j)| self[(i, j)].into())
    }
}

#[test]
fn ndarray_roundtrip() {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    for _ in 0..1000 {
        let mat = Matrix::random(&mut rng, 31, 20);
        let rmat = Matrix::from_ndarray(mat.to_ndarray());
        assert_eq!(mat, rmat);

        let mat = ndarray::Array2::from_shape_fn((20, 31), |_| rng.random::<bool>());
        let rmat = Matrix::from_ndarray(mat.view()).to_ndarray();
        assert_eq!(mat, rmat);
    }
}

#[allow(private_bounds)]
pub trait Slice<'r>: private::Slice<'r> {}

#[derive(Clone)]
enum SliceIndices<'r> {
    Range(std::ops::Range<usize>),
    Slice(&'r [usize])
}

#[derive(Debug, Clone)]
pub(crate) enum SliceIndicesOwned {
    Index(usize),
    Range(std::ops::Range<usize>),
    Slice(Vec<usize>)
}

impl<'r> SliceIndices<'r> {
    fn len(&self) -> usize {
        match self {
            SliceIndices::Range(r) => r.end - r.start,
            SliceIndices::Slice(s) => s.len()
        }
    }

    fn to_owned(&self) -> SliceIndicesOwned {
        match self {
            SliceIndices::Range(r) => SliceIndicesOwned::Range(r.clone()),
            SliceIndices::Slice(s) => SliceIndicesOwned::Slice(s.to_vec())
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

impl<'r> std::fmt::Display for SliceIndicesOwned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SliceIndicesOwned::Index(idx) => write!(f, "{}", idx),
            SliceIndicesOwned::Range(r) => write!(f, "{}", SliceIndices::Range(r.clone())),
            SliceIndicesOwned::Slice(s) => write!(f, "{}", SliceIndices::Slice(s.as_slice()))
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

mod private {
    pub(super) trait Slice<'r> {
        fn to_slice_indices(self, len: usize) -> (super::SliceIndices<'r>, bool);
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
    pub fn slice_checked<'m>(&'m self, rows: impl Slice<'m>, cols: impl Slice<'m>) -> Result<Matrix, SliceError> {
        let (rows, rvalid) = rows.to_slice_indices(self.shape.0);
        let (cols, cvalid) = cols.to_slice_indices(self.shape.1);
        if !rvalid || !cvalid {
            return Err(SliceError {
                rows: rows.to_owned(),
                cols: cols.to_owned(),
                shape: self.shape
            });
        }
        
        let mut inner = matrix_impl::MatrixInner::from_data(Vec::new(), cols.len());
        for row in rows.clone() {
            match &cols {
                SliceIndices::Range(cols) => {
                    inner.extend_range(&self.inner, self.shape.1 * row + cols.start .. self.shape.1 * row + cols.end);
                },
                SliceIndices::Slice(cols) => {
                    inner.extend_iter(cols.iter().map(|&col| self.inner.get(self.shape.1 * row + col)));
                }
            }
            
        }
        Ok(Matrix { inner, shape: (rows.len(), cols.len()) })
    }

    pub fn slice<'m>(&'m self, rows: impl Slice<'m>, cols: impl Slice<'m>) -> Matrix {
        self.slice_checked(rows, cols).unwrap()
    }

    pub fn get_checked(&self, row: usize, col: usize) -> Result<GF2, SliceError> {
        if !(row < self.shape.0 && col < self.shape.1) {
            return Err(SliceError {
                rows: SliceIndicesOwned::Index(row),
                cols: SliceIndicesOwned::Index(col),
                shape: self.shape
            })
        }
        Ok(self.inner.get(self.shape.1 * row + col))
    }

    pub fn get_mut_checked(&mut self, row: usize, col: usize) -> Result<&mut GF2, SliceError> {
        if !(row < self.shape.0 && col < self.shape.1) {
            return Err(SliceError {
                rows: SliceIndicesOwned::Index(row),
                cols: SliceIndicesOwned::Index(col),
                shape: self.shape
            })
        }
        Ok(self.inner.index_mut(self.shape.1 * row + col))
    }
    
    pub fn set_checked(&mut self, row: usize, col: usize, value: GF2) -> Result<(), SliceError> {
        if !(row < self.shape.0 && col < self.shape.1) {
            return Err(SliceError {
                rows: SliceIndicesOwned::Index(row),
                cols: SliceIndicesOwned::Index(col),
                shape: self.shape
            })
        }
        Ok(self.inner.set(self.shape.1 * row + col, value))
    }

    pub fn row(&self, row: usize) -> Matrix {
        self.slice(row, ..)
    }

    pub fn col(&self, col: usize) -> Matrix {
        self.slice(.., col)
    }

    pub fn transpose(&self) -> Matrix {
        Matrix { inner: self.inner.transpose(self.shape.0, self.shape.1), shape: (self.shape.1, self.shape.0) }
    }

    pub fn broadcast_to_checked(&self, shape: (usize, usize)) -> Result<Matrix, InvalidInputError> {
        validate_input!(
            self.shape == shape 
                || self.shape == (1, 1) 
                || (self.shape.0 == shape.0 && self.shape.1 == 1) 
                || (self.shape.1 == shape.1 && self.shape.0 == 1),
            "shape {:?} cannot be broadcast to shape {:?}" in broadcast_to, self.shape, shape
        );
        
        Ok(if self.shape == shape {
            self.clone()
        } else if self.shape == (1, 1) {
            Matrix::from_data(vec![self.inner.get(0); shape.0 * shape.1], shape)
        } else if self.shape.0 == 1 {
            let mut data = Vec::with_capacity(shape.0 * shape.1);
            for _ in 0..shape.0 {
                data.extend(self.inner.iter());
            }
            Matrix::from_data(data, shape)
        } else {
            let mut data = vec![GF2::ZERO; shape.0 * shape.1];
            for i in 0..shape.0 {
                data[i*shape.1..(i+1)*shape.1].fill(self.inner.get(i));
            }
            Matrix::from_data(data, shape)
        })
    }

    pub fn broadcast_to(&self, shape: (usize, usize)) -> Matrix {
        self.broadcast_to_checked(shape).unwrap()
    }

    pub fn reshape_checked(mut self, rows: usize, cols: usize) -> Result<Matrix, ShapeMismatchError> {
        self.reshape_in_place_checked(rows, cols)?;
        Ok(self)
    }

    pub fn reshape(self, rows: usize, cols: usize) -> Matrix {
        self.reshape_checked(rows, cols).unwrap()
    }

    pub fn reshape_in_place_checked(&mut self, rows: usize, cols: usize) -> Result<(), ShapeMismatchError> {
        validate_shape!(
            self.shape.0 * self.shape.1 == rows * cols, 
            self.shape, (rows, cols), reshape
        );
        self.shape = (rows, cols);
        Ok(())
    }

    pub fn ravel(self) -> Matrix {
        let num_elements = self.shape.0 * self.shape.1;
        self.reshape(num_elements, 1)
    }

    pub fn hstack_checked(mats: &[impl Borrow<Matrix>]) -> Result<Matrix, InvalidInputError> {
        validate_input!(mats.len() > 0, "cannot stack list of empty matrices" in hstack);
        let height = mats[0].borrow().shape.0;
        validate_input!(
            mats.iter().all(|m| m.borrow().shape.0 == height), 
            "cannot hstack matrices of shapes {:?}" in hstack, mats.iter().map(|m| m.borrow().shape).collect::<Vec<_>>()
        );
        
        let width = mats.iter().map(|m| m.borrow().shape.1).sum::<usize>();
        let mut inner = matrix_impl::MatrixInner::from_data(Vec::new(), width);
        for i in 0..height {
            for mat in mats {
                let mat = mat.borrow();
                inner.extend_range(&mat.inner, i*mat.shape.1..(i+1)*mat.shape.1);
            }
        }

        Ok(Matrix { inner, shape: (height, width) })
    }

    pub fn hconcat(&self, other: &Matrix) -> Matrix {
        Matrix::hstack(&[self, other])
    }

    pub fn hstack(mats: &[impl Borrow<Matrix>]) -> Matrix {
        Self::hstack_checked(mats).unwrap()
    }

    pub fn vstack_checked(mats: &[impl Borrow<Matrix>]) -> Result<Matrix, InvalidInputError> {
        validate_input!(mats.len() > 0, "cannot stack list of empty matrices" in vstack);
        let width = mats[0].borrow().shape.1;
        validate_input!(
            mats.iter().all(|m| m.borrow().shape.1 == width),
            "cannot vstack matrices of shapes {:?}" in vstack, mats.iter().map(|m| m.borrow().shape).collect::<Vec<_>>()
        );
        let height = mats.iter().map(|m| m.borrow().shape.0).sum::<usize>();
        let mut inner = matrix_impl::MatrixInner::from_data(Vec::new(), width);
        for mat in mats {
            inner.extend(&mat.borrow().inner);
        }
        Ok(Matrix { inner, shape: (height, width) })
    }

    pub fn vconcat(&self, other: &Matrix) -> Matrix {
        Matrix::vstack(&[self, other])
    }

    pub fn vstack(mats: &[impl Borrow<Matrix>]) -> Matrix {
        Self::vstack_checked(mats).unwrap()
    }

    pub fn vappend_checked(&mut self, other: &Matrix) -> Result<(), ShapeMismatchError> {
        validate_shape!(
            other.shape.1 == self.shape.1,
            self.shape, other.shape, vappend
        );
        self.inner.extend(&other.inner);
        self.shape.0 += other.shape.0;
        Ok(())
    }

    pub fn vappend(&mut self, other: &Matrix) {
        self.vappend_checked(other).unwrap();
    }

    pub fn swap_remove_row_checked(&mut self, i: usize) -> Result<(), IndexError> {
        validate_index!(i < self.shape.0, row i, self.shape, swap_remove_row);
        self.row_swap(i, self.num_rows() - 1);
        self.shape.0 -= 1;
        self.inner.truncate(self.shape.0 * self.shape.1);
        Ok(())
    }

    pub fn swap_remove_row(&mut self, i: usize) {
        self.swap_remove_row_checked(i).unwrap();
    }

    pub fn remove_col_checked(&mut self, i: usize) -> Result<(), IndexError> {
        validate_index!(i < self.shape.1, col i, self.shape, remove_col);
        let mut data = Vec::new();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                if c != i {
                    data.push(self.inner.get(r * self.shape.1 + c));
                }
            }
        }
        self.shape.1 -= 1;
        self.inner = matrix_impl::MatrixInner::from_data(data, self.shape.1);
        Ok(())
    }

    pub fn remove_col(&mut self, i: usize) {
        self.remove_col_checked(i).unwrap();
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

    pub fn from_diag_checked(diag: &Matrix) -> Result<Matrix, InvalidInputError> {
        validate_input!(diag.num_rows() == 1, "must be a row vector" in from_diag);
        let mut out = Matrix::zeros(diag.num_cols(), diag.num_cols());
        for i in 0..diag.num_cols() {
            out[(i, i)] = diag[(0, i)];
        }
        Ok(out)
    }

    pub fn from_diag(diag: &Matrix) -> Matrix {
        Self::from_diag_checked(diag).unwrap()
    }

    pub fn vector_dot_checked(&self, other: &Matrix) -> Result<GF2, ShapeMismatchError> {
        validate_shape!(
            self.shape == other.shape,
            self.shape, other.shape, vector_dot
        );

        Ok(matrix_impl::MatrixInner::vector_dot_range(
            &self.inner, 0..self.shape.0*self.shape.1, 
            &other.inner, 0..self.shape.0*self.shape.1
        ))
    }

    pub fn vector_dot(&self, other: &Matrix) -> GF2 {
        self.vector_dot_checked(other).unwrap()
    }

    pub fn row_space(&self) -> LinearSpace {
        LinearSpace::new(self.clone())
    }

    pub fn col_space(&self) -> LinearSpace {
        LinearSpace::new(self.transpose())
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = GF2;

    fn index(&self, (row, col): (usize, usize)) -> &GF2 {
        match self.get_checked(row, col).unwrap() {
            GF2::ONE => &GF2::ONE,
            GF2::ZERO => &GF2::ZERO
        }
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut GF2 {
        self.get_mut_checked(row, col).unwrap()
    }
}

impl Matrix {
    fn add_checked(&self, rhs: &Matrix) -> Result<Matrix, ShapeMismatchError> {
        validate_shape!(
            self.shape == rhs.shape,
            self.shape, rhs.shape, add
        );
        Ok(Matrix { inner: self.inner.add(&rhs.inner), shape: self.shape })
    }

    fn add_assign_checked(&mut self, rhs: &Matrix) -> Result<(), ShapeMismatchError> {
        validate_shape!(
            self.shape == rhs.shape,
            self.shape, rhs.shape, add_assign
        );
        self.inner.add_assign(&rhs.inner);
        Ok(())
    }

    fn mul_checked(&self, rhs: &Matrix) -> Result<Matrix, ShapeMismatchError> {
        validate_shape!(
            self.shape == rhs.shape,
            self.shape, rhs.shape, mul
        );
        Ok(Matrix { inner: self.inner.mul(&rhs.inner), shape: self.shape })
    }

    fn mul_assign_checked(&mut self, rhs: &Matrix) -> Result<(), ShapeMismatchError>{
        validate_shape!(
            self.shape == rhs.shape,
            self.shape, rhs.shape, mul_assign
        );
        self.inner.mul_assign(&rhs.inner);
        Ok(())
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        self.add_checked(rhs).unwrap()
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
        self.add_assign_checked(rhs).unwrap();
    }
}
impl AddAssign<Matrix> for Matrix { fn add_assign(&mut self, rhs: Matrix) { *self += &rhs; } }
impl SubAssign<&Matrix> for Matrix { fn sub_assign(&mut self, rhs: &Matrix) { *self += rhs; } }
impl SubAssign<Matrix> for Matrix { fn sub_assign(&mut self, rhs: Matrix) { *self += &rhs; } }

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        self.mul_checked(rhs).unwrap()
    }
}
impl Mul<Matrix> for &Matrix { type Output = Matrix; fn mul(self, rhs: Matrix) -> Matrix { self * &rhs } }
impl Mul<&Matrix> for Matrix { type Output = Matrix; fn mul(self, rhs: &Matrix) -> Matrix { &self * rhs } }
impl Mul<Matrix> for Matrix { type Output = Matrix; fn mul(self, rhs: Matrix) -> Matrix { &self * &rhs } }

impl MulAssign<&Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &Matrix) {
        self.mul_assign_checked(rhs).unwrap();
    }
}
impl MulAssign<Matrix> for Matrix { fn mul_assign(&mut self, rhs: Matrix) { *self *= &rhs; } }

impl Matrix {
    pub fn dot_checked(&self, other: &Matrix) -> Result<Matrix, ShapeMismatchError> {
        validate_shape!(
            self.shape.1 == other.shape.0,
            self.shape, other.shape, dot
        );

        let mut out = Matrix::zeros(self.shape.0, other.shape.1);
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                if self[(i, j)] == GF2::ONE {
                    out.inner.add_range_assign(
                        i*other.shape.1..(i+1)*other.shape.1, 
                        &other.inner,
                        j*other.shape.1..(j+1)*other.shape.1
                    );
                }
            }
        }
        Ok(out)
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        self.dot_checked(other).unwrap()
    }

    pub fn row_sum_checked(&self, i: usize) -> Result<GF2, IndexError> {
        validate_index!(i < self.shape.0, row i, self.shape, row_sum);
        Ok(self.inner.sum_range(i*self.shape.1..(i+1)*self.shape.1))
    }

    pub fn row_sum(&self, i: usize) -> GF2 {
        self.row_sum_checked(i).unwrap()
    }

    pub fn row_weight_checked(&self, i: usize) -> Result<usize, IndexError> {
        validate_index!(i < self.shape.0, row i, self.shape, row_weight);
        Ok(self.inner.weight_range(i*self.shape.1..(i+1)*self.shape.1))
    }

    pub fn row_weight(&self, i: usize) -> usize {
        self.row_weight_checked(i).unwrap()
    }

    pub fn col_sum_checked(&self, j: usize) -> Result<GF2, IndexError> {
        validate_index!(j < self.shape.1, col j, self.shape, col_sum);
        Ok((0..self.shape.0).map(|i| self[(i, j)]).sum())
    }

    pub fn col_sum(&self, j: usize) -> GF2 {
        self.col_sum_checked(j).unwrap()
    }

    pub fn col_weight_checked(&self, j: usize) -> Result<usize, IndexError> {
        validate_index!(j < self.shape.1, col j, self.shape, col_weight);
        Ok((0..self.shape.0).map(|i| self[(i, j)]).sum())
    }

    pub fn col_weight(&self, j: usize) -> usize {
        self.col_weight_checked(j).unwrap()
    }

    pub fn row_add_checked(&mut self, source: usize, target: usize) -> Result<(), IndexError> {
        validate_index!(source < self.shape.0, row source, self.shape, row_add);
        validate_index!(target < self.shape.0, row target, self.shape, row_add);
        self.inner.add_range_within(source*self.shape.1..(source+1)*self.shape.1, target*self.shape.1..(target+1)*self.shape.1);
        Ok(())
    }

    pub fn row_add(&mut self, source: usize, target: usize) {
        self.row_add_checked(source, target).unwrap();
    }

    pub fn row_swap_checked(&mut self, a: usize, b: usize) -> Result<(), IndexError> {
        validate_index!(a < self.shape.0, row a, self.shape, row_swap);
        validate_index!(b < self.shape.0, row b, self.shape, row_swap);
        self.inner.swap_range_within(a*self.shape.1..(a+1)*self.shape.1, b*self.shape.1..(b+1)*self.shape.1);
        Ok(())
    }

    pub fn row_swap(&mut self, a: usize, b: usize) {
        self.row_swap_checked(a, b).unwrap();
    }

    pub fn col_add_checked(&mut self, source: usize, target: usize) -> Result<(), IndexError> {
        validate_index!(source < self.shape.1, col source, self.shape, col_add);
        validate_index!(target < self.shape.1, col target, self.shape, col_add);
        for j in 0..self.shape.0 {
            let value = self[(j, source)];
            self[(j, target)] += value;
        }
        Ok(())
    }

    pub fn col_add(&mut self, source: usize, target: usize) {
        self.col_add_checked(source, target).unwrap();
    }

    pub fn col_swap_checked(&mut self, a: usize, b: usize) -> Result<(), IndexError> {
        validate_index!(a < self.shape.1, col a, self.shape, col_swap);
        validate_index!(b < self.shape.1, col b, self.shape, col_swap);
        for j in 0..self.shape.0 {
            let a_val = self[(j, a)];
            let b_val = self[(j, b)];
            self[(j, a)] = b_val;
            self[(j, b)] = a_val;
        }
        Ok(())
    }

    pub fn col_swap(&mut self, a: usize, b: usize) {
        self.col_swap_checked(a, b).unwrap();
    }
}
