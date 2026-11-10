use pyo3::{
    IntoPyObjectExt, PyTypeInfo, exceptions::{PyIndexError, PyTypeError, PyValueError}, intern, prelude::*, types::{PyBytes, PyDict, PyInt, PySlice}
};
use crate::{GF2, InvalidInputError, ShapeMismatchError, index_to_indexerror, linalg::FromInner, rng::with_rng, slice_to_indexerror, LinearSpace};
use gf2_linalg as gf2;

#[pyclass(eq, module = "gf2_linalg")]
#[derive(PartialEq, Eq)]
pub(crate) struct Matrix(pub(crate) gf2::Matrix);

impl Matrix {
    fn repr_with_prefix(&self, prefix: &str, suffix: &str) -> String {
        if self.0.num_rows() == 0 || self.0.num_cols() == 0 {
            return format!("{}empty: shape=({},{}){}", prefix, self.0.num_rows(), self.0.num_cols(), suffix)
        }

        use std::fmt::Write;
        let mut s = String::new();
        for i in 0..self.0.num_rows() {
            if i == 0 {
                let _ = write!(&mut s, "{}[", prefix);
            } else {
                let _ = write!(&mut s, "{:>width$}[", "", width=prefix.len());
            }
            for j in 0..self.0.num_cols() {
                if j < self.0.num_cols() - 1 {
                    let _ = write!(&mut s, "{},", bool::from(self.0[(i, j)]) as u32);
                } else {
                    let _ = write!(&mut s, "{}", bool::from(self.0[(i, j)]) as u32);
                }
            }
            if i < self.0.num_rows() - 1 {
                let _ = write!(&mut s, "],\n");
            } else {
                let _ = write!(&mut s, "]{}", suffix);
            }
        }
        s
    }
}

#[pymethods]
impl Matrix {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Matrix> {
        let mut rows = value.try_iter()?.peekable();
        let row_vec = if let Some(row) = rows.peek() && let Ok(row) = row {
            GF2::new(row).is_ok()
        } else {
            return Ok(Matrix(gf2::Matrix::zeros(0,0)))
        };

        if row_vec {
            let mut data = Vec::new();
            for val in rows {
                data.push(GF2::new(&val?)?.0)
            }
            let shape = (1, data.len());
            Ok(Matrix(gf2::Matrix::from_data(data, shape)))
        } else {
            let mut data = Vec::new();
            let mut num_rows = 0;
            let mut num_cols = 0;
            for row in rows {
                let row = row?.try_iter()?;
                let mut row_cols = 0;
                for val in row {
                    data.push(GF2::new(&val?)?.0);
                    row_cols += 1;
                }

                if num_rows == 0 {
                    num_cols = row_cols;
                } else if row_cols != num_cols {
                    return Err(PyValueError::new_err(format!("expected rows of matrix to be of equal length, but found lengths {} and {}", row_cols, num_cols)))
                }
                num_rows += 1;
            }
            Ok(Matrix(gf2::Matrix::from_data(data, (num_rows, num_cols))))
        }
    }

    fn num_rows(&self) -> usize {
        self.0.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.0.num_cols()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.0.shape
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix(gf2::Matrix::zeros(rows, cols))
    }

    #[staticmethod]
    fn zeros_like(other: &Matrix) -> Matrix {
        Matrix(gf2::Matrix::zeros_like(&other.0))
    }

    #[staticmethod]
    fn ones(rows: usize, cols: usize) -> Matrix {
        Matrix(gf2::Matrix::ones(rows, cols))
    }

    #[staticmethod]
    fn ones_like(other: &Matrix) -> Matrix {
        Matrix(gf2::Matrix::ones_like(&other.0))
    }

    #[staticmethod]
    fn row_vector(i: usize, n: usize) -> Matrix {
        Matrix(gf2::Matrix::row_vector(i, n))
    }

    #[staticmethod]
    fn col_vector(i: usize, n: usize) -> Matrix {
        Matrix(gf2::Matrix::row_vector(i, n))
    }

    #[staticmethod]
    fn eye(n: usize) -> Matrix {
        Matrix(gf2::Matrix::eye(n))
    }

    #[staticmethod]
    fn from_scalar(elem: &GF2) -> Matrix {
        Matrix(gf2::Matrix::from_scalar(elem.0))
    }

    #[staticmethod]
    fn random(py: Python<'_>, rows: usize, cols: usize) -> PyResult<Matrix> {
        with_rng(py, |rng| {
            Matrix(gf2::Matrix::random(rng, rows, cols))
        })
    }

    #[staticmethod]
    fn random_invertible(py: Python<'_>, n: usize) -> PyResult<Matrix> {
        with_rng(py, |rng| {
            Matrix(gf2::Matrix::random_invertible(rng, n))
        })
    }

    #[staticmethod]
    fn block_diagonal(blocks: Bound<'_, PyAny>) -> PyResult<Matrix> {
        let blocks = blocks.try_iter()?
            .map(|block| Ok(PyRefWrapper(block?.cast::<Matrix>()?.borrow())))
            .collect::<PyResult<Vec<_>>>()?;
        gf2::Matrix::block_diagonal_checked(&blocks)
            .map_err(InvalidInputError::from_inner).map(Matrix)
    }

    #[staticmethod]
    fn from_diag(diag: &Matrix) -> PyResult<Matrix> {
        gf2::Matrix::from_diag_checked(&diag.0).map(Matrix).map_err(InvalidInputError::from_inner)
    }

    fn is_identity(&self) -> bool {
        self.0.is_identity()
    }

    fn is_ones(&self) -> bool {
        self.0.is_ones()
    }

    fn is_square(&self) -> bool {
        self.0.is_square()
    }

    fn is_symmetric(&self) -> bool {
        self.0.is_symmetric()
    }

    fn is_zeros(&self) -> bool {
        self.0.is_zeros()
    }

    fn hamming_weight(&self) -> usize {
        self.0.hamming_weight()
    }

    fn fill(&mut self, value: &GF2) {
        self.0.fill(value.0)
    }

    fn to_latex(&self) -> String {
        self.0.to_latex()
    }

    fn col(&self, i: i64) -> PyResult<Matrix> {
        let i = self.normalize_col_index(i)?;
        self.0.slice_checked(.., i).map_err(slice_to_indexerror).map(Matrix)
    }

    fn row(&self, i: i64) -> PyResult<Matrix> {
        let i = self.normalize_row_index(i)?;
        self.0.slice_checked(i, ..).map_err(slice_to_indexerror).map(Matrix)
    }

    #[allow(non_snake_case)]
    #[getter]
    fn T(&self) -> Matrix {
        self.transpose()
    }

    fn transpose(&self) -> Matrix {
        Matrix(self.0.transpose())
    }

    fn broadcast_to(&self, rows: usize, cols: usize) -> PyResult<Matrix> {
        self.0.broadcast_to_checked((rows, cols))
            .map(Matrix).map_err(InvalidInputError::from_inner)
    }

    fn reshape(&self, rows: usize, cols: usize) -> PyResult<Matrix> {
        self.0.clone().reshape_checked(rows, cols)
            .map(Matrix).map_err(ShapeMismatchError::from_inner)
    }

    fn reshape_in_place(&mut self, rows: usize, cols: usize) -> PyResult<()> {
        self.0.reshape_in_place_checked(rows, cols).map_err(ShapeMismatchError::from_inner)
    }

    fn ravel(&self) -> Matrix {
        Matrix(self.0.clone().ravel())
    }

    fn hconcat(&self, other: &Matrix) -> PyResult<Matrix> {
        gf2::Matrix::hstack_checked(&[&self.0, &other.0]).map_err(InvalidInputError::from_inner).map(Matrix)
    }

    fn vconcat(&self, other: &Matrix) -> PyResult<Matrix> {
        gf2::Matrix::vstack_checked(&[&self.0, &other.0]).map_err(InvalidInputError::from_inner).map(Matrix)
    }

    #[staticmethod]
    fn hstack(matrices: Bound<'_, PyAny>) -> PyResult<Matrix> {
        let matrices = matrices.try_iter()?
            .map(|mat| Ok(PyRefWrapper(mat?.cast::<Matrix>()?.borrow())))
            .collect::<PyResult<Vec<_>>>()?;
        gf2::Matrix::hstack_checked(matrices.as_slice())
            .map(Matrix).map_err(InvalidInputError::from_inner)
    }

    #[staticmethod]
    fn vstack(matrices: Bound<'_, PyAny>) -> PyResult<Matrix> {
        let matrices = matrices.try_iter()?
            .map(|mat| Ok(PyRefWrapper(mat?.cast::<Matrix>()?.borrow())))
            .collect::<PyResult<Vec<_>>>()?;
        gf2::Matrix::vstack_checked(matrices.as_slice())
            .map(Matrix).map_err(InvalidInputError::from_inner)
    }

    fn vappend(&mut self, other: &Matrix) -> PyResult<()> {
        self.0.vappend_checked(&other.0).map_err(ShapeMismatchError::from_inner)
    }

    fn swap_remove_row(&mut self, row: i64) -> PyResult<()> {
        let row = self.normalize_row_index(row)?;
        self.0.swap_remove_row_checked(row).map_err(index_to_indexerror)
    }

    fn remove_col(&mut self, col: i64) -> PyResult<()> {
        let col = self.normalize_col_index(col)?;
        self.0.remove_col_checked(col).map_err(index_to_indexerror)
    }

    fn triu(&self) -> Matrix {
        Matrix(self.0.triu())
    }

    fn tril(&self) -> Matrix {
        Matrix(self.0.tril())
    }

    fn diag(&self) -> Matrix {
        Matrix(self.0.diag())
    }

    fn vector_dot(&self, other: &Matrix) -> PyResult<GF2> {
        self.0.vector_dot_checked(&other.0).map(GF2).map_err(ShapeMismatchError::from_inner)
    }

    fn row_space(&self) -> LinearSpace {
        LinearSpace(self.0.row_space())
    }

    fn col_space(&self) -> LinearSpace {
        LinearSpace(self.0.col_space())
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(&self, py: Python<'py>, dtype: Option<&Bound<'py, PyAny>>, copy: Option<bool>) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err("`copy=False` isn't supported. A copy is always created."))
        }

        let np = py.import(intern!(py, "numpy"))?;
        let np_empty = np.getattr(intern!(py, "empty"))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "dtype"), dtype.unwrap_or(PyInt::type_object(py).as_any()))?;
        let output = np_empty.call((self.0.shape,), Some(&kwargs))?;
        
        for i in 0..self.0.shape.0 {
            for j in 0..self.0.shape.1 {
                output.set_item((i,j), bool::from(self.0[(i,j)]))?;
            }
        }

        Ok(output)
    }

    fn __repr__(&self) -> String {
        self.repr_with_prefix("Matrix([", "])")
    }

    pub(crate) fn __str__(&self) -> String {
        self.repr_with_prefix("[", "]")
    }

    fn __getnewargs__(&self) -> ((),) { ((),) }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let len = self.0.num_rows() * self.0.num_cols();
        let bytes = PyBytes::new_with(py, len, |slice| {
            slice.iter_mut().zip(self.0.iter()).for_each(|(v, e)| *v = bool::from(e) as u8);
            Ok(())
        })?;
        Ok((self.num_rows(), self.num_cols(), bytes))
    }

    fn __setstate__(&mut self, state: (usize, usize, Bound<'_, PyBytes>)) {
        use gf2::ToGF2;
        let data = state.2.as_bytes().iter().map(|&e| e.to_gf2()).collect::<Vec<_>>();
        self.0 = gf2::Matrix::from_data(data, (state.0, state.1));
    }

    fn copy(&self) -> Matrix {
        Matrix(self.0.clone())
    }

    fn __getitem__<'py>(&self, py: Python<'py>, idx: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        enum IdxKind {
            Range(std::ops::Range<usize>, bool),
            Slice(Vec<usize>)
        }

        let Ok((row, col)) = idx.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>() else {
            return Err(PyTypeError::new_err(
                "index for Matrix should be a tuple of row and column indices"
            ))
        };
    
        let row = if let Ok(row) = row.extract::<i64>() {
            let row = self.normalize_row_index(row)?;
            IdxKind::Range(row..row+1, true)
        } else if let Ok(rows) = row.cast::<PySlice>() {
            let rows = rows.indices(self.0.num_rows() as isize)?;
            if rows.step != 1 {
                return Err(PyValueError::new_err("indexing by a slice is only supported for step size one"))
            }
            IdxKind::Range(rows.start as usize..rows.stop as usize, false)
        } else {
            let Ok(rows) = row.extract::<Vec<i64>>() else {
                return Err(PyTypeError::new_err("row index for Matrix should be an integer, slice, or list of integers"));
            };
            IdxKind::Slice(rows.into_iter().map(|r| self.normalize_row_index(r)).collect::<PyResult<Vec<_>>>()?)
        };

        let col = if let Ok(col) = col.extract::<i64>() {
            let cols = self.normalize_col_index(col)?;
            IdxKind::Range(cols..cols+1, true)
        } else if let Ok(cols) = col.cast::<PySlice>() {
            let cols = cols.indices(self.0.num_cols() as isize)?;
            if cols.step != 1 {
                return Err(PyValueError::new_err("indexing by a slice is only supported for step size one"))
            }
            IdxKind::Range(cols.start as usize..cols.stop as usize, false)
        } else {
            let Ok(cols) = col.extract::<Vec<i64>>() else {
                return Err(PyTypeError::new_err("column index for Matrix should be an integer, slice, or list of integers"));
            };
            IdxKind::Slice(cols.into_iter().map(|r| self.normalize_col_index(r)).collect::<PyResult<Vec<_>>>()?)
        };

        let sliced = match (row, col) {
            (IdxKind::Range(row, true), IdxKind::Range(col, true)) => {
                self.0.get_checked(row.start, col.start).map(GF2).map(|x| x.into_bound_py_any(py))
            },
            (IdxKind::Range(row, _), IdxKind::Range(col, _)) => {
                self.0.slice_checked(row, col).map(Matrix).map(|x| x.into_bound_py_any(py))
            },
            (IdxKind::Range(row, _), IdxKind::Slice(col)) => {
                self.0.slice_checked(row, col.as_slice()).map(Matrix).map(|x| x.into_bound_py_any(py))
            },
            (IdxKind::Slice(row), IdxKind::Range(col, _)) => {
                self.0.slice_checked(row.as_slice(), col).map(Matrix).map(|x| x.into_bound_py_any(py))
            },
            (IdxKind::Slice(row), IdxKind::Slice(col)) => {
                self.0.slice_checked(row.as_slice(), col.as_slice()).map(Matrix).map(|x| x.into_bound_py_any(py))
            }
        };

        sliced.map_err(slice_to_indexerror)?
    }

    fn __setitem__(&mut self, idx: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let Ok((row, col)) = idx.extract::<(i64, i64)>() else {
            return Err(PyTypeError::new_err("can only set elements of a Matrix, setting a submatrix is not supported"));
        };

        let row = self.normalize_row_index(row)?;
        let col = self.normalize_col_index(col)?;
        let value = GF2::new(&value)?.0;
        let elem = self.0.get_mut_checked(row, col).map_err(slice_to_indexerror)?;

        *elem = value;

        Ok(())
    }

    fn __add__(&self, other: &Matrix) -> PyResult<Matrix> {
        self.0.add_checked(&other.0).map(Matrix).map_err(ShapeMismatchError::from_inner)
    }

    fn __iadd__(&mut self, other: &Matrix) -> PyResult<()> {
        self.0.add_assign_checked(&other.0).map_err(ShapeMismatchError::from_inner)
    }

    fn __sub__(&self, other: &Matrix) -> PyResult<Matrix> {
        self.0.add_checked(&other.0).map(Matrix).map_err(ShapeMismatchError::from_inner)
    }

    fn __isub__(&mut self, other: &Matrix) -> PyResult<()> {
        self.0.add_assign_checked(&other.0).map_err(ShapeMismatchError::from_inner)
    }

    fn __mul__(&self, other: &Matrix) -> PyResult<Matrix> {
        self.0.mul_checked(&other.0).map(Matrix).map_err(ShapeMismatchError::from_inner)
    }

    fn __imul__(&mut self, other: &Matrix) -> PyResult<()> {
        self.0.mul_assign_checked(&other.0).map_err(ShapeMismatchError::from_inner)
    }

    fn dot(&self, other: &Matrix) -> PyResult<Matrix> {
        self.0.dot_checked(&other.0).map(Matrix).map_err(ShapeMismatchError::from_inner)
    }

    fn __matmul__(&self, other: &Matrix) -> PyResult<Matrix> {
        self.dot(other)
    }

    fn __imatmul__(&mut self, other: &Matrix) -> PyResult<()> {
        *self = self.dot(other)?;
        Ok(())
    }

    fn row_sum(&self, row: i64) -> PyResult<GF2> {
        let row = self.normalize_row_index(row)?;
        self.0.row_sum_checked(row).map(GF2).map_err(index_to_indexerror)
    }

    fn row_weight(&self, row: i64) -> PyResult<usize> {
        let row = self.normalize_row_index(row)?;
        self.0.row_weight_checked(row).map_err(index_to_indexerror)
    }

    fn col_sum(&self, col: i64) -> PyResult<GF2> {
        let col = self.normalize_col_index(col)?;
        self.0.col_sum_checked(col).map(GF2).map_err(index_to_indexerror)
    }

    fn col_weight(&self, col: i64) -> PyResult<usize> {
        let col = self.normalize_col_index(col)?;
        self.0.col_weight_checked(col).map_err(index_to_indexerror)
    }

    fn row_add(&mut self, source: i64, target: i64) -> PyResult<()> {
        let source = self.normalize_row_index(source)?;
        let target = self.normalize_row_index(target)?;
        self.0.row_add_checked(source, target).map_err(index_to_indexerror)
    }

    fn row_swap(&mut self, a: i64, b: i64) -> PyResult<()> {
        let a = self.normalize_row_index(a)?;
        let b = self.normalize_row_index(b)?;
        self.0.row_swap_checked(a, b).map_err(index_to_indexerror)
    }

    fn col_add(&mut self, source: i64, target: i64) -> PyResult<()> {
        let source = self.normalize_col_index(source)?;
        let target = self.normalize_col_index(target)?;
        self.0.col_add_checked(source, target).map_err(index_to_indexerror)
    }

    fn col_swap(&mut self, a: i64, b: i64) -> PyResult<()> {
        let a = self.normalize_col_index(a)?;
        let b = self.normalize_col_index(b)?;
        self.0.col_swap_checked(a, b).map_err(index_to_indexerror)
    }
}

impl FromInner for Matrix {
    type Inner = gf2::Matrix;
    fn from_inner(inner: Self::Inner) -> Self { Self(inner) }
}

impl Matrix {
    fn normalize_col_index(&self, i: i64) -> PyResult<usize> {
        if i >= 0 {
            Ok(i as usize)
        } else if (-i) as usize <= self.0.num_cols() {
            Ok(self.0.num_cols() - (-i) as usize)
        } else {
            Err(PyIndexError::new_err(format!("column {} is out of bounds for matrix of shape {:?}", i, self.0.shape)))
        }
    }

    fn normalize_row_index(&self, i: i64) -> PyResult<usize> {
        if i >= 0 {
            Ok(i as usize)
        } else if (-i) as usize <= self.0.num_rows() {
            Ok(self.0.num_cols() - (-i) as usize)
        } else {
            Err(PyIndexError::new_err(format!("row {} is out of bounds for matrix of shape {:?}", i, self.0.shape)))
        }
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

struct PyRefWrapper<'a>(PyRef<'a, Matrix>);

impl<'a> std::borrow::Borrow<gf2::Matrix> for PyRefWrapper<'a> {
    fn borrow(&self) -> &gf2::Matrix {
        &self.0.0
    }
}
