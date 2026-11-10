use gf2_linalg as gf2;
use pyo3::prelude::*;
use crate::{InvalidInputError, Matrix, ShapeMismatchError, FromInner};

#[pyclass(module = "gf2_linalg")]
pub(crate) struct LinearSpace(pub(crate) gf2::LinearSpace);

#[pymethods]
impl LinearSpace {
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __getnewargs__(&self) -> (Matrix,) {
        (self.basis(),)
    }

    fn copy(&self) -> LinearSpace {
        LinearSpace(self.0.clone())
    }

    #[staticmethod]
    fn empty(n: usize) -> LinearSpace {
        LinearSpace(gf2::LinearSpace::empty(n))
    }

    #[staticmethod]
    fn full(n: usize) -> LinearSpace {
        LinearSpace(gf2::LinearSpace::full(n))
    }

    #[new]
    fn new(rows: &Matrix) -> LinearSpace {
        LinearSpace(gf2::LinearSpace::new(rows.0.clone()))
    }

    pub fn basis(&self) -> Matrix {
        Matrix(self.0.basis().clone())
    }

    pub fn dim(&self) -> usize {
        self.0.dim()
    }

    pub fn ambient_dim(&self) -> usize {
        self.0.ambient_dim()
    }

    pub fn reduce(&self, target: &mut Matrix) -> PyResult<()> {
        self.0.reduce_checked(&mut target.0)
            .map_err(ShapeMismatchError::from_inner)
    }

    pub fn contains(&self, v: &Matrix) -> bool {
        self.0.contains(&v.0)
    }

    pub fn is_subset_of(&self, parent: &LinearSpace) -> bool {
        self.0.is_subset_of(&parent.0)
    }

    pub fn is_equal(&self, other: &LinearSpace) -> bool {
        self.0.is_equal(&other.0)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.0.is_full()
    }

    pub fn push(&mut self, v: &Matrix) -> PyResult<bool> {
        self.0.push_checked(&v.0)
            .map_err(ShapeMismatchError::from_inner)
    }

    pub fn union(&self, other: &LinearSpace) -> PyResult<LinearSpace> {
        self.0.union_checked(&other.0)
            .map_err(ShapeMismatchError::from_inner)
            .map(LinearSpace)
    }

    pub fn intersect(&self, other: &LinearSpace) -> PyResult<LinearSpace> {
        self.0.intersect_checked(&other.0)
            .map_err(InvalidInputError::from_inner)
            .map(LinearSpace)
    }

    pub fn complement(&self) -> LinearSpace {
        LinearSpace(self.0.complement())
    }
}

impl std::fmt::Debug for LinearSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

impl FromInner for LinearSpace {
    type Inner = gf2::LinearSpace;
    fn from_inner(inner: Self::Inner) -> Self { Self(inner) }
}
