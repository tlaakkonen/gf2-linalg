use pyo3::{
    IntoPyObjectExt, exceptions::PyTypeError, prelude::*, 
    types::{PyInt, PyNotImplemented}
};
use gf2_linalg as gf2;

#[pyclass(module = "gf2_linalg")]
pub(crate) struct GF2(pub(crate) gf2::GF2);

#[pymethods]
impl GF2 {
    #[new]
    #[pyo3(signature = (value: "int"))]
    pub(crate) fn new(value: &Bound<'_, PyAny>) -> PyResult<GF2> {
        if let Ok(value) = value.cast::<GF2>() {
            Ok(GF2(value.borrow().0))
        } else if let Ok(value) = value.cast::<PyInt>() {
            let bool_value = value.rem(2)?.is_truthy()?;
            Ok(GF2(gf2::GF2::new(bool_value)))
        } else if let Ok(value) = value.extract::<i64>() {
            Ok(GF2(gf2::ToGF2::to_gf2(&value)))
        } else {
            Err(PyTypeError::new_err("GF2 can only be created from value of integer type"))
        }
    }

    #[classattr]
    const ONE: GF2 = GF2(gf2::GF2::ONE);
    #[classattr]
    const ZERO: GF2 = GF2(gf2::GF2::ZERO);

    fn __str__(&self) -> &'static str {
        match self.0 {
            gf2::GF2::ONE => "1",
            gf2::GF2::ZERO => "0"
        }
    }

    fn __repr__(&self) -> &'static str {
        match self.0 {
            gf2::GF2::ONE => "GF2(1)",
            gf2::GF2::ZERO => "GF2(0)"
        }
    }
     
    fn __hash__(&self) -> isize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(other) = other.cast::<Self>() {
            (other.borrow().0 == self.0).into_bound_py_any(py)
        } else if let Ok(other) = other.cast::<PyInt>() {
            let int = PyInt::new(py, u32::from(bool::from(self.0))).into_bound_py_any(py)?;
            int.eq(other)?.into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    fn __getnewargs__(&self) -> (bool,) {
        (self.0.into(),)
    }

    fn __lt__(&self, other: &GF2) -> bool {
        self.0 < other.0
    }

    fn __gt__(&self, other: &GF2) -> bool {
        self.0 > other.0
    }

    fn __le__(&self, other: &GF2) -> bool {
        self.0 <= other.0
    }

    fn __ge__(&self, other: &GF2) -> bool {
        self.0 >= other.0
    }

    fn __int__(&self) -> u32 {
        bool::from(self.0) as u32
    }

    fn __bool__(&self) -> bool {
        self.0.into()
    }

    fn __add__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(other) = other.cast::<Self>() {
            GF2(self.0 + other.borrow().0).into_bound_py_any(py)
        } else if let Ok(other) = other.cast::<PyInt>() {
            GF2(self.0 + GF2::new(other)?.0).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    fn __sub__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__add__(py, other)
    }

    fn __iadd__<'py>(&mut self, other: Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(other) = other.cast::<Self>() {
            self.0 += other.borrow().0;
            Ok(())
        } else if let Ok(other) = other.cast::<PyInt>() {
            self.0 += GF2::new(other)?.0;
            Ok(())
        } else {
            Err(PyTypeError::new_err("can only add GF2 or int to GF2"))
        }
    }

    fn __isub__<'py>(&mut self, other: Bound<'py, PyAny>) -> PyResult<()> {
        self.__iadd__(other)
    }

    fn __mul__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(other) = other.cast::<Self>() {
            GF2(self.0 * other.borrow().0).into_bound_py_any(py)
        } else if let Ok(other) = other.cast::<PyInt>() {
            GF2(self.0 * GF2::new(other)?.0).into_bound_py_any(py)
        } else {
            PyNotImplemented::get(py).into_bound_py_any(py)
        }
    }

    fn __imul__<'py>(&mut self, other: Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(other) = other.cast::<Self>() {
            self.0 *= other.borrow().0;
            Ok(())
        } else if let Ok(other) = other.cast::<PyInt>() {
            self.0 *= GF2::new(other)?.0;
            Ok(())
        } else {
            Err(PyTypeError::new_err("can only multiply GF2 by GF2 or int"))
        }
    }
}