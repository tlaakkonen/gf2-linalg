mod elem;
mod matrix;
mod linalg;
use {elem::*, matrix::*, linalg::*};

use pyo3::{create_exception, exceptions::{PyValueError, PyIndexError}, prelude::*};
use gf2_linalg as gf2;

create_exception!(python_bindings, ShapeMismatchError, PyValueError);
create_exception!(python_bindings, InvalidInputError, PyValueError);

impl ShapeMismatchError {
    fn from_inner(value: gf2::ShapeMismatchError) -> PyErr {
        ShapeMismatchError::new_err(format!("{}", value))
    }
}

impl InvalidInputError {
    fn from_inner(value: gf2::InvalidInputError) -> PyErr {
        InvalidInputError::new_err(format!("{}", value))
    }
}

fn slice_to_indexerror(s: gf2::SliceError) -> PyErr {
    PyIndexError::new_err(format!("{}", s))
}

fn index_to_indexerror(s: gf2::IndexError) -> PyErr {
    PyIndexError::new_err(format!("{}", s))
}

mod rng {
    use pyo3::{Python, Bound, PyAny, PyErr, PyResult, intern, types::{PyBytes, PyBytesMethods, PyAnyMethods}};

    pub struct PyRng<'py> {
        randbits_fn: Bound<'py, PyAny>,
        randbytes_fn: Bound<'py, PyAny>,
        error: Option<PyErr>
    }

    impl<'py> PyRng<'py> {
        fn call_checked<T>(&mut self, f: impl FnOnce(&PyRng<'py>) -> PyResult<T>) -> Option<T> {
            if self.error.is_some() {
                return None
            }

            match f(self) {
                Ok(value) => Some(value),
                Err(e) => {
                    self.error = Some(e);
                    None
                }
            }
        }
    }

    impl<'py> rand::TryRng for PyRng<'py> {
        type Error = std::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            Ok(self.call_checked(|rng| rng.randbits_fn.call1((32,))?.extract::<u32>()).unwrap_or(0))
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            Ok(self.call_checked(|rng| rng.randbits_fn.call1((64,))?.extract::<u64>()).unwrap_or(0))
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            self.call_checked(|rng| {
                let bytes = rng.randbytes_fn.call1((dst.len(),))?;
                let bytes = bytes.cast::<PyBytes>()?;
                dst.copy_from_slice(bytes.as_bytes());
                Ok(())
            });
            Ok(())
        }
    }   

    pub fn with_rng<T>(py: Python<'_>, f: impl for<'py> FnOnce(&mut PyRng<'py>) -> T) -> PyResult<T> {
        let random = py.import(intern!(py, "random"))?;
        let randbits_fn = random.getattr(intern!(py, "getrandbits"))?;
        let randbytes_fn = random.getattr(intern!(py, "randbytes"))?;
        let mut rng = PyRng { randbits_fn, randbytes_fn, error: None }; 
        let res = f(&mut rng);
        if let Some(err) = rng.error {
            Err(err)
        } else {
            Ok(res)
        }
    }
}

#[pymodule(name="gf2_linalg")]
mod python_bindings {
    #[pymodule_export]
    use super::{
        GF2, Matrix, GaussRecorder, 
        PLUDecomposition, RankDecomposition, 
        FullRankSymmetricDecomposition, SymmetricRankDecomposition, 
        WittDecomposition, check_congruence_symmetric,
        CanonicalQuadraticForm, FittingDecomposition, TransvectionDecomposition
    };
}
