use pyo3::{exceptions::PyTypeError, intern, prelude::*};
use gf2_linalg as gf2;
use crate::{InvalidInputError, Matrix, ShapeMismatchError, elem::GF2, index_to_indexerror};
use concat_idents::concat_idents;

#[pyclass(subclass)]
pub(crate) struct GaussRecorder;

#[pymethods]
impl GaussRecorder {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn __new__(_args: Bound<'_, PyAny>, _kwargs: Option<Bound<'_, PyAny>>) -> Self { GaussRecorder }

    #[allow(unused_variables)]
    fn row_add(&mut self, source: usize, target: usize) {}
    
    fn row_swap<'py>(slf: Bound<'_, Self>, py: Python<'py>, a: usize, b: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "row_add"), (a, b), None)?;
        slf.call_method(intern!(py, "row_add"), (b, a), None)?;
        slf.call_method(intern!(py, "row_add"), (a, b), None)?;
        Ok(())
    }

    #[allow(unused_variables)]
    fn col_add(&mut self, source: usize, target: usize) {}
    
    fn col_swap<'py>(slf: Bound<'_, Self>, py: Python<'py>, a: usize, b: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "col_add"), (a, b), None)?;
        slf.call_method(intern!(py, "col_add"), (b, a), None)?;
        slf.call_method(intern!(py, "col_add"), (a, b), None)?;
        Ok(())
    }
}

#[allow(unused_variables)]
trait PyGaussRecorder: Sized {
    fn row_add<'py>(py: Python<'py>, slf: &Bound<'py, Self>, source: usize, target: usize) -> PyResult<()> { Ok(()) }
    fn col_add<'py>(py: Python<'py>, slf: &Bound<'py, Self>, source: usize, target: usize) -> PyResult<()> { Ok(()) }
    fn row_swap<'py>(py: Python<'py>, slf: &Bound<'py, Self>, a: usize, b: usize) -> PyResult<()> { Ok(()) }
    fn col_swap<'py>(py: Python<'py>, slf: &Bound<'py, Self>, a: usize, b: usize) -> PyResult<()> { Ok(()) }
}

#[allow(unused_variables)]
impl PyGaussRecorder for Matrix {
    fn row_add<'py>(py: Python<'py>, slf: &Bound<'py, Self>, source: usize, target: usize) -> PyResult<()> {
        slf.borrow_mut().0.row_add_checked(source, target).map_err(index_to_indexerror)
    }

    fn col_add<'py>(py: Python<'py>, slf: &Bound<'py, Self>, source: usize, target: usize) -> PyResult<()> {
        slf.borrow_mut().0.col_add_checked(source, target).map_err(index_to_indexerror)
    }

    fn row_swap<'py>(py: Python<'py>, slf: &Bound<'py, Self>, a: usize, b: usize) -> PyResult<()> {
        slf.borrow_mut().0.row_swap_checked(a, b).map_err(index_to_indexerror)
    }

    fn col_swap<'py>(py: Python<'py>, slf: &Bound<'py, Self>, a: usize, b: usize) -> PyResult<()> {
        slf.borrow_mut().0.col_swap_checked(a, b).map_err(index_to_indexerror)
    }
}

impl PyGaussRecorder for PyAny {
    fn row_add<'py>(py: Python<'py>, slf: &Bound<'_, Self>, source: usize, target: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "row_add"), (source, target), None).map(|_| ())
    }

    fn row_swap<'py>(py: Python<'py>, slf: &Bound<'_, Self>, a: usize, b: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "row_swap"), (a, b), None).map(|_| ())
    }

    fn col_add<'py>(py: Python<'py>, slf: &Bound<'_, Self>, source: usize, target: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "col_add"), (source, target), None).map(|_| ())
    }

    fn col_swap<'py>(py: Python<'py>, slf: &Bound<'_, Self>, a: usize, b: usize) -> PyResult<()> {
        slf.call_method(intern!(py, "col_swap"), (a, b), None).map(|_| ())
    }
}

enum FallibleGaussRecorderInner<'py> {
    None,
    Matrix(Bound<'py, Matrix>),
    Any(Bound<'py, PyAny>)
}

macro_rules! impl_fallible_gauss_recorder_inner {
    ($method:ident; $($argname:ident: $argty:ty),*) => {
        fn $method(&self, py: Python<'py>, $($argname: $argty),*) -> PyResult<()> {
            match self {
                FallibleGaussRecorderInner::None => Ok(()),
                FallibleGaussRecorderInner::Matrix(mat) => 
                    PyGaussRecorder::$method(py, &mat, $($argname),*),
                FallibleGaussRecorderInner::Any(any) => 
                    PyGaussRecorder::$method(py, &any, $($argname),*)
            }
        }
    }
}

impl<'py> FallibleGaussRecorderInner<'py> {
    impl_fallible_gauss_recorder_inner!(row_add; source: usize, target: usize);
    impl_fallible_gauss_recorder_inner!(row_swap; a: usize, b: usize);
    impl_fallible_gauss_recorder_inner!(col_add; source: usize, target: usize);
    impl_fallible_gauss_recorder_inner!(col_swap; a: usize, b: usize);
}

enum FallibleGaussRecorderInners<'py> {
    One(FallibleGaussRecorderInner<'py>),
    Many(Vec<FallibleGaussRecorderInner<'py>>)
}

macro_rules! impl_fallible_gauss_recorder_inners {
    ($method:ident; $($argname:ident: $argty:ty),*) => {
        fn $method(&self, py: Python<'py>, $($argname: $argty),*) -> PyResult<()> {
            match self {
                FallibleGaussRecorderInners::One(inner) => inner.$method(py, $($argname),*),
                FallibleGaussRecorderInners::Many(inners) => {
                    for inner in inners {
                        inner.$method(py, $($argname),*)?;
                    }
                    Ok(())
                } 
            }
        }
    }
}

impl<'py> FallibleGaussRecorderInners<'py> {
    impl_fallible_gauss_recorder_inners!(row_add; source: usize, target: usize);
    impl_fallible_gauss_recorder_inners!(row_swap; a: usize, b: usize);
    impl_fallible_gauss_recorder_inners!(col_add; source: usize, target: usize);
    impl_fallible_gauss_recorder_inners!(col_swap; a: usize, b: usize);
}

struct FallibleGaussRecorder<'py> {
    py: Python<'py>,
    inner: FallibleGaussRecorderInners<'py>,
    err: Option<PyErr>
}

macro_rules! impl_fallible_gauss_recorder {
    ($method:ident; $($argname:ident: $argty:ty),*) => {
        fn $method(&mut self, $($argname: $argty),*) {
            if self.err.is_some() { return }
            let res = self.inner.$method(self.py, $($argname),*);
            if let Err(err) = res {
                self.err = Some(err);
            }
        }
    }
}

impl<'py> gf2::GaussRecorder for &mut FallibleGaussRecorder<'py> {
    impl_fallible_gauss_recorder!(row_add; source: usize, target: usize);
    impl_fallible_gauss_recorder!(row_swap; a: usize, b: usize);
    impl_fallible_gauss_recorder!(col_add; source: usize, target: usize);
    impl_fallible_gauss_recorder!(col_swap; a: usize, b: usize);
}

fn try_to_gaussrecorder<'py>(py: Python<'py>, rec: Option<Bound<'py, PyAny>>) -> PyResult<FallibleGaussRecorder<'py>> {
    let Some(rec) = rec else {
        return Ok(FallibleGaussRecorder { py, inner: FallibleGaussRecorderInners::One(FallibleGaussRecorderInner::None), err: None })
    };

    if rec.is_none() {
        Ok(FallibleGaussRecorder { py, inner: FallibleGaussRecorderInners::One(FallibleGaussRecorderInner::None), err: None })
    } else if rec.is_instance_of::<GaussRecorder>() {
        Ok(FallibleGaussRecorder { py, inner: FallibleGaussRecorderInners::One(FallibleGaussRecorderInner::Any(rec)), err: None })
    } else {
        let rec = match rec.cast_into::<Matrix>() {
            Ok(mat) => {
                return Ok(FallibleGaussRecorder { py, inner: FallibleGaussRecorderInners::One(FallibleGaussRecorderInner::Matrix(mat)), err: None })
            },
            Err(cast_err) => cast_err.into_inner()
        };

        let rec = rec.try_iter()
            .map_err(|_| PyTypeError::new_err(
                "rec must be a subclass of GaussRecorder, a Matrix, None, or an iterable of the above"
            ))?;
        let mut inners = Vec::new();
        for inner in rec {
            let inner = inner?;
            inners.push(if inner.is_none() {
                FallibleGaussRecorderInner::None
            } else if inner.is_instance_of::<GaussRecorder>() {
                FallibleGaussRecorderInner::Any(inner)
            } else if let Ok(inner) = inner.cast_into::<Matrix>() {
                FallibleGaussRecorderInner::Matrix(inner)
            } else {
                return Err(PyTypeError::new_err(
                    "rec must be a subclass of GaussRecorder, a Matrix, None, or an iterable of the above"
                ));
            });
        }
        Ok(FallibleGaussRecorder { py, inner: FallibleGaussRecorderInners::Many(inners), err: None })
    }
}

#[pymethods]
impl Matrix {
    #[pyo3(signature = (full = true, rec = None))]
    fn row_reduce(&mut self, py: Python<'_>, full: bool, rec: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let mut rec = try_to_gaussrecorder(py, rec)?;
        self.0.row_reduce_ext(full, &mut rec);
        if let Some(err) = rec.err {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn pivot_cols(&self) -> (Vec<usize>, usize) {
        self.0.pivot_cols()
    }

    #[pyo3(signature = (full = true, rec = None))]
    fn col_reduce(&mut self, py: Python<'_>, full: bool, rec: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let mut rec = try_to_gaussrecorder(py, rec)?;
        self.0.col_reduce_ext(full, &mut rec);
        if let Some(err) = rec.err {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn pivot_rows(&self) -> (Vec<usize>, usize) {
        self.0.pivot_rows()
    }

    fn rank(&self) -> usize {
        self.0.rank()
    }

    fn determinant(&self) -> PyResult<GF2> {
        self.0.determinant_checked().map(GF2).map_err(InvalidInputError::from_inner)
    }

    fn is_invertible(&self) -> bool {
        self.0.is_invertible()
    }

    fn null_space(&self) -> Matrix {
        Matrix(self.0.null_space())
    }

    fn left_null_space(&self) -> Matrix {
        Matrix(self.0.left_null_space())
    }

    fn solve(&self, rhs: &Matrix) -> PyResult<Option<Matrix>> {
        self.0.solve_checked(&rhs.0)
            .map(|m| m.map(Matrix))
            .map_err(ShapeMismatchError::from_inner)
    }

    fn inverse(&self) -> PyResult<Option<Matrix>> {
        self.0.inverse_checked()
            .map(|m| m.map(Matrix))
            .map_err(InvalidInputError::from_inner)
    }

    fn right_inverse(&self) -> Option<Matrix> {
        self.0.right_inverse().map(Matrix)
    }

    fn left_inverse(&self) -> Option<Matrix> {
        self.0.left_inverse().map(Matrix)
    }

    fn solve_triangular(&self, lower: bool, rhs: &Matrix) -> PyResult<Option<Matrix>> {
        self.0.solve_triangular_checked(lower, &rhs.0)
            .map(|m| m.map(Matrix))
            .map_err(InvalidInputError::from_inner)
    }

    fn pow(&self, pow: i64) -> PyResult<Matrix> {
        self.0.pow_checked(pow as isize)
            .map(Matrix).map_err(InvalidInputError::from_inner)
    }
}

macro_rules! decomp_struct_getter {
    ($name:ident; #[unwrapped] $fname:ident : $fty:ty) => {
        pub(super) fn $fname(slf: &$name) -> $fty {
            slf.0.$fname.clone()
        }
    };
    ($name:ident; $fname:ident : $fty:ty) => {
        pub(super) fn $fname(slf: &$name) -> $fty {
            <$fty>::from_inner(slf.0.$fname.clone())
        }
    }
}

macro_rules! decomp_struct {
    { 
        struct $name:ident($inner:ty) {
            $($(#[$fmeta:tt])? $fname:ident : $fty:ty),*
        }
    } => {
        #[pyclass]
        pub(crate) struct $name ($inner);

        concat_idents!(mod_name = inner_, $name {
            #[allow(non_snake_case)]
            mod mod_name {
                use super::*;

                $(
                    decomp_struct_getter!($name; $(#[$fmeta])? $fname : $fty);
                )*
            }

            #[pymethods]
            impl $name {
                $(
                    #[getter]
                    fn $fname(&self) -> $fty {
                        mod_name::$fname(self)
                    }
                )*

                fn __repr__(&self) -> String {
                    let fvalues = [$(
                        (stringify!($fname), format!("{}", self.$fname()))
                    ),*];
                    let formatted = format_multiline_repr(&fvalues);
                    format!("{}({})", stringify!($name), formatted)
                }

                fn __str__(&self) -> String {
                    self.__repr__()
                }
            }

            impl $name {
                #[allow(dead_code)]
                pub(crate) fn from_inner(inner: $inner) -> Self {
                    Self(inner)
                }
            }
        });
    }
}

fn format_multiline_repr(values: &[(&str, String)]) -> String {
    use std::fmt::Write;

    let mut any_multiline = false;
    let mut max_width = 0;
    for (name, s) in values {
        if s.contains('\n') {
            any_multiline = true;
            max_width = max_width.max(name.len());
        }
    }

    let mut next_newline = false;
    let mut output = String::new();
    let pad = format!("\n{:width$}", "", width=max_width+3);
    for (i, (name, s)) in values.iter().enumerate() {
        let end = if i == values.len() - 1 { if any_multiline { "\n" } else { "" } } else { "," };
        if s.contains('\n') {
            let val = s.split('\n').collect::<Vec<_>>().join(&pad);
            let _ = write!(&mut output, "\n  {name:>max_width$}={val}{end}");
            next_newline = true;
        } else if next_newline {
            let _ = write!(&mut output, "\n  {name:>max_width$}={s}{end}");
            next_newline = false;
        } else {
            let _ = write!(&mut output, "{name}={s}{end}");
        }
    }
    output
}

decomp_struct! {
    struct PLUDecomposition(gf2::PLUDecomposition) {
        p: Matrix,
        l: Matrix,
        u: Matrix
    }
}

#[pymethods]
impl Matrix {
    fn plu_decomposition(&self) -> PLUDecomposition {
        PLUDecomposition::from_inner(self.0.plu_decomposition())
    }
}

#[pymethods]
impl PLUDecomposition {
    fn solve(&self, rhs: &Matrix) -> Option<Matrix> {
        self.0.solve(&rhs.0).map(Matrix)
    }
}

decomp_struct! {
    struct RankDecomposition(gf2::RankDecomposition) {
        c: Matrix,
        f: Matrix
    }
}

#[pymethods]
impl RankDecomposition {
    fn generalized_inverse(&self) -> Matrix {
        Matrix(self.0.generalized_inverse())
    }
}

#[pymethods]
impl Matrix {
    fn rank_decomposition(&self) -> RankDecomposition {
        RankDecomposition::from_inner(self.0.rank_decomposition())
    }

    fn generalized_inverse(&self) -> Matrix {
        Matrix(self.0.generalized_inverse())
    }
}

decomp_struct! {
    struct FullRankSymmetricDecomposition(gf2::FullRankSymmetricDecomposition) {
        lam: Matrix,
        m: Matrix
    }
}

#[pymethods]
impl Matrix {
    fn full_rank_symmetric_decomposition(&self) -> PyResult<FullRankSymmetricDecomposition> {
        let frsd = self.0.full_rank_symmetric_decomposition_checked()
            .map_err(InvalidInputError::from_inner)?;
        Ok(FullRankSymmetricDecomposition::from_inner(frsd))
    }
}

decomp_struct! {
    struct WittDecomposition(gf2::WittDecomposition) {
        r: Matrix,
        m: Matrix,
        #[unwrapped] num_i: usize,
        #[unwrapped] num_h: usize
    }
}

#[pymethods]
impl WittDecomposition {
    fn congruence_class(&self) -> (usize, bool) {
        self.0.congruence_class()
    }
}

#[pymethods]
impl Matrix {
    fn witt_decomposition(&self) -> PyResult<WittDecomposition> {
        self.0.witt_decomposition_checked()
            .map(WittDecomposition::from_inner).map_err(InvalidInputError::from_inner)
    }
}

decomp_struct! {
    struct SymmetricRankDecomposition(gf2::SymmetricRankDecomposition) {
        m: Matrix
    }
}

#[pymethods]
impl Matrix {
    fn symmetric_rank_decomposition(&self) -> PyResult<SymmetricRankDecomposition> {
        self.0.symmetric_rank_decomposition_checked()
            .map(SymmetricRankDecomposition::from_inner)
            .map_err(InvalidInputError::from_inner)
    }
}

#[pyfunction]
pub(crate) fn check_congruence_symmetric(a: &Matrix, b: &Matrix) -> PyResult<Option<Matrix>> {
    gf2::check_congruence_symmetric_checked(&a.0, &b.0)
        .map(|m| m.map(Matrix))
        .map_err(InvalidInputError::from_inner)
}

decomp_struct! {
    struct CanonicalQuadraticForm(gf2::CanonicalQuadraticForm) {
        p: Matrix,
        k: Matrix,
        c: Matrix,
        #[unwrapped] arf: bool,
        #[unwrapped] rad: bool,
        #[unwrapped] dim: usize
    }
}

#[pymethods]
impl CanonicalQuadraticForm {
    fn equivalence_class(&self) -> (bool, bool, usize) {
        self.0.equivalence_class()
    }
}

#[pymethods]
impl Matrix {
    fn solve_quadratic(&self, y: &GF2) -> PyResult<Option<Matrix>> {
        self.0.solve_quadratic_checked(y.0)
            .map(|m| m.map(Matrix))
            .map_err(InvalidInputError::from_inner)
    }

    fn canonical_quadratic_form(&self) -> PyResult<CanonicalQuadraticForm> {
        self.0.canonical_quadratic_form_checked()
            .map(CanonicalQuadraticForm::from_inner)
            .map_err(InvalidInputError::from_inner)
    }
}

decomp_struct! {
    struct FittingDecomposition(gf2::FittingDecomposition) {
        invertible: Matrix,
        nilpotent: Matrix,
        basis: Matrix
    }
}

#[pymethods]
impl FittingDecomposition {
    fn drazin_inverse(&self) -> Matrix {
        Matrix(self.0.drazin_inverse())
    }
}

#[pymethods]
impl Matrix {
    fn fitting_decomposition(&self) -> PyResult<FittingDecomposition> {
        self.0.fitting_decomposition_checked()
            .map(FittingDecomposition::from_inner)
            .map_err(InvalidInputError::from_inner)
    }

    fn drazin_inverse(&self) -> PyResult<Matrix> {
        self.0.drazin_inverse_checked()
            .map(Matrix).map_err(InvalidInputError::from_inner)
    }
}

decomp_struct! {
    struct TransvectionDecomposition(gf2::TransvectionDecomposition) {
        u: Matrix,
        v: Matrix,
        b: Matrix
    }
}

#[pymethods]
impl Matrix {
    fn transvection_decomposition(&self) -> PyResult<TransvectionDecomposition> {
        self.0.transvection_decomposition_checked()
            .map(TransvectionDecomposition::from_inner)
            .map_err(InvalidInputError::from_inner)
    }
}
