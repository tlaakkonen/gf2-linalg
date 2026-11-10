use pyo3::{exceptions::{PyTypeError, PyValueError}, prelude::*, types::PyNotImplemented, IntoPyObjectExt};
use gf2_linalg as gf2;
use crate::{GF2, InvalidInputError, matrix::Matrix, rng, FromInner};

#[pyclass(frozen, ord, eq, hash, module = "gf2_linalg")]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Poly(pub(crate) gf2::Poly);

#[pymethods]
impl Poly {
    #[new]
    #[pyo3(signature = (coeffs = None, terms = None))]
    fn new(coeffs: Option<&Bound<'_, PyAny>>, terms: Option<Vec<usize>>) -> PyResult<Poly> {
        if let Some(coeffs) = coeffs {
            if terms.is_some() {
                return Err(PyValueError::new_err("at most one of `coeffs` and `terms` should be specified"))
            }

            Ok(Poly(gf2::Poly::new(coeffs.try_iter()
                .map_err(|_| PyTypeError::new_err("`coeffs` should be an iterable of int or GF2 coefficients"))?
                .map(|coeff| coeff.and_then(|c| GF2::new(&c)).map(|g| g.0))
                .collect::<PyResult<Vec<gf2::GF2>>>()?)))
        } else if let Some(terms) = terms {
            Ok(Poly(gf2::Poly::from_terms(terms)))
        } else {
            Err(PyValueError::new_err("at least one of `coeffs` and `terms` should be specified"))
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __repr__(&self) -> String {
        format!("Poly({})", self.__str__())
    }

    fn __getnewargs__(&self) -> (Option<()>, Option<Vec<usize>>) { 
        (None, Some(self.0.terms().to_vec())) 
    }

    fn copy(&self) -> Poly {
        Poly(self.0.clone())
    }

    #[staticmethod]
    fn from_terms(terms: Vec<usize>) -> Poly {
        Poly(gf2::Poly::from_terms(terms))
    }

    #[classattr]
    const ZERO: Poly = Poly(gf2::Poly::ZERO);

    #[allow(non_snake_case)]
    #[classattr]
    fn ONE() -> Poly {
        Poly(gf2::Poly::one())
    }
    
    #[allow(non_snake_case)]
    #[classattr]
    fn X() -> Poly {
        Poly(gf2::Poly::monom(1))
    }

    #[staticmethod]
    fn monom(i: usize) -> Poly {
        Poly(gf2::Poly::monom(i))
    }

    fn terms(&self) -> Vec<usize> {
        self.0.terms().to_vec()
    }

    fn coeffs(&self) -> Vec<GF2> {
        self.0.coeffs().map(GF2).collect()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    fn degree(&self) -> usize {
        self.0.degree()
    }

    fn eval(&self, x: &Bound<'_, PyAny>) -> PyResult<GF2> {
        Ok(GF2(self.0.eval(GF2::new(x)?.0)))
    }

    #[staticmethod]
    fn random(py: Python<'_>, max_degree: usize) -> PyResult<Poly> {
        rng::with_rng(py, |rng| {
            Poly(gf2::Poly::random(rng, max_degree))
        })
    }

    fn coeff(&self, term: usize) -> GF2 {
        GF2(self.0.coeff(term))
    }

    fn __add__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let rhs = if let Ok(rhs) = rhs.cast::<Poly>() {
            rhs.get()
        } else if let Ok(rhs) = GF2::new(rhs) {
            match rhs.0 {
                gf2::GF2::ONE => &Poly::ONE(),
                gf2::GF2::ZERO => &Poly::ZERO
            }
        } else {
            return PyNotImplemented::get(py).into_bound_py_any(py)
        };
        Poly(&self.0 + &rhs.0).into_bound_py_any(py)
    }

    fn __radd__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__add__(py, rhs)
    }

    fn __sub__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__add__(py, rhs)
    }

    fn __rsub__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__add__(py, rhs)
    }

    fn __mul__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let rhs = if let Ok(rhs) = rhs.cast::<Poly>() {
            rhs.get()
        } else if let Ok(rhs) = GF2::new(rhs) {
            match rhs.0 {
                gf2::GF2::ONE => &Poly::ONE(),
                gf2::GF2::ZERO => &Poly::ZERO
            }
        } else {
            return PyNotImplemented::get(py).into_bound_py_any(py)
        };
        Poly(&self.0 * &rhs.0).into_bound_py_any(py)
    }

    fn __rmul__<'py>(&self, py: Python<'py>, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__mul__(py, rhs)
    }


    fn reciprocal(&self) -> Poly {
        Poly(self.0.reciprocal())
    }

    fn trace_mod(&self, f: &Poly) -> PyResult<GF2> {
        self.0.trace_mod_checked(&f.0).map_err(InvalidInputError::from_inner).map(GF2)
    }

    fn square_root(&self) -> Option<Poly> {
        self.0.square_root().map(Poly)
    }

    #[pyo3(signature = (n, m = None))]
    fn pow(&self, n: isize, m: Option<&Poly>) -> PyResult<Option<Poly>> {
        if let Some(m) = m {
            self.0.pow_mod_signed_checked(n, &m.0)
                .map_err(InvalidInputError::from_inner)
                .map(|x| x.map(Poly))
        } else if n > 0 {
            Ok(Some(Poly(self.0.pow(n as usize))))
        } else {
            Err(PyValueError::new_err("for negative exponents a modulus must be specified"))
        }
    }

    fn __pow__(&self, n: isize, m: Option<&Poly>) -> PyResult<Poly> {
        Ok(self.pow(n, m)?.ok_or_else(|| 
            PyValueError::new_err("cannot invert input that is not coprime to modulus"))?)
    }

    fn quot_rem(&self, d: &Poly) -> PyResult<(Poly, Poly)> {
        self.0.quot_rem_checked(&d.0)
            .map_err(InvalidInputError::from_inner)
            .map(|(q, r)| (Poly(q), Poly(r)))
    }

    fn __divmod__(&self, d: &Poly) -> PyResult<(Poly, Poly)> {
        self.quot_rem(d)
    }

    fn quot(&self, d: &Poly) -> PyResult<Poly> {
        self.0.quot_checked(&d.0).map_err(InvalidInputError::from_inner).map(Poly)
    }

    fn __floordiv__(&self, d: &Poly) -> PyResult<Poly> {
        self.quot(d)
    }

    fn rem(&self, d: &Poly) -> PyResult<Poly> {
        self.0.rem_checked(&d.0).map_err(InvalidInputError::from_inner).map(Poly)
    }

    fn __mod__(&self, d: &Poly) -> PyResult<Poly> {
        self.rem(d)
    }

    fn gcd(&self, rhs: &Poly) -> Poly {
        Poly(self.0.gcd(&rhs.0))
    }

    fn extended_gcd(&self, rhs: &Poly) -> (Poly, Poly, Poly) {
        let (r, s, t) = self.0.extended_gcd(&rhs.0);
        (Poly(r), Poly(s), Poly(t))
    }

    fn inv_mod(&self, f: &Poly) -> PyResult<Option<Poly>> {
        self.0.inv_mod_checked(&f.0)
            .map_err(InvalidInputError::from_inner)
            .map(|x| x.map(Poly))
    }

    fn lcm(&self, rhs: &Poly) -> Poly {
        Poly(self.0.lcm(&rhs.0))
    }

    fn diff(&self) -> Poly {
        Poly(self.0.diff())
    }

    fn is_square_free(&self) -> bool {
        self.0.is_square_free()
    }

    fn square_free_factors(&self) -> Vec<(Poly, usize)> {
        self.0.square_free_factors()
            .into_iter()
            .map(|(p, m)| (Poly(p), m))
            .collect()
    }

    fn is_irreducible(&self) -> bool {
        self.0.is_irreducible()
    }

    fn berlekamp_matrix(&self) -> Matrix {
        Matrix(self.0.berlekamp_matrix())
    }

    #[pyo3(signature = (multiplicities = false))]
    fn factor<'py>(&self, py: Python<'py>, multiplicities: bool) -> PyResult<Bound<'py, PyAny>> {
        if multiplicities {
            let factors: Vec<(Poly, usize)> = self.0.factor_with_multiplicities()
                .into_iter()
                .map(|(p, m)| (Poly(p), m))
                .collect();
            factors.into_bound_py_any(py)
        } else {
            let factors: Vec<Poly> = self.0.factor().into_iter().map(Poly).collect();
            factors.into_bound_py_any(py)
        }
    }
}   

impl std::fmt::Debug for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

impl FromInner for Poly {
    type Inner = gf2::Poly;
    fn from_inner(inner: Self::Inner) -> Self { Self(inner) }
}

