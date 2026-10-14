use crate::SliceIndicesOwned;

#[derive(Debug, Clone)]
pub struct InvalidInputError {
    pub method: &'static str,
    pub message: String   
}

impl InvalidInputError {
    pub(crate) fn with_method(self, method: &'static str) -> Self {
        InvalidInputError { method, message: self.message }
    }
}

impl std::fmt::Display for InvalidInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input for method `{}` is not valid: {}", self.method, self.message)
    }
}

impl std::error::Error for InvalidInputError {}

macro_rules! validate_input {
    ($cond:expr, $fmt:literal in $method:ident $(,$arg:expr)*) => {
        if !($cond) {
            return Err(crate::error::InvalidInputError {
                method: stringify!($method),
                message: format!($fmt $(,$arg)*)
            })
        }
    };
}

pub(crate) use validate_input;

#[derive(Debug, Clone)]
pub struct ShapeMismatchError {
    pub method: &'static str,
    pub expected: (usize, usize),
    pub found: (usize, usize)
}

impl std::fmt::Display for ShapeMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "in `{}`, shapes {:?} and {:?} are not compatible", self.method, self.expected, self.found)
    }
}

impl std::error::Error for ShapeMismatchError {}

macro_rules! validate_shape {
    ($cond:expr, $expected:expr, $found:expr, $method:ident) => {
        if !($cond) {
            return Err(crate::error::ShapeMismatchError {
                method: stringify!($method),
                expected: $expected,
                found: $found
            });
        }
    }
}

pub(crate) use validate_shape;

#[derive(Debug, Clone)]
pub struct SliceError {
    pub(crate) rows: SliceIndicesOwned,
    pub(crate) cols: SliceIndicesOwned,
    pub(crate) shape: (usize, usize)
}

impl std::fmt::Display for SliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "indices ({}, {}) are out of bounds for matrix of shape {:?}", self.rows, self.cols, self.shape)
    }
}

impl std::error::Error for SliceError {}

#[derive(Debug, Clone)]
pub struct IndexError {
    pub(crate) method: &'static str,
    pub(crate) kind: IndexErrorKind,
    pub(crate) shape: (usize, usize)
}

#[derive(Debug, Clone)]
pub(crate) enum IndexErrorKind {
    Row(usize),
    Col(usize)
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            IndexErrorKind::Row(row) => write!(f, "in `{}: row {} is out of bounds for matrix of shape {:?}", self.method, row, self.shape),
            IndexErrorKind::Col(col) => write!(f, "in `{}: col {} is out of bounds for matrix of shape {:?}", self.method, col, self.shape)
        }
    }
}

impl std::error::Error for IndexError {}

macro_rules! validate_index {
    ($cond:expr, row $row:expr, $shape:expr, $method:ident) => {
        if !($cond) { return Err(crate::error::IndexError { method: stringify!($method), kind: crate::error::IndexErrorKind::Row($row), shape: $shape }); }
    };
    ($cond:expr, col $col:expr, $shape:expr, $method:ident) => {
        if !($cond) { return Err(crate::error::IndexError { method: stringify!($method), kind: crate::error::IndexErrorKind::Col($col), shape: $shape }); }
    };
}

pub(crate) use validate_index;
