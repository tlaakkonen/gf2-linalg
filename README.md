# `gf2-linalg`

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: light)" srcset="./logo.svg#light">
    <source media="(prefers-color-scheme: dark)" srcset="./logo.svg#dark">
    <img src="./logo.svg#light">
    </picture>
</p>

`gf2-linalg` is a Rust crate for linear algebra over the binary finite field. It aims to be a comprehensive and easy to use tool for manipulating matrices and univariate polynomials over this field.

> [!WARNING]
> This library is under active development and is not yet feature complete! If there's something not on the roadmap that you would like to see, feel free to file a feature request.

The goals of the project are to be:
* **Comprehensive**: While some other similar libraries contain most of the common linear algebraic operations, this library aims to cover as many possible. If there is a binary matrix decomposition or polynomial manipulation that you think is important but isn't in this library, then I consider that a bug, and you should file an issue! (PRs also welcome.)
* **Easy to use**: Despite being written in Rust, this library aims to be easy to use and avoid the need for its users to be intimately familiar with lifetimes and the borrow checker. Consequently, we work mostly with owned values and avoid things like matrix views.
* **Self-contained**: Pure Rust implementation to make building and linking as easy as possible. Minimal dependencies to avoid version conflicts and keep compile times fast. So far the only dependency is `rand` and it's optional.

I also have some explicit anti-goals:
* **Extremely high performance**: While I'm not trying to write slow code, I'm not worrying about performance at the moment. For example, matrices are currently implemented without SIMD. If you need to work with 10000x10000 matrices, you may want to look elsewhere. There are plenty of fast Rust matrix libraries.
* **Other finite fields**: This crate focuses on the binary finite field, and the special structure that this affords. There is no plan to support arbitrary finite fields, consider the `galois` Python package if you need that.
* **Objects except matrices and univariate polynomials**: I am not planning to support objects like polynomial rings, ideals, etc. Most of the operations on these objects can be phrased in terms of polynomials and matrices, so consider this if you want to work with these objects (or consider using SageMath instead). I am not currently planning to support multivariate polynomials, but I may reconsider.

## Roadmap

**General Features**

Currently supported:

Planned:
* Python bindings
* `ndarray` interface

**Matrix Operations**

Currently supported:
* Multiplication, general utilities
* Random generation methods
* Gaussian elimination
* (P)L(D)U decompositions and linear solvers
* Rank calculation and decomposition
* The Frobenius / rational normal form
* Characteristic and minimal polynomials, maximal vectors
* Krylov subspaces

Planned:
* Markov-Patel-Hayes' algorithm
* Lempel's symmetric factorization
* Block-wise decompositions

**Polynomial Operations**

Currently supported:
* Arithmetic, Euclidean division
* Extended GCD and Bezout's algorithm
* Differentiation
* Square-free factorization
* Factorization into irreducibles
* Irreducibility testing

Planned:
* Berlekamp-Massey

## Examples


