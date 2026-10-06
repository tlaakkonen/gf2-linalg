

use gf2_linalg::{Matrix};

fn main() {
    // let m = Matrix::random_invertible(&mut rand::rng(), 20);
    // println!("{:?}", m);

    // let sigma = m.solve(&m.transpose()).unwrap();
    // println!("{:?}", sigma);

    // let primary_spaces = sigma
    //     .minimal_polynomial()
    //     .factor_with_multiplicities()
    //     .into_iter().map(|(p, k)| {
    //         let sp = sigma.eval_poly(&p.pow(k), &Matrix::eye(sigma.num_rows())).null_space();
    //         (p, k, sp)
    //     }).collect::<Vec<_>>();
    
    
    // for (p, k, sp) in &primary_spaces {
    //     println!("p = {:?} p* = {:?} k = {:?}", p, p.reciprocal(), k);
    //     println!("{:?}", sp);
    // }

    // let mut blocks = HashMap::<Poly, (usize, Matrix)>::new();
    // for (p, k, sp) in &primary_spaces {
    //     if blocks.contains_key(p) { continue }
    //     if let Some((_, rsp)) = blocks.get_mut(&p.reciprocal()) {
    //         *rsp = rsp.hconcat(sp);
    //     } else {
    //         blocks.insert(p.clone(), (*k, sp.clone()));
    //     }
    // }
    // let (polys, bases) = blocks.into_iter()
    //     .map(|(p, (k, sp))| ((p, k), sp))
    //     .collect::<(Vec<_>, Vec<_>)>();
    // let block_basis = Matrix::hstack(&bases);

    // let action = block_basis.transpose().dot(&m).dot(&block_basis);

    // println!("{:?}", block_basis);
    // println!("{:?}", action);

    // let f = Poly::new([1, 0, 0, 1, 0, 0, 1]);
    // let gamma = Poly::monom(1).pow_mod(1 << (f.degree() - 1), &f);
    // let xinv = Poly::monom(1).inv_mod(&f).unwrap();
    // let mut terms = HashMap::new();
    // terms.insert(0, gamma.trace_mod(&f));
    // for i in 1..f.degree() {
    //     terms.insert(i as isize, (&gamma * Poly::monom(i)).trace_mod(&f));
    //     terms.insert(-(i as isize), (&gamma * xinv.pow_mod(i, &f)).trace_mod(&f));
    // }
    // let mut b = Matrix::zeros(f.degree(), f.degree());
    // for i in 0..f.degree() {
    //     for j in 0..f.degree() {
    //         b[(i, j)] = terms[&((i as isize) - (j as isize))];
    //     }
    // }
    // println!("{:?}", b);
    // let sigma = b.solve(&b.transpose()).unwrap();
    // println!("{:?}", sigma);

    let m = Matrix::random(&mut rand::rng(), 20, 20);
    let m = Matrix::eye(20) + Matrix::from_diag(&m.diag()) + m.triu();
    let gjf = m.generalized_jordan_form();
    println!("{:?}", gjf.irreducible_factors);
    println!("{:?}", gjf.normal_form);
}