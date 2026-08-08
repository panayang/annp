//! Topology bench — is the long-range exponent derived or just asserted?
//!
//! DESIGN.md §1.9 puts long-range contacts on a distance-decaying law and
//! claims `alpha = 2` follows from the lattice being 2-D rather than from
//! tuning. That claim has a sharp, checkable form (Kleinberg 2000): with one
//! long-range contact per node, decentralised greedy routing takes
//!
//! ```text
//!   alpha < 2 :  Omega(n^((2-alpha)/3))
//!   alpha = 2 :  O(log^2 n)              <- the only polylogarithmic point
//!   alpha > 2 :  Omega(n^((alpha-2)/(alpha-1)))
//! ```
//!
//! Those are in the lattice **side** `n`, not the node count `N = n^2`, so a
//! `hops ~ N^beta` fit must halve every exponent. They are also *lower* bounds
//! on any decentralised algorithm, so greedy is allowed to sit above them and
//! only the `alpha = 0` end, where greedy happens to be near-optimal, is a
//! tight check on the implementation.
//!
//! So we do not merely look for a minimum at 2: we fit `hops ~ N^beta` across
//! grid sizes and compare. A minimum in the right place with the wrong scaling
//! would mean we got the right answer for the wrong reason.
//!
//! The theorem is stated for exactly one long-range contact, so the sweep runs
//! at `long_range = 1`. A separate pass shows what more contacts buy at
//! `alpha = 2`, which is a constant factor, not a change of regime.

use std::fmt::Write as _;
use std::path::Path;

use annp_core::graph::{Grid, SmallWorld, Topology};
use annp_core::linalg::linear_fit;
use annp_core::rng::Rng;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Config {
    pub sides: Vec<usize>,
    pub exponents: Vec<f64>,
    pub contacts: Vec<usize>,
    pub pairs: u64,
    pub seed: u64,
}

/// Kleinberg's lower bound on delivery time, re-expressed as the exponent of
/// `N = n^2` rather than of the lattice side `n` — hence the factor of two.
/// `None` at the critical point, where growth is polylogarithmic and no power
/// of `N` applies.
fn predicted_beta(alpha: f64) -> Option<f64> {
    const DIM: f64 = 2.0;
    let in_side = if (alpha - DIM).abs() < 1e-9 {
        return None;
    } else if alpha < DIM {
        (DIM - alpha) / 3.0
    } else {
        (alpha - DIM) / (alpha - 1.0)
    };
    Some(in_side / DIM)
}

fn mean_hops(side: usize, spec: SmallWorld, pairs: u64, seed: u64) -> f64 {
    let mut build = Rng::new(seed);
    let t = Topology::small_world(Grid::new(side), spec, &mut build);
    let n = t.grid().len() as u64;
    // A separate generator for the source/target pairs, so changing `pairs`
    // cannot alter the graph being measured.
    let mut pick = Rng::new(seed ^ 0x51_7C_C1_B7_27_22_0A_95);
    let mut total = 0.0;
    for _ in 0..pairs {
        let a = pick.next_below(n) as u32;
        let b = pick.next_below(n) as u32;
        total += t.greedy_hops(a, b) as f64;
    }
    total / pairs as f64
}

struct Row {
    exponent: f64,
    contacts: usize,
    side: usize,
    nodes: usize,
    hops: f64,
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;

    let mut jobs: Vec<(f64, usize, usize)> = Vec::new();
    for &exponent in &cfg.exponents {
        for &side in &cfg.sides {
            jobs.push((exponent, 1, side));
        }
    }
    for &contacts in &cfg.contacts {
        if contacts != 1 {
            for &side in &cfg.sides {
                jobs.push((2.0, contacts, side));
            }
        }
    }

    let rows: Vec<Row> = jobs
        .par_iter()
        .map(|&(exponent, contacts, side)| {
            let spec = SmallWorld {
                long_range: contacts,
                exponent,
            };
            Row {
                exponent,
                contacts,
                side,
                nodes: side * side,
                hops: mean_hops(side, spec, cfg.pairs, cfg.seed),
            }
        })
        .collect();

    println!("topology — decentralised greedy routing on a 2-D torus");
    println!(
        "  sides={:?} pairs/config={} seed={}",
        cfg.sides, cfg.pairs, cfg.seed
    );
    println!();

    println!("mean greedy hops, one long-range contact per node");
    print!("  {:<10}", "alpha");
    for &side in &cfg.sides {
        print!("{:>12}", format!("N={}", side * side));
    }
    println!("{:>10}{:>12}", "beta", "predicted");
    for &exponent in &cfg.exponents {
        print!("  {exponent:<10.1}");
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for &side in &cfg.sides {
            let r = rows
                .iter()
                .find(|r| r.exponent == exponent && r.contacts == 1 && r.side == side)
                .expect("every job produced a row");
            print!("{:>12.2}", r.hops);
            xs.push((r.nodes as f64).ln());
            ys.push(r.hops.ln());
        }
        let beta = linear_fit(&xs, &ys).0;
        let pred = predicted_beta(exponent)
            .map(|b| format!("{b:.3}"))
            .unwrap_or_else(|| "polylog".to_string());
        println!("{beta:>10.3}{pred:>12}");
    }
    println!("  beta fits hops ~ N^beta. Predictions are lower bounds on any");
    println!("  decentralised algorithm, so measured >= predicted is expected.");
    println!();

    // At the critical exponent, hops should be linear in (ln N)^2.
    let critical: Vec<&Row> = rows
        .iter()
        .filter(|r| r.exponent == 2.0 && r.contacts == 1)
        .collect();
    if critical.len() >= 2 {
        println!("alpha = 2: hops against (ln N)^2");
        for r in &critical {
            let l2 = (r.nodes as f64).ln().powi(2);
            println!(
                "  N={:<8} hops={:<8.2} hops/(ln N)^2 = {:.4}",
                r.nodes,
                r.hops,
                r.hops / l2
            );
        }
        println!("  a constant ratio is the polylogarithmic regime; a rising one is not.");
        println!();
    }

    if cfg.contacts.len() > 1 {
        println!("alpha = 2: what more contacts buy");
        print!("  {:<10}", "contacts");
        for &side in &cfg.sides {
            print!("{:>12}", format!("N={}", side * side));
        }
        println!();
        for &contacts in &cfg.contacts {
            print!("  {contacts:<10}");
            for &side in &cfg.sides {
                let r = rows
                    .iter()
                    .find(|r| r.exponent == 2.0 && r.contacts == contacts && r.side == side);
                match r {
                    Some(r) => print!("{:>12.2}", r.hops),
                    None => print!("{:>12}", "-"),
                }
            }
            println!();
        }
        println!();
    }

    let mut csv = String::from("exponent,contacts,side,nodes,mean_greedy_hops\n");
    for r in &rows {
        let _ = writeln!(
            csv,
            "{},{},{},{},{:.6}",
            r.exponent, r.contacts, r.side, r.nodes, r.hops
        );
    }
    let path = out_dir.join("topology.csv");
    std::fs::write(&path, csv)?;
    println!("  wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_exponents_match_the_published_formulas() {
        // Published in the lattice side; halved here because we fit against N.
        assert!((predicted_beta(0.0).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((predicted_beta(1.0).unwrap() - 1.0 / 6.0).abs() < 1e-12);
        assert!(
            predicted_beta(2.0).is_none(),
            "the critical point has no power law"
        );
        assert!((predicted_beta(3.0).unwrap() - 0.25).abs() < 1e-12);
        assert!((predicted_beta(4.0).unwrap() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn beta_is_smallest_at_the_lattice_dimension() {
        // Whatever else the sweep shows, the prediction itself must have its
        // minimum at alpha = 2, or the comparison is meaningless.
        for alpha in [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 4.0] {
            assert!(predicted_beta(alpha).unwrap() > 0.0, "alpha={alpha}");
        }
    }

    #[test]
    fn more_contacts_never_lengthen_a_greedy_route() {
        // Adding out-edges can only widen the choice greedy makes at each step.
        let sparse = mean_hops(
            24,
            SmallWorld {
                long_range: 1,
                exponent: 2.0,
            },
            400,
            4,
        );
        let dense = mean_hops(
            24,
            SmallWorld {
                long_range: 8,
                exponent: 2.0,
            },
            400,
            4,
        );
        assert!(dense < sparse, "sparse {sparse} vs dense {dense}");
    }
}
