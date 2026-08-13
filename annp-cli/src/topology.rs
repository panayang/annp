//! Topology bench — is the long-range exponent derived or just asserted?
//!
//! DESIGN.md §1.9 puts long-range contacts on a distance-decaying law and
//! claims `alpha` follows from the lattice dimension rather than from tuning.
//! The lattice is a ring now (§36), so the claim is `alpha = 1`, and it has a
//! sharp checkable form (Kleinberg 2000): with one long-range contact per node,
//! decentralised greedy routing on a `d`-dimensional lattice of `N` nodes takes
//!
//! ```text
//!   alpha < d :  Omega(N^((d-alpha)/(d+1)))
//!   alpha = d :  O(log^2 N)              <- the only polylogarithmic point
//!   alpha > d :  Omega(N^((alpha-d)/alpha))
//! ```
//!
//! On a ring `N` *is* the lattice extent, so unlike the 2-D case there is no
//! factor to undo between "exponent in the side" and "exponent in the node
//! count". These are also *lower* bounds on any decentralised algorithm, so
//! greedy is allowed to sit above them; only the `alpha = 0` end, where greedy
//! happens to be near-optimal, is a tight check on the implementation.
//!
//! So we do not merely look for a minimum at 1: we fit `hops ~ N^beta` across
//! ring sizes and compare. A minimum in the right place with the wrong scaling
//! would mean the right answer for the wrong reason.
//!
//! What this bench is *for* has changed with the lattice. On a torus the
//! long-range law governed spatial distance, which under cursor ingress is only
//! loosely related to stream lag. On a ring, distance is lag exactly, so this
//! same law is what makes a single hop a delay of scale-free length — and the
//! hop count measured here is the hop count for reaching that far back.
//!
//! The theorem is stated for exactly one long-range contact, so the sweep runs
//! at `long_range = 1`. A separate pass shows what more contacts buy at the
//! critical exponent, which is a constant factor, not a change of regime.

use std::fmt::Write as _;
use std::path::Path;

use annp_core::graph::{Ring, SmallWorld, Topology};
use annp_core::linalg::linear_fit;
use annp_core::rng::Rng;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Config {
    /// Ring lengths to sweep. These are node counts, not lattice sides.
    pub sizes: Vec<usize>,
    pub exponents: Vec<f64>,
    pub contacts: Vec<usize>,
    pub pairs: u64,
    pub seed: u64,
}

/// Kleinberg's lower bound on delivery time as an exponent of `N`. `None` at
/// the critical point, where growth is polylogarithmic and no power applies.
///
/// One dimension, so `N` is the lattice extent and the exponent needs no
/// conversion — the 2-D version of this function divided by the dimension to
/// turn an exponent in the side into one in the node count, and keeping that
/// division here would have quietly halved every prediction.
fn predicted_beta(alpha: f64) -> Option<f64> {
    const DIM: f64 = 1.0;
    if (alpha - DIM).abs() < 1e-9 {
        None
    } else if alpha < DIM {
        Some((DIM - alpha) / (DIM + 1.0))
    } else {
        Some((alpha - DIM) / alpha)
    }
}

fn mean_hops(nodes: usize, spec: SmallWorld, pairs: u64, seed: u64) -> f64 {
    let mut build = Rng::new(seed);
    let t = Topology::small_world(Ring::new(nodes), spec, &mut build);
    let n = t.ring().len() as u64;
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
    size: usize,
    nodes: usize,
    hops: f64,
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    crate::write_manifest(out_dir, "topology", cfg);
    std::fs::create_dir_all(out_dir)?;

    let mut jobs: Vec<(f64, usize, usize)> = Vec::new();
    for &exponent in &cfg.exponents {
        for &size in &cfg.sizes {
            jobs.push((exponent, 1, size));
        }
    }
    for &contacts in &cfg.contacts {
        if contacts != 1 {
            for &size in &cfg.sizes {
                jobs.push((1.0, contacts, size));
            }
        }
    }

    let rows: Vec<Row> = jobs
        .par_iter()
        .map(|&(exponent, contacts, size)| {
            let spec = SmallWorld {
                long_range: contacts,
                exponent,
            };
            Row {
                exponent,
                contacts,
                size,
                nodes: size,
                hops: mean_hops(size, spec, cfg.pairs, cfg.seed),
            }
        })
        .collect();

    println!("topology — decentralised greedy routing on a ring");
    println!(
        "  sizes={:?} pairs/config={} seed={}",
        cfg.sizes, cfg.pairs, cfg.seed
    );
    println!();

    println!("mean greedy hops, one long-range contact per node");
    print!("  {:<10}", "alpha");
    for &size in &cfg.sizes {
        print!("{:>12}", format!("N={size}"));
    }
    println!("{:>10}{:>12}", "beta", "predicted");
    for &exponent in &cfg.exponents {
        print!("  {exponent:<10.1}");
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for &size in &cfg.sizes {
            let r = rows
                .iter()
                .find(|r| r.exponent == exponent && r.contacts == 1 && r.size == size)
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
        .filter(|r| r.exponent == 1.0 && r.contacts == 1)
        .collect();
    if critical.len() >= 2 {
        println!("alpha = 1: hops against (ln N)^2");
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
        println!("alpha = 1: what more contacts buy");
        print!("  {:<10}", "contacts");
        for &size in &cfg.sizes {
            print!("{:>12}", format!("N={size}"));
        }
        println!();
        for &contacts in &cfg.contacts {
            print!("  {contacts:<10}");
            for &size in &cfg.sizes {
                let r = rows
                    .iter()
                    .find(|r| r.exponent == 1.0 && r.contacts == contacts && r.size == size);
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
            r.exponent, r.contacts, r.size, r.nodes, r.hops
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
        // One dimension, so N is the lattice extent and the published exponents
        // apply directly. The 2-D version of this test halved them, which would
        // now be wrong by a factor of two in every entry.
        assert!((predicted_beta(0.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((predicted_beta(0.5).unwrap() - 0.25).abs() < 1e-12);
        assert!(
            predicted_beta(1.0).is_none(),
            "the critical point has no power law"
        );
        assert!((predicted_beta(2.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((predicted_beta(4.0).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn the_prediction_bottoms_out_at_the_lattice_dimension() {
        // Whatever the sweep shows, the prediction itself has to have its
        // minimum where the lattice dimension is, or the comparison it is meant
        // to support says nothing. Approaching alpha = 1 from either side must
        // drive the exponent toward zero.
        let near = 1e-6;
        assert!(predicted_beta(1.0 - near).unwrap() < 1e-6);
        assert!(predicted_beta(1.0 + near).unwrap() < 1e-6);
        for alpha in [0.0, 0.5, 1.5, 2.5, 3.0, 4.0] {
            assert!(predicted_beta(alpha).unwrap() > 0.0, "alpha={alpha}");
        }
    }

    #[test]
    fn more_contacts_never_lengthen_a_greedy_route() {
        // Adding out-edges can only widen the choice greedy makes at each step.
        let at = |long_range| {
            mean_hops(
                512,
                SmallWorld {
                    long_range,
                    exponent: 1.0,
                },
                400,
                4,
            )
        };
        let (sparse, dense) = (at(1), at(8));
        assert!(dense < sparse, "sparse {sparse} vs dense {dense}");
    }
}
