//! End-to-end bench on a synthetic source with a known entropy rate.
//!
//! Real text comes later. What this answers first is whether the assembled
//! system does anything at all, and whether the claims that only make sense
//! once the whole loop is running actually hold:
//!
//! * **Does it learn?** The source is a fixed random Markov chain, so its
//!   entropy rate is computable exactly and the prequential loss has a floor to
//!   be compared against rather than an arbitrary "it went down".
//! * **Two baselines, not one.** Uniform (`ln V`) is the trivial one, but a
//!   tied embedding table starts out predicting the token it was just shown
//!   (DESIGN.md §11.2 ①), so *passthrough* — what a perfectly transparent
//!   network scores — is the bar that means the network contributed something.
//! * **Is per-token compute independent of position?** DESIGN.md §1.6 claims
//!   cost does not grow with sequence length. Node visits are charged per token,
//!   so this is a direct measurement, not an argument.
//! * **Where does the load fall?** Ingress is content-addressed now, so the
//!   visit distribution finally means something (DESIGN.md §10.4 could not say
//!   anything with uniform ingress).

use std::fmt::Write as _;
use std::path::Path;

use annp_core::engine::EngineParams;
use annp_core::graph::{Grid, SmallWorld, Topology};
use annp_core::ladder::Schedule;
use annp_core::model::ModelParams;
use annp_core::node::{AbsorbRule, NodeParams};
use annp_core::rng::Rng;
use annp_core::runtime::{Mode, Runtime, Scored};

/// A fixed random Markov chain, sparse enough to have a low entropy rate.
///
/// Each state moves to one of `fanout` successors with probabilities drawn from
/// a Dirichlet-ish normalisation of uniforms. The entropy rate is the
/// stationary-weighted average of the per-state entropies, computed exactly by
/// power iteration rather than estimated.
pub struct MarkovSource {
    successors: Vec<Vec<u32>>,
    probabilities: Vec<Vec<f64>>,
    state: u32,
}

impl MarkovSource {
    pub fn new(vocab: usize, fanout: usize, rng: &mut Rng) -> Self {
        assert!(fanout >= 1 && fanout < vocab, "fanout must be a proper subset of the vocabulary");
        let mut successors = Vec::with_capacity(vocab);
        let mut probabilities = Vec::with_capacity(vocab);
        for _ in 0..vocab {
            let mut to: Vec<u32> = Vec::with_capacity(fanout);
            while to.len() < fanout {
                let candidate = rng.next_below(vocab as u64) as u32;
                if !to.contains(&candidate) {
                    to.push(candidate);
                }
            }
            let mut p: Vec<f64> = (0..fanout).map(|_| rng.next_f64() + 0.05).collect();
            let total: f64 = p.iter().sum();
            for x in p.iter_mut() {
                *x /= total;
            }
            successors.push(to);
            probabilities.push(p);
        }
        Self { successors, probabilities, state: 0 }
    }

    pub fn next(&mut self, rng: &mut Rng) -> u32 {
        let u = rng.next_f64();
        let (to, p) =
            (&self.successors[self.state as usize], &self.probabilities[self.state as usize]);
        let mut acc = 0.0;
        for (t, w) in to.iter().zip(p) {
            acc += w;
            if u < acc {
                self.state = *t;
                return *t;
            }
        }
        self.state = *to.last().expect("fanout is at least one");
        self.state
    }

    /// Exact entropy rate in nats: `sum_i pi_i H(p_i)`.
    pub fn entropy_rate(&self) -> f64 {
        let n = self.successors.len();
        let mut pi = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];
        // Power iteration. The chain is sparse and random, so this converges
        // fast; a fixed generous budget keeps the result reproducible.
        for _ in 0..10_000 {
            next.fill(0.0);
            for ((to, p), mass) in self.successors.iter().zip(&self.probabilities).zip(&pi) {
                for (t, w) in to.iter().zip(p) {
                    next[*t as usize] += mass * w;
                }
            }
            std::mem::swap(&mut pi, &mut next);
        }
        let total: f64 = pi.iter().sum();
        (0..n)
            .map(|i| {
                let h: f64 = self.probabilities[i]
                    .iter()
                    .map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 })
                    .sum();
                pi[i] / total * h
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub fanout: usize,
    pub d_head: usize,
    pub slots: usize,
    pub grid_side: usize,
    pub long_range: usize,
    pub rungs: usize,
    pub embed_rungs: usize,
    pub mass_floor: f64,
    pub eta: f64,
    pub learning_rate: f64,
    pub ladder_ratio: f64,
    pub seed: u64,
    /// Run the control: score the input embedding instead of the network's
    /// output, everything else identical.
    pub bypass: bool,
    /// Freeze the topology instead of rewiring long-range contacts.
    pub frozen_topology: bool,
    /// Rewire to the first candidate drawn rather than the least-visited.
    pub blind_turnover: bool,
    /// Admit tokens one per tick regardless of what is in flight. Leaks the
    /// future into earlier predictions; only for measuring by how much.
    pub overlapped: bool,
    /// Pin every token to its phase position instead of remembering where its
    /// mass came to rest. The control for the adaptive anchor.
    pub fixed_ingress: bool,
    /// Use the old constant absorb logit. Reproduces the measurements that
    /// condemned it; not for new results.
    pub absorb: AbsorbRule,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Mean over a window given as a fraction of the run, so summaries are
/// comparable across lengths.
fn window(scored: &[Scored], from: f64, to: f64, f: impl Fn(&Scored) -> f64) -> f64 {
    let lo = (scored.len() as f64 * from) as usize;
    let hi = ((scored.len() as f64 * to) as usize).max(lo + 1).min(scored.len());
    mean(&scored[lo..hi].iter().map(f).collect::<Vec<_>>())
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut rng = Rng::new(cfg.seed);
    let schedule = Schedule::Geometric { r: cfg.ladder_ratio, g1: 0.5 };

    let topology = Topology::small_world(
        Grid::new(cfg.grid_side),
        SmallWorld { long_range: cfg.long_range, exponent: 2.0 },
        &mut rng,
    );
    let mut runtime = Runtime::new(
        topology,
        ModelParams {
            vocab: cfg.vocab,
            d_head: cfg.d_head,
            slots: cfg.slots,
            grid_side: cfg.grid_side,
            schedule,
            embed_rungs: cfg.embed_rungs,
            learning_rate: cfg.learning_rate,
        },
        NodeParams {
            absorb: cfg.absorb,
            d_head: cfg.d_head,
            eta: cfg.eta,
            schedule,
            rungs: cfg.rungs,
        },
        EngineParams { mass_floor: cfg.mass_floor, slots: cfg.slots },
        &mut rng,
    );

    runtime.set_bypass(cfg.bypass);
    runtime.set_turnover(!cfg.frozen_topology);
    runtime.set_blind_turnover(cfg.blind_turnover);
    runtime.set_mode(if cfg.overlapped { Mode::Overlapped } else { Mode::Serial });
    runtime.set_adaptive_ingress(!cfg.fixed_ingress);

    // The source gets its own generator. Drawing it from the one the model
    // construction just used would make the data depend on the architecture:
    // `Ingress::new` consumes a number of draws that varies with `slots`, so a
    // sweep over slot counts silently swept over Markov chains too. The printed
    // entropy rate is the guard — it must not move across a sweep.
    let mut source =
        MarkovSource::new(cfg.vocab, cfg.fanout, &mut Rng::new(cfg.seed ^ 0x50_17_CE_50_17_CE_00));
    let entropy_rate = source.entropy_rate();

    let started = std::time::Instant::now();
    let mut scored: Vec<Scored> = Vec::with_capacity(cfg.tokens);
    let mut stream_rng = Rng::new(cfg.seed ^ 0x5EED);
    for _ in 0..cfg.tokens {
        let token = source.next(&mut stream_rng);
        scored.extend(runtime.advance(Some(token)));
    }
    scored.extend(runtime.drain(100_000));
    let elapsed = started.elapsed();

    let uniform = (cfg.vocab as f64).ln();
    let nats_to_bits = 1.0 / std::f64::consts::LN_2;

    println!("run — synthetic Markov source, {} tokens{}", scored.len(),
        if cfg.bypass { "  [BYPASS]" } else { "" });
    println!("  absorb rule: {:?}", cfg.absorb);
    println!("  protocol: {}", if cfg.overlapped {
        "OVERLAPPED — leaks future tokens, loss is NOT a compression bound"
    } else {
        "serial — one token in flight, loss is a valid compression bound"
    });
    println!(
        "  vocab={} fanout={} d_head={} slots={} grid={}x{} deg={} floor={}",
        cfg.vocab,
        cfg.fanout,
        cfg.d_head,
        cfg.slots,
        cfg.grid_side,
        cfg.grid_side,
        4 + cfg.long_range,
        cfg.mass_floor
    );
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        scored.len() as f64 / elapsed.as_secs_f64()
    );
    println!();

    println!("loss, bits per token");
    println!("  {:<26} {:>8}", "uniform (ln V)", uniform * nats_to_bits);
    println!(
        "  {:<26} {:>8.4}",
        "passthrough baseline",
        window(&scored, 0.75, 1.0, |s| s.passthrough_loss) * nats_to_bits
    );
    for (label, from, to) in
        [("network, first decile", 0.0, 0.1), ("network, last decile", 0.9, 1.0)]
    {
        println!(
            "  {:<26} {:>8.4}",
            label,
            window(&scored, from, to, |s| s.loss) * nats_to_bits
        );
    }
    println!("  {:<26} {:>8.4}", "source entropy rate", entropy_rate * nats_to_bits);
    println!("  prequential total: {:.1} bits over {} tokens",
        scored.iter().map(|s| s.loss).sum::<f64>() * nats_to_bits, scored.len());
    println!();

    println!("compute per token (DESIGN.md §1.6: must not grow with position)");
    println!("  {:<26} {:>10} {:>10}", "window", "visits", "mean hops");
    for (label, from, to) in [
        ("first tenth", 0.0, 0.1),
        ("middle tenth", 0.45, 0.55),
        ("last tenth", 0.9, 1.0),
    ] {
        println!(
            "  {:<26} {:>10.2} {:>10.2}",
            label,
            window(&scored, from, to, |s| s.visits as f64),
            window(&scored, from, to, |s| s.mean_hops)
        );
    }
    println!();

    let mut visits = runtime.engine().visits().to_vec();
    let total: u64 = visits.iter().sum();
    visits.sort_unstable();
    let idle = visits.iter().filter(|c| **c == 0).count();
    let decile = visits.len() * 9 / 10;
    println!("load across {} nodes, with content-addressed ingress", visits.len());
    println!("  never visited      {idle} ({:.1}%)", 100.0 * idle as f64 / visits.len() as f64);
    println!(
        "  busiest decile     {:.1}% of all visits (10% would be uniform)",
        100.0 * visits[decile..].iter().sum::<u64>() as f64 / total as f64
    );
    println!("  busiest single     {:.2}% of all visits", 100.0 * visits[visits.len() - 1] as f64 / total as f64);
    println!();

    println!("topology");
    println!(
        "  {}",
        if cfg.frozen_topology { "frozen".to_string() } else { format!("{} rewirings", runtime.rewirings()) }
    );
    println!();

    let (known, distinct, mean_move) = runtime.model().ingress().drift();
    println!("ingress anchors ({})", if cfg.fixed_ingress { "fixed at phase position" } else { "readout of resting place" });
    println!("  tokens with a remembered anchor  {known} / {}", cfg.vocab);
    println!("  distinct cells occupied          {distinct}");
    println!("  mean move per observation        {mean_move:.2} (torus L1)");
    println!("  anchor movement over the run (a settling map decays, a jittering one does not)");
    for (label, from, to) in [
        ("first tenth", 0.0, 0.1),
        ("middle tenth", 0.45, 0.55),
        ("last tenth", 0.9, 1.0),
    ] {
        println!("    {:<16} {:>6.2}", label, window(&scored, from, to, |s| s.anchor_move as f64));
    }
    println!();

    let mut csv = String::from("position,token,target,loss_nats,passthrough_nats,visits,mean_hops\n");
    for s in &scored {
        let _ = writeln!(
            csv,
            "{},{},{},{:.6},{:.6},{},{:.4}",
            s.position, s.token, s.target, s.loss, s.passthrough_loss, s.visits, s.mean_hops
        );
    }
    let path = out_dir.join("run.csv");
    std::fs::write(&path, csv)?;
    println!("  wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_deterministic_chain_has_zero_entropy_rate() {
        let mut rng = Rng::new(1);
        let source = MarkovSource::new(64, 1, &mut rng);
        assert!(source.entropy_rate() < 1e-12, "{}", source.entropy_rate());
    }

    #[test]
    fn entropy_rate_is_bounded_by_the_fanout() {
        // A state with `fanout` successors cannot exceed ln(fanout) nats, and a
        // chain of such states cannot exceed it on average either.
        let mut rng = Rng::new(2);
        for fanout in [2usize, 4, 8] {
            let source = MarkovSource::new(64, fanout, &mut rng);
            let h = source.entropy_rate();
            assert!(h > 0.0 && h < (fanout as f64).ln(), "fanout {fanout}: {h}");
        }
    }

    #[test]
    fn the_source_only_emits_declared_successors() {
        let mut rng = Rng::new(3);
    let mut source = MarkovSource::new(32, 3, &mut rng);
        let mut stream = Rng::new(4);
        let mut state = source.state;
        for _ in 0..5_000 {
            let allowed = source.successors[state as usize].clone();
            let next = source.next(&mut stream);
            assert!(allowed.contains(&next), "{state} -> {next} is not an edge");
            state = next;
        }
    }
}
