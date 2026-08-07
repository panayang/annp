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

/// A fixed random Markov chain of order `k`, sparse enough to have a low
/// entropy rate.
///
/// Order matters more than anything else about this source. At order 1 the
/// current token is a sufficient statistic, so a model that reads only the
/// current token is optimal *by construction* and every additional mechanism
/// can do nothing but add noise — which is exactly what §19 found. At order
/// `k > 1` the current token is provably insufficient, so context is worth
/// something and an architecture built to accumulate it has something to earn.
pub struct MarkovSource {
    vocab: usize,
    order: usize,
    /// Contexts are the last `order` tokens, most recent in the lowest digit.
    contexts: usize,
    successors: Vec<Vec<u32>>,
    probabilities: Vec<Vec<f64>>,
    context: usize,
}

impl MarkovSource {
    pub fn new(vocab: usize, order: usize, fanout: usize, rng: &mut Rng) -> Self {
        assert!(order >= 1, "order must be at least 1");
        assert!(fanout >= 1 && fanout < vocab, "fanout must be a proper subset of the vocabulary");
        let contexts = vocab
            .checked_pow(order as u32)
            .expect("vocab^order overflowed; lower the vocabulary or the order");
        let mut successors = Vec::with_capacity(contexts);
        let mut probabilities = Vec::with_capacity(contexts);
        for _ in 0..contexts {
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
        Self { vocab, order, contexts, successors, probabilities, context: 0 }
    }

    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Context reached by appending `next` and dropping the oldest token.
    #[inline]
    fn shift(&self, context: usize, next: u32) -> usize {
        (context * self.vocab + next as usize) % self.contexts
    }

    pub fn next(&mut self, rng: &mut Rng) -> u32 {
        let u = rng.next_f64();
        let (to, p) = (&self.successors[self.context], &self.probabilities[self.context]);
        let mut acc = 0.0;
        let mut chosen = *to.last().expect("fanout is at least one");
        for (t, w) in to.iter().zip(p) {
            acc += w;
            if u < acc {
                chosen = *t;
                break;
            }
        }
        self.context = self.shift(self.context, chosen);
        chosen
    }

    /// Exact entropy rate in nats: `sum_c pi_c H(p_c)` over contexts.
    pub fn entropy_rate(&self) -> f64 {
        let n = self.contexts;
        let mut pi = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];
        for _ in 0..20_000 {
            next.fill(0.0);
            for (c, ((to, p), mass)) in
                self.successors.iter().zip(&self.probabilities).zip(&pi).enumerate()
            {
                for (t, w) in to.iter().zip(p) {
                    next[self.shift(c, *t)] += mass * w;
                }
            }
            std::mem::swap(&mut pi, &mut next);
        }
        let total: f64 = pi.iter().sum();
        (0..n)
            .map(|c| {
                let h: f64 = self.probabilities[c]
                    .iter()
                    .map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 })
                    .sum();
                pi[c] / total * h
            })
            .sum()
    }
}

/// Prequential counting coder of a given order: predict the next token from
/// counts of what has followed this context so far, then update.
///
/// The absolute yardsticks this bench needs, and two are reported. The
/// **order-1** coder is what is achievable knowing only the current token, so
/// it is the ceiling on any context-free model — including a bypass control.
/// The **order-k** coder matches the source and is the ceiling on anything.
/// The gap between them is the headroom that context provides, and therefore
/// the only part of the task an architecture built to accumulate context can
/// possibly be credited for.
///
/// Both are online and single-pass exactly like everything else here, so their
/// code lengths are valid on the same terms. Krichevsky-Trofimov smoothing (add
/// one half) so an unseen transition is improbable rather than impossible.
struct CountingCoder {
    vocab: usize,
    order: usize,
    contexts: usize,
    counts: Vec<f64>,
    totals: Vec<f64>,
    context: usize,
}

impl CountingCoder {
    fn new(vocab: usize, order: usize) -> Self {
        let contexts = vocab.checked_pow(order as u32).expect("vocab^order overflowed");
        Self {
            vocab,
            order,
            contexts,
            counts: vec![0.0; contexts * vocab],
            totals: vec![0.0; contexts],
            context: 0,
        }
    }

    /// Nats charged for coding `next` from the context seen so far, then the
    /// counts and the context advance.
    fn observe(&mut self, next: u32) -> f64 {
        let alpha = 0.5;
        let (c, n) = (self.context, next as usize);
        let p = (self.counts[c * self.vocab + n] + alpha)
            / (self.totals[c] + alpha * self.vocab as f64);
        self.counts[c * self.vocab + n] += 1.0;
        self.totals[c] += 1.0;
        self.context = (c * self.vocab + n) % self.contexts;
        let _ = self.order;
        -p.ln()
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub order: usize,
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
    /// Give the output head its own weights instead of reusing the embedding
    /// table. Removes the symmetry the tied head imposes.
    pub untied: bool,
    /// Send every token to the same anchor, ignoring content.
    pub constant_ingress: bool,
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
            tied: !cfg.untied,
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
    runtime.set_constant_ingress(cfg.constant_ingress);

    // The source gets its own generator. Drawing it from the one the model
    // construction just used would make the data depend on the architecture:
    // `Ingress::new` consumes a number of draws that varies with `slots`, so a
    // sweep over slot counts silently swept over Markov chains too. The printed
    // entropy rate is the guard — it must not move across a sweep.
    let mut source =
        MarkovSource::new(cfg.vocab, cfg.order, cfg.fanout, &mut Rng::new(cfg.seed ^ 0x50_17_CE_50_17_CE_00));
    let entropy_rate = source.entropy_rate();

    let started = std::time::Instant::now();
    let mut scored: Vec<Scored> = Vec::with_capacity(cfg.tokens);
    let mut stream_rng = Rng::new(cfg.seed ^ 0x5EED);
    let mut coders =
        [CountingCoder::new(cfg.vocab, 1), CountingCoder::new(cfg.vocab, cfg.order)];
    let mut coder_nats = [0.0f64; 2];
    let mut coder_tail = [0.0f64; 2];
    for i in 0..cfg.tokens {
        let token = source.next(&mut stream_rng);
        for (k, coder) in coders.iter_mut().enumerate() {
            let nats = coder.observe(token);
            coder_nats[k] += nats;
            if i * 10 >= cfg.tokens * 9 {
                coder_tail[k] += nats;
            }
        }
        scored.extend(runtime.advance(Some(token)));
    }
    let tail_count = (cfg.tokens - cfg.tokens * 9 / 10).max(1) as f64;
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
        "  vocab={} order={} fanout={} d_head={} slots={} grid={}x{} deg={} floor={}",
        cfg.vocab,
        cfg.order,
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
    println!(
        "  {:<26} {:>8.4}   <- ceiling on any context-free model",
        "order-1 counting coder",
        coder_tail[0] / tail_count * nats_to_bits
    );
    println!(
        "  {:<26} {:>8.4}   <- ceiling on anything",
        format!("order-{} counting coder", source.order()),
        coder_tail[1] / tail_count * nats_to_bits
    );
    println!("  {:<26} {:>8.4}", "source entropy rate", entropy_rate * nats_to_bits);
    println!(
        "  context is worth {:.4} bits/token: everything between the two coders",
        (coder_tail[0] - coder_tail[1]) / tail_count * nats_to_bits
    );
    println!(
        "  prequential total: {:.1} bits over {} tokens   (bigram coder: {:.1})",
        scored.iter().map(|s| s.loss).sum::<f64>() * nats_to_bits,
        scored.len(),
        coder_nats[1] * nats_to_bits
    );
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
        for order in [1usize, 2] {
            let source = MarkovSource::new(16, order, 1, &mut rng);
            assert!(source.entropy_rate() < 1e-12, "order {order}: {}", source.entropy_rate());
        }
    }

    #[test]
    fn entropy_rate_is_bounded_by_the_fanout() {
        let mut rng = Rng::new(2);
        for order in [1usize, 2] {
            for fanout in [2usize, 4, 8] {
                let source = MarkovSource::new(16, order, fanout, &mut rng);
                let h = source.entropy_rate();
                assert!(h > 0.0 && h < (fanout as f64).ln(), "order {order} fanout {fanout}: {h}");
            }
        }
    }

    #[test]
    fn the_source_only_emits_successors_of_its_current_context() {
        let mut rng = Rng::new(3);
        let mut source = MarkovSource::new(16, 2, 3, &mut rng);
        let mut stream = Rng::new(4);
        for _ in 0..5_000 {
            let context = source.context;
            let allowed = source.successors[context].clone();
            let next = source.next(&mut stream);
            assert!(allowed.contains(&next), "context {context} -> {next} is not an edge");
        }
    }

    #[test]
    fn a_higher_order_source_is_not_predictable_from_one_token() {
        // The property that makes this bench able to measure a context-using
        // model at all. At order 2 the same token is followed by different
        // things depending on what preceded it; at order 1 it never is, which
        // is why §19 found the network could only ever add noise.
        let vocab = 16usize;
        let mut rng = Rng::new(5);
        let source = MarkovSource::new(vocab, 2, 3, &mut rng);
        let mut context_dependent = 0;
        for token in 0..vocab {
            let futures: Vec<Vec<u32>> = (0..vocab)
                .map(|earlier| {
                    let mut s = source.successors[earlier * vocab + token].clone();
                    s.sort_unstable();
                    s
                })
                .collect();
            if futures.iter().any(|f| *f != futures[0]) {
                context_dependent += 1;
            }
        }
        assert!(
            context_dependent > vocab / 2,
            "only {context_dependent} of {vocab} tokens have context-dependent futures"
        );
    }

    #[test]
    fn the_order_k_coder_beats_the_order_1_coder_on_a_higher_order_source() {
        // The gap between the two yardsticks is the headroom context provides.
        // If it were not positive, the bench would be back to being unable to
        // measure a context-using model.
        let vocab = 16usize;
        let mut rng = Rng::new(7);
        let mut source = MarkovSource::new(vocab, 2, 3, &mut rng);
        let mut stream = Rng::new(8);
        let (mut c1, mut c2) = (CountingCoder::new(vocab, 1), CountingCoder::new(vocab, 2));
        let (mut n1, mut n2) = (0.0, 0.0);
        for i in 0..60_000 {
            let t = source.next(&mut stream);
            let (a, b) = (c1.observe(t), c2.observe(t));
            if i >= 40_000 {
                n1 += a;
                n2 += b;
            }
        }
        assert!(n2 < n1 * 0.8, "order-2 coder {n2} did not clearly beat order-1 {n1}");
    }
}
