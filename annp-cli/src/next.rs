//! Candidate A end to end: rotation-addressed context, ladder persistence,
//! one shared readout head.
//!
//! The point of this file is the 2×2. `--no-addressing` removes the rotation
//! and `--no-ladder` replaces the ladder with a single exponential decay of the
//! same nominal horizon, so the four arms separate the two factors of
//! `DESIGN-NEXT.md` §4 against each other:
//!
//! ```text
//!                    ladder            single decay
//!   rotation      candidate A        candidate C  (only the ladder removed)
//!   identity      persistence only   plain decaying bag of codes
//! ```
//!
//! The bottom-left cell is the architecture `DESIGN.md` died as: long
//! persistence, no way to address it. Candidate C is not skipped, it is the
//! top-right cell — the arm that answers whether the ladder is worth anything
//! once it sits in a host that can actually use what it retains.
//!
//! Everything except the memory is shared with `baseline.rs`: the same stream,
//! the same `Head`, the same raced learning rates, the same prequential
//! accounting. A difference between the two therefore comes from the memory,
//! which is the only claim this project still has.

use std::path::Path;

use crate::head::Head;
use annp_core::context::{Context, Spacing};
use annp_core::rng::Rng;

pub struct RotaryContext {
    ctx: Context,
    head: Head,
    vocab: usize,
    memory: bool,
    /// `[context state | code of the most recent token]`, width `2d`.
    ///
    /// The second half is a lossless path to the token just seen. Without it
    /// the head had to decode even lag 0 out of a superposition of the whole
    /// history, and the first real-text run came out *worse than the
    /// context-free ceiling* — a head fed a constant learns the marginal and
    /// scores 7.6475, while the same head fed our state scored 8.03. At that
    /// signal-to-noise ratio the state was acting as noise injection, and
    /// every arm was being judged on a handicap that has nothing to do with
    /// what candidate A claims. The baseline reads three tokens losslessly;
    /// this gives one, and everything beyond lag 0 still has to come out of
    /// the addressed memory.
    x: Vec<f64>,
    last: Option<u32>,
}

impl RotaryContext {
    pub fn new(cfg: &Config, rng: &mut Rng) -> Self {
        let (v, d, h) = (cfg.vocab, cfg.d_model, cfg.horizon);
        let ctx = if cfg.addressing {
            Context::new(v, d, h, cfg.ladder, cfg.spacing(), rng)
        } else {
            Context::without_addressing(v, d, h, cfg.ladder, rng)
        };
        let head = Head::new(2 * d, cfg.hidden, v, rng);
        Self {
            ctx,
            head,
            vocab: v,
            memory: cfg.memory,
            x: vec![0.0; 2 * d],
            last: None,
        }
    }

    /// Trained parameters only. The write codes are fixed, so they are state,
    /// not parameters, and are reported separately rather than folded in.
    pub fn parameters(&self) -> usize {
        self.head.parameters()
    }

    pub fn fixed_state(&self) -> usize {
        self.vocab * self.ctx.width() + self.ctx.rungs() * self.ctx.width()
    }

    pub fn rungs(&self) -> usize {
        self.ctx.rungs()
    }

    /// Predict, pay, then write. The write happens after every rate has been
    /// charged, so no rate ever sees the token it is being scored on.
    pub fn observe(&mut self, target: u32, in_tail: bool) -> f64 {
        let d = self.ctx.width();
        if self.memory {
            self.x[..d].copy_from_slice(self.ctx.read());
        }
        match self.last {
            Some(t) => self.x[d..].copy_from_slice(self.ctx.code(t)),
            // Nothing seen yet, which is what a model at the start of a stream
            // legitimately knows.
            None => self.x[d..].fill(0.0),
        }
        let mut best = f64::INFINITY;
        for r in 0..self.head.num_rates() {
            let nats = self.head.step(r, &self.x, target);
            self.head.charge(r, nats, in_tail);
            best = best.min(nats);
        }
        self.ctx.observe(target);
        self.last = Some(target);
        self.head.relax();
        best
    }

    /// Replace the head with one whose weights sit on a consolidation ladder.
    /// Done after construction so the two arms draw identical initial weights.
    pub fn consolidate(&mut self, rungs: Option<usize>, g1: f64, rng: &mut Rng) {
        if let Some(m) = rungs {
            self.head = Head::with_consolidation(
                self.head.in_dim(),
                self.head.hidden(),
                self.vocab,
                Some((m, g1)),
                rng,
            );
        }
    }

    pub fn head_rungs(&self) -> usize {
        self.head.rungs()
    }

    pub fn best(&self) -> (f64, f64) {
        self.head.best()
    }

    pub fn best_tail(&self) -> f64 {
        self.head.best_tail()
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub d_model: usize,
    pub hidden: usize,
    pub horizon: f64,
    pub ladder: bool,
    pub addressing: bool,
    /// Zero the context half of the input, leaving only the lossless current
    /// token. This is the order-1 control realised through our own head, so
    /// the 2x2 measures what the memory adds on top of it rather than what it
    /// adds on top of nothing.
    pub memory: bool,
    pub linear_spacing: bool,
    pub order: usize,
    pub fanout: usize,
    pub seed: u64,
    pub corpus: Option<std::path::PathBuf>,
    pub tokenizer: Option<std::path::PathBuf>,
    /// Above 1, the stream cycles through this many independent Markov chains
    /// in blocks of `domain_span`, and the run reports retention per revisit
    /// instead of a single loss.
    pub domains: usize,
    pub domain_span: usize,
    /// Rungs on the readout head's weight tensors. `None` is plain SGD.
    pub consolidate: Option<usize>,
    /// Rung 1's conductance. Its inverse is the leak time, which must be long
    /// enough to hold what one visit teaches. Defaults to `1 / domain_span`.
    pub consolidate_g1: Option<f64>,
}

impl Config {
    pub fn spacing(&self) -> Spacing {
        if self.linear_spacing {
            Spacing::Linear
        } else {
            Spacing::Geometric
        }
    }
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    crate::write_manifest(out_dir, "next", cfg);
    std::fs::create_dir_all(out_dir)?;
    if cfg.domains > 1 {
        return retention(cfg);
    }
    let mut rng = Rng::new(cfg.seed);

    let stream: Vec<u32> = match (&cfg.corpus, &cfg.tokenizer) {
        (Some(c), Some(t)) => {
            let st = crate::corpus::stream(c, t, cfg.tokens, cfg.vocab)?;
            println!("corpus: {}", c.display());
            println!(
                "  {} documents, {} tokens, {:.2}% in the catch-all id",
                st.documents,
                st.tokens.len(),
                100.0 * st.unknown_share
            );
            st.tokens
        }
        (None, None) => {
            let mut src = crate::run::MarkovSource::new(cfg.vocab, cfg.order, cfg.fanout, &mut rng);
            let mut s = Rng::new(cfg.seed ^ 0x5EED);
            (0..cfg.tokens).map(|_| src.next(&mut s)).collect()
        }
        _ => panic!("--corpus and --tokenizer go together"),
    };

    let mut model = RotaryContext::new(cfg, &mut rng);

    let started = std::time::Instant::now();
    for (i, &tok) in stream.iter().enumerate() {
        model.observe(tok, i * 10 >= stream.len() * 9);
    }
    let elapsed = started.elapsed();
    let nats_to_bits = std::f64::consts::LOG2_E;
    let (total, rate) = model.best();
    let tail_count = (stream.len() / 10).max(1) as f64;

    let arm = if !cfg.memory {
        "current token only     (order-1 through our own head)"
    } else {
        match (cfg.addressing, cfg.ladder) {
            (true, true) => "rotation + ladder      (candidate A)",
            (true, false) => "rotation + decay       (ladder removed)",
            (false, true) => "identity + ladder      (what DESIGN.md died as)",
            (false, false) => "identity + decay       (decaying bag of codes)",
        }
    };

    println!();
    println!("=== {arm} ===");
    println!(
        "  vocab={} d={} hidden={} horizon={} rungs={} spacing={:?}",
        cfg.vocab,
        cfg.d_model,
        cfg.hidden,
        cfg.horizon,
        model.rungs(),
        cfg.spacing()
    );
    println!(
        "  trained parameters={}  fixed state={}",
        model.parameters(),
        model.fixed_state()
    );
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        stream.len() as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("loss, bits per token");
    println!(
        "  {:<30} {:>8.4}   <- best of the raced learning rates ({rate})",
        "last decile",
        model.best_tail() / tail_count * nats_to_bits
    );
    println!(
        "  prequential total: {:.1} bits over {} tokens",
        total * nats_to_bits,
        stream.len()
    );
    Ok(())
}

/// Retention across revisits, which is the claim the ladder was built for.
///
/// The stream cycles through `domains` independent chains in blocks of
/// `domain_span`, so every domain is abandoned and later resumed. What is
/// reported is the cost of the **first tokens after coming back**: a learner
/// that forgot pays full price again, a learner that retained does not. No task
/// label is given, no boundary is announced, nothing is replayed, and the state
/// is bounded — `m` rungs, whatever the stream's length.
///
/// The overall loss is reported next to it and is expected to be *worse* than a
/// plain learner's on a stationary stream. That is not a defeat to be hidden.
/// An architecture that keeps capacity free for distributions it has not met
/// yet cannot also spend all of it on the one in front of it, so the gap is the
/// price of the property, and the point of printing both is to quote the price
/// rather than to pretend there is none.
fn retention(cfg: &Config) -> std::io::Result<()> {
    let mut rng = Rng::new(cfg.seed);
    let (d, span) = (cfg.domains, cfg.domain_span.max(1));
    // Each domain gets its own slice of the alphabet, and this is not a
    // detail. With every domain drawing from the same symbols and the same
    // marginal, the input never says which domain is active, so the model
    // cannot specialise to one; it learns the mixture, the mixture is stable,
    // and there is nothing domain-specific left to forget. Measured: at 3, 8
    // and 16 shared-alphabet domains the plain arm's re-entry cost fell
    // monotonically, which looks like retention but is only convergence to an
    // average. Real non-stationarity announces itself in the content. Disjoint
    // slices are the smallest change that restores that.
    let width = (cfg.vocab / d).max(2);
    let mut sources: Vec<crate::run::MarkovSource> = (0..d)
        .map(|_| crate::run::MarkovSource::new(width, cfg.order, cfg.fanout.min(width - 1), &mut rng))
        .collect();
    let mut draw = Rng::new(cfg.seed ^ 0x5EED);
    let stream: Vec<u32> = (0..cfg.tokens)
        .map(|i| {
            let dom = (i / span) % d;
            sources[dom].next(&mut draw) + (dom * width) as u32
        })
        .collect();

    let mut model = RotaryContext::new(cfg, &mut rng);
    // Derived, not tuned: rung 1 holds a visit's worth of learning.
    let g1 = cfg.consolidate_g1.unwrap_or(1.0 / span as f64);
    model.consolidate(cfg.consolidate, g1, &mut Rng::new(cfg.seed ^ 0xC0FFEE));

    // Two probes per visit, and the *gap* between them is the measurement.
    //
    // Re-entry cost alone cannot be read. It fell monotonically in every
    // shared-alphabet protocol tried, which looks like retention until you
    // notice there is no reference: absolute bits also fall because the model
    // is getting better at everything. What forgetting means is that coming
    // back costs more than staying, so the settled cost at the end of a visit
    // is the null this has to be read against -- the same mistake, and the same
    // fix, as the two wrong null hypotheses in DIAGNOSIS.md section 5.
    //
    // The probe is short on purpose. If relearning takes fifty tokens, a probe
    // spanning a tenth of a four-thousand-token visit measures the relearned
    // state and reports no forgetting whatever happened.
    let probe = (span / 50).clamp(1, 64);
    let visits = cfg.tokens / (span * d) + 1;
    let mut reentry = vec![vec![0.0f64; d]; visits];
    let mut counts = vec![vec![0.0f64; d]; visits];
    let mut settled = vec![vec![0.0f64; d]; visits];
    let mut settled_n = vec![vec![0.0f64; d]; visits];
    let started = std::time::Instant::now();
    for (i, &tok) in stream.iter().enumerate() {
        let nats = model.observe(tok, i * 10 >= stream.len() * 9);
        let dom = (i / span) % d;
        let visit = i / (span * d);
        if visit < visits {
            if i % span < probe {
                reentry[visit][dom] += nats;
                counts[visit][dom] += 1.0;
            } else if i % span >= span - probe {
                settled[visit][dom] += nats;
                settled_n[visit][dom] += 1.0;
            }
        }
    }
    let elapsed = started.elapsed();
    let bits = std::f64::consts::LOG2_E;
    let (total, rate) = model.best();

    println!();
    println!(
        "=== retention over {d} domains, span {span}, {} rungs on the head (g1={g1:.2e}) ===",
        model.head_rungs()
    );
    println!(
        "  vocab={} order={} tokens={} memory={}",
        cfg.vocab, cfg.order, cfg.tokens, cfg.memory
    );
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        stream.len() as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("bits per token over the first and last {probe} tokens of each visit");
    println!("  visit   re-entry   settled       gap  <- the gap is the forgetting");
    for v in 0..visits {
        let (n, m) = (
            counts[v].iter().sum::<f64>(),
            settled_n[v].iter().sum::<f64>(),
        );
        if n < 1.0 || m < 1.0 {
            continue;
        }
        let r = reentry[v].iter().sum::<f64>() / n * bits;
        let s = settled[v].iter().sum::<f64>() / m * bits;
        println!("  {:>5}   {:>8.4}  {:>8.4}  {:>+8.4}", v + 1, r, s, r - s);
    }
    println!();
    println!(
        "  prequential total: {:.1} bits over {} tokens (best rate {rate})",
        total * bits,
        stream.len()
    );
    println!("  <- expected to be worse than a plain learner. That is the price, not a defeat.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream where the answer sits at a fixed lag and nowhere else. The
    /// addressed arm should learn it; the identity arm has no way to tell that
    /// lag from any other and should not.
    #[test]
    fn a_fixed_lag_dependency_is_learned_only_with_addressing() {
        let lag = 4usize;
        let vocab = 6usize;
        let build = |addressing: bool| {
            let mut rng = Rng::new(9);
            let cfg = Config {
                tokens: 0,
                vocab,
                d_model: 64,
                hidden: 32,
                horizon: 128.0,
                ladder: true,
                addressing,
                memory: true,
                domains: 1,
                domain_span: 4000,
                consolidate: None,
                consolidate_g1: None,
                linear_spacing: false,
                order: 2,
                fanout: 3,
                seed: 9,
                corpus: None,
                tokenizer: None,
            };
            let mut m = RotaryContext::new(&cfg, &mut rng);
            let mut s = Rng::new(31);
            let mut past: Vec<u32> = Vec::new();
            let n = 12_000usize;
            let mut tail = 0.0;
            for i in 0..n {
                // Odd positions repeat what was seen `lag` steps ago; even
                // positions are noise, so the marginal stays flat and a
                // context-free model gains nothing.
                let tok = if i % 2 == 1 && past.len() > lag {
                    past[past.len() - lag]
                } else {
                    s.next_below(vocab as u64) as u32
                };
                past.push(tok);
                let nats = m.observe(tok, i * 10 >= n * 9);
                if i * 10 >= n * 9 {
                    tail += nats;
                }
            }
            tail / (n / 10) as f64 * std::f64::consts::LOG2_E
        };
        let addressed = build(true);
        let flat = build(false);
        assert!(
            addressed < flat - 0.15,
            "addressing should pay on a fixed-lag dependency: {addressed} vs {flat}"
        );
    }
}
