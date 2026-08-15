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
    norm_state: f64,
    norm_code: f64,
    norm_n: f64,
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
            norm_state: 0.0,
            norm_code: 0.0,
            norm_n: 0.0,
        }
    }

    /// Trained parameters only. The write codes are fixed, so they are state,
    /// not parameters, and are reported separately rather than folded in.
    pub fn parameters(&self) -> usize {
        self.head.parameters()
    }

    /// Predict, pay, then write. The write happens after every rate has been
    /// charged, so no rate ever sees the token it is being scored on.
    pub fn observe_at(&mut self, target: u32, in_tail: bool, decile: usize) -> f64 {
        let d = self.ctx.width();
        if self.memory {
            self.x[..d].copy_from_slice(self.ctx.read());
            // Normalise the state half to match the code half.
            //
            // Both halves go through one `w1` under one learning rate. Measured
            // unnormalised: the state's mean norm is 10.797 against the code's
            // 1.000, and the rate that survives the race drops from 0.1 to 0.01
            // -- the same factor of ten. The clean order-1 pathway was being
            // run an order of magnitude slower because it shared an optimiser
            // with a block whose gradients were ten times larger, and the loss
            // rose across deciles instead of falling. "Memory is worse than no
            // memory" was measuring that, not measuring memory.
            let n = self.x[..d].iter().map(|v| v * v).sum::<f64>().sqrt();
            if n > f64::MIN_POSITIVE {
                self.x[..d].iter_mut().for_each(|v| *v /= n);
            }
        }
        match self.last {
            Some(t) => self.x[d..].copy_from_slice(self.ctx.code(t)),
            // Nothing seen yet, which is what a model at the start of a stream
            // legitimately knows.
            None => self.x[d..].fill(0.0),
        }
        // The two halves are fed through one w1 with one learning rate. If the
        // state half carries a much larger norm than the clean code half, its
        // gradients dominate and the rate that survives is set by the noisy
        // half -- which would slow the clean half down by the same factor and
        // look exactly like "memory hurts".
        let (a, b) = self.x.split_at(d);
        self.norm_state += a.iter().map(|v| v * v).sum::<f64>().sqrt();
        self.norm_code += b.iter().map(|v| v * v).sum::<f64>().sqrt();
        self.norm_n += 1.0;
        let mut best = f64::INFINITY;
        for r in 0..self.head.num_rates() {
            let nats = self.head.step(r, &self.x, target);
            self.head.charge(r, nats, in_tail, decile);
            best = best.min(nats);
        }
        self.ctx.observe(target);
        self.last = Some(target);
        self.head.relax();
        best
    }

    /// Replace the head with one whose weights sit on a consolidation ladder.
    /// Done after construction so the two arms draw identical initial weights.
    /// Rebuild the head with consolidation and/or routing. Done after
    /// construction so every arm of the 2x2 draws the same initial weights from
    /// the same point in the stream.
    pub fn rebuild_head(
        &mut self,
        rungs: Option<usize>,
        g1: f64,
        experts: usize,
        gate_on_state: bool,
        rng: &mut Rng,
    ) {
        if rungs.is_none() && experts == 1 {
            return;
        }
        self.head = Head::routed(
            self.head.in_dim(),
            self.head.hidden(),
            self.vocab,
            rungs.map(|m| (m, g1)),
            experts,
            rng,
        );
        if gate_on_state {
            self.head.set_gate_width(self.ctx.width());
        }
    }

    pub fn expert_shares(&self) -> Vec<f64> {
        self.head.expert_shares()
    }

    pub fn last_expert(&self) -> usize {
        self.head.last_expert()
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

    pub fn mean_norms(&self) -> (f64, f64) {
        (self.norm_state / self.norm_n, self.norm_code / self.norm_n)
    }

    pub fn best_curve(&self) -> Vec<f64> {
        self.head.best_curve()
    }
}

/// A single associative node: one ladder-backed matrix, and the ladder is the
/// learner.
///
/// Stripped to the bone on purpose. `DESIGN-NEXT.md` §9–§11 put five mechanisms
/// into one design — rotation, ladder persistence, concatenated input, routing,
/// gating — and every one of them generated its own confound. Nothing here but
/// a write rule and a ladder.
///
/// The write is the delta rule with its prediction passed through the softmax:
///
/// ```text
///     W += eta * (onehot(y) - softmax(W k)) k^T
/// ```
///
/// which is simultaneously a nonlinear local write and the exact cross-entropy
/// gradient of a linear softmax layer. So "nonlinear write" and "no
/// backpropagation" are the same statement here rather than two assumptions.
///
/// Rungs are deliberately more than the horizon asks for. The chain is closed
/// and cheap in the forward pass — only rung 1 is read — so extra depth is
/// redundancy held against timescales we have not thought of, and the cost is
/// memory rather than compute.
pub struct LadderNode {
    /// One independent memory per learning rate, raced like the head's.
    mem: Vec<annp_core::ladder::AssocMemory>,
    etas: Vec<f64>,
    nats: Vec<f64>,
    tail: Vec<f64>,
    deciles: Vec<Vec<f64>>,
    decile_n: Vec<f64>,
    logits: Vec<f64>,
    resid: Vec<f64>,
    vocab: usize,
    in_dim: usize,
    rungs: usize,
}

impl LadderNode {
    pub fn new(vocab: usize, in_dim: usize, rungs: usize, g1: f64) -> Self {
        let etas = vec![0.003, 0.01, 0.03, 0.1];
        let schedule = annp_core::ladder::Schedule::Geometric { r: 2.0, g1 };
        Self {
            mem: (0..etas.len())
                .map(|_| {
                    annp_core::ladder::AssocMemory::ladder_rect(vocab, in_dim, schedule, rungs)
                })
                .collect(),
            nats: vec![0.0; etas.len()],
            tail: vec![0.0; etas.len()],
            deciles: vec![vec![0.0; 10]; etas.len()],
            decile_n: vec![0.0; 10],
            logits: vec![0.0; vocab],
            resid: vec![0.0; vocab],
            etas,
            vocab,
            in_dim,
            rungs,
        }
    }

    pub fn parameters(&self) -> usize {
        self.vocab * self.in_dim
    }

    pub fn rungs(&self) -> usize {
        self.rungs
    }

    pub fn observe_at(&mut self, key: &[f64], target: u32, in_tail: bool, decile: usize) -> f64 {
        let mut best = f64::INFINITY;
        for e in 0..self.etas.len() {
            self.mem[e].read().mul_vec(key, &mut self.logits);
            let peak = self
                .logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for l in self.logits.iter_mut() {
                *l = (*l - peak).exp();
                total += *l;
            }
            for l in self.logits.iter_mut() {
                *l /= total;
            }
            let nats = -self.logits[target as usize].max(f64::MIN_POSITIVE).ln();
            // residual = onehot(target) - softmax
            for (r, p) in self.resid.iter_mut().zip(&self.logits) {
                *r = -p;
            }
            self.resid[target as usize] += 1.0;
            self.mem[e].inject(&self.resid, key, self.etas[e]);
            self.mem[e].relax();

            self.nats[e] += nats;
            if in_tail {
                self.tail[e] += nats;
            }
            let d = decile.min(9);
            self.deciles[e][d] += nats;
            if e == 0 {
                self.decile_n[d] += 1.0;
            }
            best = best.min(nats);
        }
        best
    }

    pub fn best(&self) -> (f64, f64) {
        let mut b = (f64::INFINITY, self.etas[0]);
        for (e, &n) in self.nats.iter().enumerate() {
            if n < b.0 {
                b = (n, self.etas[e]);
            }
        }
        b
    }

    pub fn best_tail(&self) -> f64 {
        self.tail.iter().copied().fold(f64::INFINITY, f64::min)
    }

    pub fn best_curve(&self) -> Vec<f64> {
        let mut b = (f64::INFINITY, 0usize);
        for (e, &n) in self.nats.iter().enumerate() {
            if n < b.0 {
                b = (n, e);
            }
        }
        self.deciles[b.1]
            .iter()
            .zip(&self.decile_n)
            .map(|(s, n)| {
                if *n > 0.0 {
                    s / n * std::f64::consts::LOG2_E
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

/// Either arm, behind one interface, so the protocol and the metrics cannot
/// differ between them by accident.
enum Arm {
    Node(Box<LadderNode>, Vec<f64>, usize),
    Ffn(Box<RotaryContext>),
}

impl Arm {
    fn build(cfg: &Config, rng: &mut Rng) -> Self {
        if cfg.node {
            let d = cfg.d_model;
            let mut codes = vec![0.0; cfg.vocab * d];
            for tok in 0..cfg.vocab {
                rng.fill_unit_vector(&mut codes[tok * d..(tok + 1) * d]);
            }
            let g1 = 1.0 / cfg.domain_span.max(1) as f64;
            Arm::Node(
                Box::new(LadderNode::new(cfg.vocab, d, cfg.node_rungs, g1)),
                codes,
                d,
            )
        } else {
            let mut m = RotaryContext::new(cfg, rng);
            let g1 = cfg
                .consolidate_g1
                .unwrap_or(1.0 / cfg.domain_span.max(1) as f64);
            m.rebuild_head(
                cfg.consolidate,
                g1,
                cfg.experts,
                cfg.gate_on_state,
                &mut Rng::new(cfg.seed ^ 0xC0FFEE),
            );
            Arm::Ffn(Box::new(m))
        }
    }

    fn observe_at(&mut self, tok: u32, in_tail: bool, decile: usize, prev: Option<u32>) -> f64 {
        match self {
            Arm::Node(n, codes, d) => {
                let key: Vec<f64> = match prev {
                    Some(p) => codes[p as usize * *d..(p as usize + 1) * *d].to_vec(),
                    None => vec![0.0; *d],
                };
                n.observe_at(&key, tok, in_tail, decile)
            }
            Arm::Ffn(m) => m.observe_at(tok, in_tail, decile),
        }
    }

    fn label(&self, cfg: &Config) -> String {
        match self {
            Arm::Node(n, _, _) => format!(
                "single ladder node, {} rungs, {} parameters",
                n.rungs(),
                n.parameters()
            ),
            Arm::Ffn(m) => format!(
                "ffn head, {} rungs on the weights, {} experts, {} parameters",
                m.head_rungs(),
                cfg.experts,
                m.parameters()
            ),
        }
    }

    fn best(&self) -> (f64, f64) {
        match self {
            Arm::Node(n, _, _) => n.best(),
            Arm::Ffn(m) => m.best(),
        }
    }

    fn best_curve(&self) -> Vec<f64> {
        match self {
            Arm::Node(n, _, _) => n.best_curve(),
            Arm::Ffn(m) => m.best_curve(),
        }
    }

    fn best_tail(&self) -> f64 {
        match self {
            Arm::Node(n, _, _) => n.best_tail(),
            Arm::Ffn(m) => m.best_tail(),
        }
    }

    /// Which expert the last token used; the node arm has exactly one.
    fn last_expert(&self) -> usize {
        match self {
            Arm::Node(..) => 0,
            Arm::Ffn(m) => m.last_expert(),
        }
    }

    /// Mean norms of the two input halves. Printed on every run so an
    /// imbalance like the one that invalidated §10.1 is visible at the time
    /// rather than discovered a day later.
    fn mean_norms(&self) -> (f64, f64) {
        match self {
            Arm::Node(..) => (0.0, 1.0),
            Arm::Ffn(m) => m.mean_norms(),
        }
    }

    fn expert_shares(&self) -> Vec<f64> {
        match self {
            Arm::Node(..) => vec![1.0],
            Arm::Ffn(m) => m.expert_shares(),
        }
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
    /// Hidden units split into this many content-routed groups. 1 is dense.
    pub experts: usize,
    /// Route on the state half only, not on the current token's code.
    pub gate_on_state: bool,
    /// How many strides wide each domain's alphabet window is. 1.0 is disjoint
    /// windows, `domains` is a fully shared alphabet, between the two they
    /// overlap.
    pub domain_width: f64,
    /// Run the single ladder node instead of the FFN head. The three arms of
    /// the stripped-down comparison are `--node`, `--consolidate m`, and
    /// neither.
    pub node: bool,
    pub node_rungs: usize,
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

    let mut model = Arm::build(cfg, &mut rng);

    let started = std::time::Instant::now();
    let mut prev: Option<u32> = None;
    for (i, &tok) in stream.iter().enumerate() {
        model.observe_at(tok, i * 10 >= stream.len() * 9, i * 10 / stream.len(), prev);
        prev = Some(tok);
    }
    let elapsed = started.elapsed();
    let nats_to_bits = std::f64::consts::LOG2_E;
    let (total, rate) = model.best();
    let tail_count = (stream.len() / 10).max(1) as f64;

    let arm_name = model.label(cfg);
    let arm = if cfg.node {
        "single ladder node    (nonlinear local write, no backprop)"
    } else if !cfg.memory {
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
    println!("  {arm_name}");
    println!(
        "  vocab={} d={} hidden={} horizon={}",
        cfg.vocab, cfg.d_model, cfg.hidden, cfg.horizon
    );
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        stream.len() as f64 / elapsed.as_secs_f64()
    );
    let (ns, nc) = model.mean_norms();
    println!("  mean input norm: state {ns:.3}, current-token code {nc:.3}");
    println!();
    println!("loss, bits per token");
    println!(
        "  {:<30} {:>8.4}   <- best of the raced learning rates ({rate})",
        "last decile",
        model.best_tail() / tail_count * nats_to_bits
    );
    println!(
        "  learning curve, bits per token by decile:\n    {}",
        model
            .best_curve()
            .iter()
            .map(|b| format!("{b:.3}"))
            .collect::<Vec<_>>()
            .join("  ")
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
    // Overlapping alphabet windows, and the overlap is the whole point.
    //
    // The hypothesis needs two things at once and the first three protocols
    // each supplied exactly one. A shared alphabet gives interference, so
    // forgetting happens, but every domain looks identical from its content and
    // routing separates nothing -- measured expert-domain purity 0.157 against
    // a chance of 0.125. Disjoint alphabets give a perfect content signature
    // and no interference, so eight domains coexist and there is nothing to
    // forget. Sliding windows that overlap give both: the overlap is where
    // domains collide, the rest is what tells them apart.
    let stride = (cfg.vocab / d).max(1);
    let width = ((stride as f64 * cfg.domain_width).round() as usize).clamp(2, cfg.vocab);
    let mut sources: Vec<crate::run::MarkovSource> = (0..d)
        .map(|_| {
            crate::run::MarkovSource::new(width, cfg.order, cfg.fanout.min(width - 1), &mut rng)
        })
        .collect();
    let mut draw = Rng::new(cfg.seed ^ 0x5EED);
    let stream: Vec<u32> = (0..cfg.tokens)
        .map(|i| {
            let dom = (i / span) % d;
            ((sources[dom].next(&mut draw) as usize + dom * stride) % cfg.vocab) as u32
        })
        .collect();

    let mut model = Arm::build(cfg, &mut rng);

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
    // Expert x domain. A spread gate is not the same as a gate that separates
    // *domains*, and only the second is what the hypothesis needs. Checking the
    // first and not the second is how a run can look like a refutation while
    // never having tested the claim.
    let mut joint = vec![vec![0.0f64; d]; cfg.experts];
    let started = std::time::Instant::now();
    let mut prev: Option<u32> = None;
    for (i, &tok) in stream.iter().enumerate() {
        let nats = model.observe_at(tok, i * 10 >= stream.len() * 9, i * 10 / stream.len(), prev);
        prev = Some(tok);
        let dom = (i / span) % d;
        joint[model.last_expert()][dom] += 1.0;
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
        "=== retention over {d} domains, span {span} ===\n  {}",
        model.label(cfg)
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
    // Self-check before any loss is read. A gate that always picks the same
    // expert turns the routed arm into a narrower dense arm, and the whole
    // comparison would be meaningless without this being visible.
    let shares = model.expert_shares();
    println!(
        "  expert usage, most-used first: {}",
        shares
            .iter()
            .take(8)
            .map(|s| format!("{:.3}", s))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if cfg.experts > 1 && shares[0] > 0.9 {
        println!("  !! gate collapsed onto one expert -- routed arm is not routed");
    }
    if cfg.experts > 1 {
        // Mean over experts of the largest domain share inside that expert.
        // 1/domains means the expert sees every domain equally and separates
        // nothing; 1.0 means each expert belongs to one domain.
        let purity: f64 = joint
            .iter()
            .filter(|row| row.iter().sum::<f64>() > 0.0)
            .map(|row| {
                let tot: f64 = row.iter().sum();
                row.iter().fold(0.0f64, |m, v| m.max(*v)) / tot
            })
            .sum::<f64>()
            / joint.iter().filter(|r| r.iter().sum::<f64>() > 0.0).count() as f64;
        println!(
            "  expert-domain purity: {purity:.3}  (chance {:.3}, 1.000 = one domain per expert)",
            1.0 / d as f64
        );
        if purity < 2.0 / d as f64 {
            println!("  !! routing is not separating domains -- this run cannot test the claim");
        }
    }
    println!();
    println!(
        "  alphabet: {width} symbols per domain, stride {stride}, {} shared with the neighbour",
        width.saturating_sub(stride)
    );
    println!();
    println!("bits per token over the first and last {probe} tokens of each visit");
    println!("  visit   re-entry   settled       gap  <- read the trajectory, not the last row");
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
                experts: 1,
                gate_on_state: false,
                domain_width: 1.0,
                node: false,
                node_rungs: 8,
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
                let nats = m.observe_at(tok, i * 10 >= n * 9, 0);
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
