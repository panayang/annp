//! The readout head, written exactly once.
//!
//! Both the learned baseline and the rotation-addressed context read through
//! this, so a difference between them cannot come from the head: same layer
//! sizes, same initialisation, same raced learning rates, same update order.
//! `DESIGN-NEXT.md` invariant ③ is about comparing against something external;
//! this is the other half of it, making sure the comparison isolates the part
//! under test.
//!
//! It is also the only thing in the project that learns by backpropagation.
//! `DESIGN-NEXT.md` §6 settled that fork: the ladder and the rotation need no
//! gradient, so the ban costs us nothing except on the readout, and the readout
//! is the part with the least to claim.

use annp_core::ladder::{Ladder, Schedule};
use annp_core::rng::Rng;

/// One weight tensor, held either flat or on a consolidation ladder.
///
/// When consolidated, the live weights *are* rung 1 — gradients enter there and
/// the forward pass reads there, exactly as `ladder.rs` describes. The deeper
/// rungs are pure state. Nothing about the update rule changes; the diffusion
/// is the only addition, and it is the same diffusion, written once, in
/// `ladder.rs`.
///
/// This is Benna–Fusi put where Benna–Fusi belongs. `context.rs` hangs a ladder
/// on the *activation*, which buys a long context and was measured doing badly
/// at it. E0 and E2 measured a ladder on *weights*, and a ladder on weights is
/// the thing that resists overwriting: a one-off update is pulled back by rung
/// 2 and vanishes, while an update repeated across a domain's visit pushes rung
/// 2 the same way each time and survives into it. When the next domain
/// overwrites rung 1, the old domain is still held below and pulls rung 1 back.
/// Repetition is the filter, it needs no task label, and it is free.
enum Weights {
    /// `[rate][..]`
    Plain(Vec<Vec<f64>>),
    /// `[rate]`, rung 1 is live.
    Consolidated(Vec<Ladder>),
}

impl Weights {
    fn new(
        per_rate: Vec<Vec<f64>>,
        rows: usize,
        cols: usize,
        consolidation: Option<(usize, f64)>,
    ) -> Self {
        match consolidation {
            None => Weights::Plain(per_rate),
            Some((m, g1)) => Weights::Consolidated(
                per_rate
                    .into_iter()
                    .map(|init| {
                        // `g1` is not `ladder.rs`'s default. That default was
                        // picked for one-shot delta-rule writes at eta = 1,
                        // where a write is large and arrives once. Gradients are
                        // small and noisy, so rung 1's leak time `1/g1` has to
                        // be long enough to hold what a visit teaches, or
                        // consolidation washes the learning out before it can
                        // consolidate anything. Measured: at g1 = 0.25, tau_1 is
                        // four tokens and the consolidated arm learns almost
                        // nothing.
                        let mut l = Ladder::new(Schedule::Geometric { r: 2.0, g1 }, m, rows, cols);
                        // Every rung, not just rung 1. The chain conserves its
                        // capacity-weighted total, so seeding rung 1 alone
                        // would let diffusion dilute the initialisation by
                        // `1 / sum_k C_k` before any real gradient arrived.
                        for rung in l.rungs_mut() {
                            rung.as_mut_slice().copy_from_slice(&init);
                        }
                        l
                    })
                    .collect(),
            ),
        }
    }

    #[inline]
    fn live(&mut self, r: usize) -> &mut [f64] {
        match self {
            Weights::Plain(w) => &mut w[r],
            Weights::Consolidated(l) => l[r].rungs_mut()[0].as_mut_slice(),
        }
    }

    fn relax(&mut self) {
        if let Weights::Consolidated(ls) = self {
            for l in ls.iter_mut() {
                l.relax();
            }
        }
    }

    fn rungs(&self) -> usize {
        match self {
            Weights::Plain(_) => 1,
            Weights::Consolidated(l) => l[0].num_rungs(),
        }
    }

    #[cfg(test)]
    fn truncate(&mut self, n: usize) {
        match self {
            Weights::Plain(w) => w.truncate(n),
            Weights::Consolidated(l) => l.truncate(n),
        }
    }
}

/// `in_dim -> hidden (tanh) -> vocab`, trained online, several learning rates
/// raced in parallel with the best reported.
pub struct Head {
    pub(crate) in_dim: usize,
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    pub(crate) rates: Vec<f64>,
    /// `[rate][h * in_dim + k]`
    w1: Weights,
    pub(crate) b1: Vec<Vec<f64>>,
    /// `[rate][token * hidden + h]`
    w2: Weights,
    pub(crate) b2: Vec<Vec<f64>>,
    /// Accumulated code length per rate, in nats, and over the final tenth.
    pub(crate) nats: Vec<f64>,
    pub(crate) tail: Vec<f64>,
    /// `[rate][decile]`, so a learning curve can be read per rate rather than
    /// only a final number. A model still improving steeply at the end has not
    /// lost, it has run out of stream.
    deciles: Vec<Vec<f64>>,
    decile_n: Vec<f64>,
    h: Vec<f64>,
    logits: Vec<f64>,
    gh: Vec<f64>,
    /// `dL/dx` for the last `step`, taken against the pre-update `w1`. A front
    /// end with learned inputs applies this; one with fixed codes ignores it.
    pub(crate) gx: Vec<f64>,
    /// Hidden units are split into this many contiguous groups and one group is
    /// active per token. `1` is the dense head.
    ///
    /// This is the whole of candidate B, and it is small because the ladder is
    /// already elementwise: separation does not need a ladder per expert, it
    /// needs different domains to touch different weights. Routing is what
    /// supplies that. `DESIGN-NEXT.md` §10.3 measured consolidation making
    /// forgetting three times worse in a dense layer and explained it as
    /// consolidation preserving whatever repeats -- which across interleaved
    /// domains is the mixture. If that explanation is right, the ladder should
    /// pay here and only here, and the prediction is an interaction, not a main
    /// effect.
    experts: usize,
    /// `[expert][in_dim]`, fixed unit vectors. Routing is content-addressed and
    /// unlearned: no gradient enters the gate, so the memory pathway keeps the
    /// property that only the readout learns.
    gate: Vec<f64>,
    /// How often each expert won, so a collapsed gate is visible before any
    /// loss is read. A gate that always picks one expert makes the routed arm a
    /// dense arm with fewer units, and the comparison would mean nothing.
    expert_use: Vec<f64>,
    last_expert: usize,
    gate_width: usize,
}

impl Head {
    pub fn new(in_dim: usize, hidden: usize, vocab: usize, rng: &mut Rng) -> Self {
        Self::with_consolidation(in_dim, hidden, vocab, None, rng)
    }

    /// `rungs = Some(m)` puts both weight tensors on an `m`-rung consolidation
    /// ladder. Biases stay flat: they are `hidden + vocab` numbers against
    /// `hidden * in_dim + vocab * hidden`, so consolidating them would cost
    /// bookkeeping and change nothing measurable.
    pub fn with_consolidation(
        in_dim: usize,
        hidden: usize,
        vocab: usize,
        consolidation: Option<(usize, f64)>,
        rng: &mut Rng,
    ) -> Self {
        Self::routed(in_dim, hidden, vocab, consolidation, 1, rng)
    }

    pub fn routed(
        in_dim: usize,
        hidden: usize,
        vocab: usize,
        consolidation: Option<(usize, f64)>,
        experts: usize,
        rng: &mut Rng,
    ) -> Self {
        assert!(in_dim >= 1 && hidden >= 1 && vocab >= 1);
        assert!(
            experts >= 1 && hidden.is_multiple_of(experts),
            "hidden must divide by experts"
        );
        let rates = vec![0.003, 0.01, 0.03, 0.1];
        let n = rates.len();
        let draw = |count: usize, sigma: f64, rng: &mut Rng| -> Vec<f64> {
            (0..count).map(|_| rng.next_normal() * sigma).collect()
        };
        let w1 = Weights::new(
            (0..n)
                .map(|_| draw(hidden * in_dim, 1.0 / (in_dim as f64).sqrt(), rng))
                .collect(),
            hidden,
            in_dim,
            consolidation,
        );
        let w2 = Weights::new(
            (0..n)
                .map(|_| draw(vocab * hidden, 1.0 / (hidden as f64).sqrt(), rng))
                .collect(),
            vocab,
            hidden,
            consolidation,
        );
        Self {
            in_dim,
            hidden,
            vocab,
            w1,
            b1: vec![vec![0.0; hidden]; n],
            w2,
            b2: vec![vec![0.0; vocab]; n],
            nats: vec![0.0; n],
            tail: vec![0.0; n],
            deciles: vec![vec![0.0; 10]; n],
            decile_n: vec![0.0; 10],
            h: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            gh: vec![0.0; hidden],
            gx: vec![0.0; in_dim],
            rates,
            experts,
            gate: {
                let mut g = vec![0.0; experts * in_dim];
                for e in 0..experts {
                    let row = &mut g[e * in_dim..(e + 1) * in_dim];
                    rng.fill_unit_vector(row);
                }
                g
            },
            expert_use: vec![0.0; experts],
            last_expert: 0,
            gate_width: in_dim,
        }
    }

    /// Share of tokens each expert won, most-used first.
    pub fn expert_shares(&self) -> Vec<f64> {
        let total: f64 = self.expert_use.iter().sum::<f64>().max(1.0);
        let mut s: Vec<f64> = self.expert_use.iter().map(|u| u / total).collect();
        s.sort_by(|a, b| b.total_cmp(a));
        s
    }

    /// Which expert the last `step` routed to.
    pub fn last_expert(&self) -> usize {
        self.last_expert
    }

    /// Restrict the gate to the leading `w` inputs.
    pub fn set_gate_width(&mut self, w: usize) {
        assert!(w >= 1 && w <= self.in_dim);
        self.gate_width = w;
    }

    /// Which expert this input routes to. Deterministic, unlearned, content
    /// addressed: the nearest of `experts` fixed directions.
    fn route(&self, x: &[f64]) -> usize {
        if self.experts == 1 {
            return 0;
        }
        let n = self.in_dim;
        // `gate_width` lets the gate read only the leading part of the input.
        // With the state first and the current token's code second, reading the
        // state alone routes on something that varies slowly and carries the
        // domain, instead of on a code that changes every token and scatters a
        // single domain across every expert. Measured on the full input:
        // expert-domain purity 0.253 against a chance of 0.125, so the
        // separation the claim needs was barely there.
        let w = self.gate_width.min(n);
        (0..self.experts)
            .map(|e| {
                let row = &self.gate[e * n..e * n + w];
                row.iter().zip(&x[..w]).map(|(a, b)| a * b).sum::<f64>()
            })
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(e, _)| e)
            .unwrap()
    }

    pub fn num_rates(&self) -> usize {
        self.rates.len()
    }

    pub fn rate(&self, r: usize) -> f64 {
        self.rates[r]
    }

    pub fn rungs(&self) -> usize {
        self.w2.rungs()
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Collapse to a single rate with a chosen learning rate, so the
    /// finite-difference check can freeze the model (`lr = 0`) to read a loss
    /// and unfreeze it (`lr = 1`) to read the step the gradient produced.
    #[cfg(test)]
    pub(crate) fn keep_one_rate(&mut self, lr: f64) {
        self.rates = vec![lr];
        self.w1.truncate(1);
        self.w2.truncate(1);
        self.b1.truncate(1);
        self.b1[0].iter_mut().for_each(|b| *b = 0.0);
        self.b2.truncate(1);
        self.b2[0].iter_mut().for_each(|b| *b = 0.0);
        self.nats = vec![0.0];
        self.tail = vec![0.0];
    }

    #[cfg(test)]
    pub(crate) fn set_rate(&mut self, lr: f64) {
        self.rates[0] = lr;
    }

    /// Live weights for one rate, so the finite-difference check can poke them
    /// without caring whether they sit on a ladder.
    #[cfg(test)]
    pub(crate) fn w1_mut(&mut self, r: usize) -> &mut [f64] {
        self.w1.live(r)
    }

    #[cfg(test)]
    pub(crate) fn w2_mut(&mut self, r: usize) -> &mut [f64] {
        self.w2.live(r)
    }

    /// One diffusion step for every rate. Call once per token, after every rate
    /// has been charged and updated.
    pub fn relax(&mut self) {
        self.w1.relax();
        self.w2.relax();
    }

    /// Parameters in the head alone. A front end adds its own.
    pub fn parameters(&self) -> usize {
        self.hidden * self.in_dim + self.hidden + self.vocab * self.hidden + self.vocab
    }

    /// Charges `target` against the current weights for rate `r`, then updates
    /// and leaves `dL/dx` in `gx`. Returns nats.
    pub fn step(&mut self, r: usize, x: &[f64], target: u32) -> f64 {
        assert_eq!(x.len(), self.in_dim, "head: input width mismatch");
        let (hid, v, n_in) = (self.hidden, self.vocab, self.in_dim);
        let expert = self.route(x);
        let group = hid / self.experts;
        let (lo, hi) = (expert * group, expert * group + group);
        if r == 0 {
            self.expert_use[expert] += 1.0;
        }
        self.last_expert = expert;
        self.h.iter_mut().for_each(|v| *v = 0.0);
        {
            let w1 = self.w1.live(r);
            for j in lo..hi {
                let row = &w1[j * n_in..(j + 1) * n_in];
                let z: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + self.b1[r][j];
                self.h[j] = z.tanh();
            }
        }
        {
            let w2 = self.w2.live(r);
            for n in 0..v {
                let row = &w2[n * hid + lo..n * hid + hi];
                self.logits[n] = row
                    .iter()
                    .zip(&self.h[lo..hi])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    + self.b2[r][n];
            }
        }
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

        // Backward. `logits` becomes the output error in place.
        let lr = self.rates[r];
        self.logits[target as usize] -= 1.0;
        self.gh.iter_mut().for_each(|g| *g = 0.0);
        {
            let (w2, h, gh) = (self.w2.live(r), &self.h, &mut self.gh);
            for n in 0..v {
                let g = self.logits[n];
                if g == 0.0 {
                    continue;
                }
                let row = &mut w2[n * hid + lo..n * hid + hi];
                for (j, (weight, &hv)) in row.iter_mut().zip(&h[lo..hi]).enumerate() {
                    gh[lo + j] += g * *weight;
                    *weight -= lr * g * hv;
                }
            }
        }
        for n in 0..v {
            let g = self.logits[n];
            if g != 0.0 {
                self.b2[r][n] -= lr * g;
            }
        }
        self.gx.iter_mut().for_each(|g| *g = 0.0);
        {
            let (w1, h, gh, gx) = (self.w1.live(r), &self.h, &self.gh, &mut self.gx);
            for j in lo..hi {
                let gz = gh[j] * (1.0 - h[j] * h[j]);
                if gz == 0.0 {
                    continue;
                }
                let row = &mut w1[j * n_in..(j + 1) * n_in];
                for (k, (weight, &xv)) in row.iter_mut().zip(x).enumerate() {
                    // Read the weight into `gx` before it moves, so the
                    // reported input gradient belongs to this forward pass.
                    gx[k] += gz * *weight;
                    *weight -= lr * gz * xv;
                }
            }
        }
        for j in lo..hi {
            let gz = self.gh[j] * (1.0 - self.h[j] * self.h[j]);
            if gz != 0.0 {
                self.b1[r][j] -= lr * gz;
            }
        }
        nats
    }

    pub fn charge(&mut self, r: usize, nats: f64, in_tail: bool, decile: usize) {
        self.nats[r] += nats;
        if in_tail {
            self.tail[r] += nats;
        }
        let d = decile.min(9);
        self.deciles[r][d] += nats;
        if r == 0 {
            self.decile_n[d] += 1.0;
        }
    }

    /// Bits per token per decile for the rate with the lowest total.
    pub fn best_curve(&self) -> Vec<f64> {
        let mut best = (f64::INFINITY, 0usize);
        for (r, &n) in self.nats.iter().enumerate() {
            if n < best.0 {
                best = (n, r);
            }
        }
        self.deciles[best.1]
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

    /// True when the winning rate is an endpoint of the swept set.
    pub fn rate_at_boundary(&self) -> bool {
        let (_, r) = self.best();
        r == self.rates[0] || r == self.rates[self.rates.len() - 1]
    }

    /// Best accumulated code length across rates, and which rate won.
    pub fn best(&self) -> (f64, f64) {
        let mut best = (f64::INFINITY, self.rates[0]);
        for (r, &n) in self.nats.iter().enumerate() {
            if n < best.0 {
                best = (n, self.rates[r]);
            }
        }
        best
    }

    pub fn best_tail(&self) -> f64 {
        self.tail.iter().copied().fold(f64::INFINITY, f64::min)
    }
}
