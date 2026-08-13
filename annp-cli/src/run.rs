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

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use annp_core::engine::EngineParams;
use annp_core::graph::{Grid, SmallWorld, Topology};
use annp_core::ladder::Schedule;
use annp_core::model::{HeadKind, IngressMode, ModelParams};
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
    /// Contexts are the last `order` tokens, most recent in the lowest digit.
    contexts: usize,
    successors: Vec<Vec<u32>>,
    probabilities: Vec<Vec<f64>>,
    context: usize,
}

impl MarkovSource {
    pub fn new(vocab: usize, order: usize, fanout: usize, rng: &mut Rng) -> Self {
        assert!(order >= 1, "order must be at least 1");
        assert!(
            fanout >= 1 && fanout < vocab,
            "fanout must be a proper subset of the vocabulary"
        );
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
        Self {
            vocab,
            contexts,
            successors,
            probabilities,
            context: 0,
        }
    }

    /// Context reached by appending `next` and dropping the oldest token.
    #[inline]
    fn shift(&self, context: usize, next: u32) -> usize {
        (context * self.vocab + next as usize) % self.contexts
    }

    pub fn next(&mut self, rng: &mut Rng) -> u32 {
        let u = rng.next_f64();
        let (to, p) = (
            &self.successors[self.context],
            &self.probabilities[self.context],
        );
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
            for (c, ((to, p), mass)) in self
                .successors
                .iter()
                .zip(&self.probabilities)
                .zip(&pi)
                .enumerate()
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
        let contexts = vocab
            .checked_pow(order as u32)
            .expect("vocab^order overflowed");
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

/// Log-linear coder: `logits(next) = A[current] + B[previous]`.
///
/// The ceiling on any model whose two sources of evidence combine *additively*
/// in log space. That is the shape this architecture has: the reassembled
/// vector is a sum of particle contributions and the output head is linear, so
/// whatever the network retrieves about earlier tokens enters the logits as a
/// separate term rather than as an interaction with the current token.
///
/// A random order-k chain needs interactions — `P(next | cur, prev)` is a
/// general function of the pair, not a product of marginals — so if the network
/// lands near this coder rather than near the order-k coder, the gap is
/// structural and no additional mechanism of the same kind will close it.
///
/// Several learning rates run in parallel and the best is reported, because the
/// point is what additive models *can* do, not what one optimiser run does.
struct AdditiveCoder {
    vocab: usize,
    rates: Vec<f64>,
    /// `[rate][current * vocab + next]` and `[rate][previous * vocab + next]`.
    current: Vec<Vec<f64>>,
    previous: Vec<Vec<f64>>,
    nats: Vec<f64>,
    tail: Vec<f64>,
    last: Option<u32>,
    scratch: Vec<f64>,
}

impl AdditiveCoder {
    fn new(vocab: usize) -> Self {
        let rates = vec![0.02, 0.05, 0.1, 0.3];
        let n = rates.len();
        Self {
            vocab,
            current: vec![vec![0.0; vocab * vocab]; n],
            previous: vec![vec![0.0; vocab * vocab]; n],
            nats: vec![0.0; n],
            tail: vec![0.0; n],
            rates,
            last: None,
            scratch: vec![0.0; vocab],
        }
    }

    fn observe(&mut self, current: u32, next: u32, in_tail: bool) {
        let Some(previous) = self.last else {
            self.last = Some(current);
            return;
        };
        let v = self.vocab;
        for r in 0..self.rates.len() {
            let (ci, pi) = (current as usize * v, previous as usize * v);
            for n in 0..v {
                self.scratch[n] = self.current[r][ci + n] + self.previous[r][pi + n];
            }
            let peak = self
                .scratch
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for x in self.scratch.iter_mut() {
                *x = (*x - peak).exp();
                total += *x;
            }
            for x in self.scratch.iter_mut() {
                *x /= total;
            }
            let nats = -self.scratch[next as usize].max(f64::MIN_POSITIVE).ln();
            self.nats[r] += nats;
            if in_tail {
                self.tail[r] += nats;
            }
            self.scratch[next as usize] -= 1.0;
            let lr = self.rates[r];
            for n in 0..v {
                let g = lr * self.scratch[n];
                self.current[r][ci + n] -= g;
                self.previous[r][pi + n] -= g;
            }
        }
        self.last = Some(current);
    }

    /// Best tail loss across the learning rates, in nats.
    fn best_tail(&self) -> f64 {
        self.tail.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Online linear probe: how well does a representation linearly predict a
/// discrete target, scored prequentially like everything else here.
///
/// The point is localisation, not performance. Probing for the **previous**
/// token says where context information exists along the path; probing for the
/// **current** token in parallel is the control that separates "context never
/// arrives" from "the path is destroying information indiscriminately", which
/// have completely different fixes.
///
/// Several learning rates run at once and the best is reported, because the
/// question is what is linearly present, not what one optimiser run recovers.
struct Probe {
    vocab: usize,
    width: usize,
    rates: Vec<f64>,
    weights: Vec<Vec<f64>>,
    tail: Vec<f64>,
    scratch: Vec<f64>,
}

impl Probe {
    fn new(vocab: usize, width: usize) -> Self {
        let rates = vec![0.01, 0.03, 0.1];
        let n = rates.len();
        Self {
            vocab,
            width,
            weights: vec![vec![0.0; vocab * width]; n],
            tail: vec![0.0; n],
            rates,
            scratch: vec![0.0; vocab],
        }
    }

    fn observe(&mut self, x: &[f64], target: u32, in_tail: bool) {
        debug_assert_eq!(x.len(), self.width);
        for r in 0..self.rates.len() {
            for n in 0..self.vocab {
                let row = &self.weights[r][n * self.width..(n + 1) * self.width];
                self.scratch[n] = row.iter().zip(x).map(|(w, v)| w * v).sum();
            }
            let peak = self
                .scratch
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for v in self.scratch.iter_mut() {
                *v = (*v - peak).exp();
                total += *v;
            }
            for v in self.scratch.iter_mut() {
                *v /= total;
            }
            if in_tail {
                self.tail[r] += -self.scratch[target as usize].max(f64::MIN_POSITIVE).ln();
            }
            self.scratch[target as usize] -= 1.0;
            let lr = self.rates[r];
            for n in 0..self.vocab {
                let g = lr * self.scratch[n];
                if g != 0.0 {
                    let row = &mut self.weights[r][n * self.width..(n + 1) * self.width];
                    for (w, v) in row.iter_mut().zip(x) {
                        *w -= g * v;
                    }
                }
            }
        }
    }

    fn best_tail(&self) -> f64 {
        self.tail.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Most likely target under one of the rates. Used to read a symbol back
    /// out of a representation rather than to score it.
    fn argmax(&self, x: &[f64], rate: usize) -> u32 {
        let mut best = (0usize, f64::NEG_INFINITY);
        for n in 0..self.vocab {
            let row = &self.weights[rate][n * self.width..(n + 1) * self.width];
            let score: f64 = row.iter().zip(x).map(|(w, v)| w * v).sum();
            if score > best.1 {
                best = (n, score);
            }
        }
        best.0 as u32
    }

    fn rate_count(&self) -> usize {
        self.rates.len()
    }
}

/// Counting coder over `(current, decoded previous)`, where the previous token
/// is read back out of the assembled vector by a linear probe.
///
/// The measurement the trained probes cannot give. A one-hidden-layer probe
/// scored on 100k samples in a single online pass is just another weak learner,
/// so its loss is a *lower* bound on what a readout could reach, not a ceiling —
/// which is why a 1.95-bit improvement in how well the assembled vector encodes
/// the previous token moved it by 0.05 bits.
///
/// This has nothing to train. It decodes the pair from the representation and
/// then looks the answer up. If it approaches the order-k coder, everything the
/// prediction needs is already in the representation and only the readout is at
/// fault; if it does not, "linearly decodable" and "usable" are different things
/// and the problem is upstream.
struct OracleCoder {
    vocab: usize,
    counts: Vec<Vec<f64>>,
    totals: Vec<Vec<f64>>,
    tail: Vec<f64>,
    hits: Vec<u64>,
    seen: u64,
    scratch: Vec<f64>,
}

impl OracleCoder {
    fn new(vocab: usize, rates: usize) -> Self {
        Self {
            vocab,
            counts: vec![vec![0.0; vocab * vocab * vocab]; rates],
            totals: vec![vec![0.0; vocab * vocab]; rates],
            tail: vec![0.0; rates],
            hits: vec![0; rates],
            seen: 0,
            scratch: vec![0.0; vocab],
        }
    }

    fn observe(&mut self, current: u32, decoded: &[u32], truth: u32, next: u32, in_tail: bool) {
        let v = self.vocab;
        self.seen += 1;
        for (r, &prev) in decoded.iter().enumerate() {
            if prev == truth {
                self.hits[r] += 1;
            }
            let ctx = current as usize * v + prev as usize;
            let alpha = 0.5;
            let p = (self.counts[r][ctx * v + next as usize] + alpha)
                / (self.totals[r][ctx] + alpha * v as f64);
            if in_tail {
                self.tail[r] += -p.ln();
            }
            self.counts[r][ctx * v + next as usize] += 1.0;
            self.totals[r][ctx] += 1.0;
        }
        let _ = &self.scratch;
    }

    /// Best tail loss, and the decoding accuracy of the rate that achieved it.
    fn best(&self) -> (f64, f64) {
        let (r, tail) = self
            .tail
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(r, t)| (r, *t))
            .expect("at least one rate");
        (tail, self.hits[r] as f64 / self.seen.max(1) as f64)
    }
}

/// One-hidden-layer probe, for measuring what a *nonlinear* readout of a
/// representation could achieve.
///
/// Paired with the linear `Probe` on the same representation and the same
/// target, it splits the gap between the model's actual loss and what its
/// input allows into two parts that need opposite fixes: the linear probe minus
/// the model is what better *optimisation* of the existing head would recover,
/// and the nonlinear probe minus the linear probe is what only a different
/// *shape* of readout can reach.
struct NonlinearProbe {
    vocab: usize,
    width: usize,
    hidden: usize,
    rates: Vec<f64>,
    w1: Vec<Vec<f64>>,
    w2: Vec<Vec<f64>>,
    tail: Vec<f64>,
    z: Vec<f64>,
    h: Vec<f64>,
    o: Vec<f64>,
    gh: Vec<f64>,
}

impl NonlinearProbe {
    fn new(vocab: usize, width: usize, hidden: usize, rng: &mut Rng) -> Self {
        let rates = vec![0.003, 0.01, 0.03, 0.1];
        let n = rates.len();
        // One shared initialisation across rates so they differ only in step
        // size; zero second layer so the probe starts at the uniform prediction.
        let sigma = 1.0 / (width as f64).sqrt();
        let init: Vec<f64> = (0..hidden * width)
            .map(|_| rng.next_normal() * sigma)
            .collect();
        Self {
            vocab,
            width,
            hidden,
            w1: vec![init; n],
            w2: vec![vec![0.0; vocab * hidden]; n],
            tail: vec![0.0; n],
            rates,
            z: vec![0.0; hidden],
            h: vec![0.0; hidden],
            o: vec![0.0; vocab],
            gh: vec![0.0; hidden],
        }
    }

    fn observe(&mut self, x: &[f64], target: u32, in_tail: bool) {
        debug_assert_eq!(x.len(), self.width);
        for r in 0..self.rates.len() {
            for j in 0..self.hidden {
                let row = &self.w1[r][j * self.width..(j + 1) * self.width];
                self.z[j] = row.iter().zip(x).map(|(w, v)| w * v).sum();
                self.h[j] = self.z[j].tanh();
            }
            for n in 0..self.vocab {
                let row = &self.w2[r][n * self.hidden..(n + 1) * self.hidden];
                self.o[n] = row.iter().zip(&self.h).map(|(w, v)| w * v).sum();
            }
            let peak = self.o.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for v in self.o.iter_mut() {
                *v = (*v - peak).exp();
                total += *v;
            }
            for v in self.o.iter_mut() {
                *v /= total;
            }
            if in_tail {
                self.tail[r] += -self.o[target as usize].max(f64::MIN_POSITIVE).ln();
            }
            self.o[target as usize] -= 1.0;

            let lr = self.rates[r];
            self.gh.fill(0.0);
            for n in 0..self.vocab {
                let g = self.o[n];
                if g == 0.0 {
                    continue;
                }
                let row = &mut self.w2[r][n * self.hidden..(n + 1) * self.hidden];
                for (j, (w, hv)) in row.iter_mut().zip(&self.h).enumerate() {
                    self.gh[j] += g * *w;
                    *w -= lr * g * hv;
                }
            }
            for j in 0..self.hidden {
                let gz = self.gh[j] * (1.0 - self.h[j] * self.h[j]);
                if gz == 0.0 {
                    continue;
                }
                let row = &mut self.w1[r][j * self.width..(j + 1) * self.width];
                for (w, v) in row.iter_mut().zip(x) {
                    *w -= lr * gz * v;
                }
            }
        }
    }

    fn best_tail(&self) -> f64 {
        self.tail.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Symbol layout for the lost-in-the-middle test.
///
/// A needle is a prompt followed by a value, taught at a known position and
/// asked again at the end. Two layouts, and which one is in use decides what
/// the test can credit:
///
/// * **Single-symbol key.** One key symbol per needle. This is an *order-1*
///   task: §11 showed an untied head with `eta = 1` stores a bigram exactly in
///   one write, so the head alone saturates it and the network is not needed.
///   §29.6 measured exactly that — bypass beat the network.
/// * **Paired key.** `P` key symbols shared across `P * P` needles, the value
///   fixed by the *pair*. Each second symbol appears in `P` needles with `P`
///   different values, so knowing only the last token leaves the answer uniform
///   over `P`: an order-1 model is provably capped at `ln P` and anything below
///   that had to have carried the first symbol forward. That is the part only
///   the network can do.
struct Needles {
    /// Symbols taken off the top of the vocabulary.
    reserved: usize,
    /// Prompt for each needle. The last token is the scored slot, since
    /// `Scored` charges for predicting what follows it.
    prompt: Vec<Vec<u32>>,
    value: Vec<u32>,
}

impl Needles {
    /// `key_symbols` below two selects the single-symbol layout.
    fn plan(count: usize, key_symbols: usize) -> Self {
        if count == 0 {
            return Self {
                reserved: 0,
                prompt: Vec::new(),
                value: Vec::new(),
            };
        }
        if key_symbols < 2 {
            return Self {
                reserved: 2 * count,
                prompt: (0..count).map(|i| vec![2 * i as u32]).collect(),
                value: (0..count).map(|i| 2 * i as u32 + 1).collect(),
            };
        }
        let p = key_symbols;
        let n = p * p;
        Self {
            reserved: p + n,
            prompt: (0..n)
                .map(|i| vec![(i / p) as u32, (i % p) as u32])
                .collect(),
            value: (0..n).map(|i| (p + i) as u32).collect(),
        }
    }

    fn len(&self) -> usize {
        self.value.len()
    }

    /// Tokens one teaching repetition occupies.
    fn stride(&self) -> usize {
        self.prompt.first().map_or(0, |k| k.len()) + 1
    }

    /// The floor an order-1 model cannot beat, in nats, or zero when the layout
    /// does not impose one.
    fn order_one_floor(&self, key_symbols: usize) -> f64 {
        if key_symbols < 2 {
            0.0
        } else {
            (key_symbols as f64).ln()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub order: usize,
    /// Independent sources cycled through, for the continual-learning test.
    pub domains: usize,
    /// Tokens spent in each domain before switching.
    pub domain_span: usize,
    /// Replace one domain with an unseen chain at this fraction through the run.
    /// Separates domain-specific retention from a model that has simply got
    /// better at chains in general: a fresh domain should re-enter at the
    /// pass-0 level if retention is specific, and at the current level if it is
    /// not.
    pub fresh_domain_at: f64,
    /// Needle-in-a-haystack probes for the lost-in-the-middle test (§0), 0 to
    /// disable. Each needle is a key/value pair spliced into the stream once at
    /// a known position; all of them are queried in a block at the very end, so
    /// the only thing separating them is how long ago they were seen.
    ///
    /// The pairs use the top `2 * needles` symbols, and the source is built over
    /// the rest, so a key occurs exactly twice in the whole run: once when it is
    /// taught and once when it is asked. Without that reservation a key would
    /// also turn up as ordinary source output and retrieval would be ill-posed.
    pub needles: usize,
    /// How many times each needle is taught. One is the one-shot test; more is
    /// the positive control that says whether the measurement can show recall
    /// at all, so that a null result at one shot means "cannot bind in one
    /// exposure" rather than "this bench cannot see binding".
    pub needle_repeats: usize,
    /// Size of the shared key-symbol pool. Below two, each needle gets its own
    /// key and the task is order-1; at `P >= 2` there are `P * P` needles whose
    /// value is fixed by a *pair* of keys, which an order-1 model cannot
    /// resolve. See `Needles`.
    pub needle_key_symbols: usize,
    /// Subtract the current token's running mean from the assembled vector
    /// before the readout; see `ModelParams::centre_readout`.
    /// Components the lookup readout evaluates per token; 0 is adaptive.
    pub head_top_k: usize,
    /// Past keys each node keeps for the key-echo diagnostic; 0 is off.
    /// Ladder rung the forward pass reads.
    pub read_rung: usize,
    pub key_echo: usize,
    /// Measure how far back the assembled vector still names a token: a linear
    /// probe per lag, against the token that many positions earlier.
    ///
    /// The cursor maps a one-dimensional stream onto a two-dimensional torus,
    /// so stream lag becomes spatial distance along only two scales — one step
    /// in x is lag 1, one step in y is lag `grid_side`. Every other lag aliases
    /// onto those. On an order-2 chain that costs nothing, because lag 1 is the
    /// only lag carrying anything. Real text needs a continuum, and a lag that
    /// has been aliased cannot be recovered downstream by any parameter. This
    /// measures the spectrum directly instead of inferring it.
    pub lag_probe: bool,
    pub centre_readout: bool,
    /// Real text instead of the synthetic chain: a JSON corpus of `input_text`
    /// records, and the SentencePiece model to segment it with.
    pub corpus: Option<std::path::PathBuf>,
    pub tokenizer: Option<std::path::PathBuf>,
    pub fanout: usize,
    pub d_head: usize,
    pub slots: usize,
    pub grid_side: usize,
    pub long_range: usize,
    /// Long-range contact exponent. §9 measured this standalone with greedy
    /// lattice-distance routing and never end to end.
    pub exponent: f64,
    pub rungs: usize,
    pub context_scales: usize,
    /// Run linear probes along the path and report where context lives.
    pub probe: bool,
    pub embed_rungs: usize,
    pub mass_floor: f64,
    pub eta: f64,
    pub learning_rate: f64,
    pub ladder_ratio: f64,
    pub seed: u64,
    /// Seed for topology and model initialisation only. Separate from `seed`
    /// so run-to-run variance of the *architecture* can be measured against a
    /// fixed source and a fixed token stream — without it, changing seeds
    /// changes the data too and the two effects cannot be told apart.
    pub structure_seed: u64,
    /// Run the control: score the input embedding instead of the network's
    /// output, everything else identical.
    pub bypass: bool,
    /// Freeze the topology instead of rewiring long-range contacts.
    pub frozen_topology: bool,
    /// Rewire to the first candidate drawn rather than the least-visited.
    pub blind_turnover: bool,
    /// Give the output head its own weights instead of reusing the embedding
    /// Tie the output head to the embedding table. `E E^T` is symmetric and
    /// positive semi-definite, so a tied head provably cannot express an
    /// asymmetric bigram (§11.2); untied is the default for that reason.
    pub tied: bool,
    pub head_kind: HeadKind,
    pub ingress: IngressMode,
    /// Send every token to the same anchor, ignoring content.
    /// Admit tokens one per tick regardless of what is in flight. Leaks the
    /// future into earlier predictions; only for measuring by how much.
    pub overlapped: bool,
    /// Pin every token to its phase position instead of remembering where its
    /// mass came to rest. The control for the adaptive anchor.
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
    let hi = ((scored.len() as f64 * to) as usize)
        .max(lo + 1)
        .min(scored.len());
    mean(&scored[lo..hi].iter().map(f).collect::<Vec<_>>())
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    crate::write_manifest(out_dir, "run", cfg);
    std::fs::create_dir_all(out_dir)?;
    // Independent structural streams off one seed, each with a fixed salt.
    //
    // Sharing one generator is how a sweep silently changes what it is not
    // sweeping: `Topology::small_world` consumes a number of draws that depends
    // on `long_range`, so with a shared stream every point of a long-range sweep
    // also got a different embedding table, rotation and ingress plane, and the
    // measured effect was topology plus a different random architecture. Salting
    // keeps each knob's effect its own. Turnover gets its own stream inside
    // `Runtime::new`, from the same seed, for the same reason.
    let mut topology_rng = Rng::new(cfg.structure_seed ^ 0x7090_1067_5EED_0001);
    let mut rng = Rng::new(cfg.structure_seed ^ 0x0DE1_5EED_0000_0002);
    let schedule = Schedule::Geometric {
        r: cfg.ladder_ratio,
        g1: 0.5,
    };

    let topology = Topology::small_world(
        Grid::new(cfg.grid_side),
        SmallWorld {
            long_range: cfg.long_range,
            exponent: cfg.exponent,
        },
        &mut topology_rng,
    );
    let mut runtime = Runtime::new(
        topology,
        ModelParams {
            centre_readout: cfg.centre_readout,
            head_top_k: cfg.head_top_k,
            vocab: cfg.vocab,
            d_head: cfg.d_head,
            slots: cfg.slots,
            grid_side: cfg.grid_side,
            schedule,
            embed_rungs: cfg.embed_rungs,
            learning_rate: cfg.learning_rate,
            tied: cfg.tied,
            head_kind: cfg.head_kind,
        },
        NodeParams {
            absorb: cfg.absorb,
            d_head: cfg.d_head,
            eta: cfg.eta,
            schedule,
            rungs: cfg.rungs,
            context_scales: cfg.context_scales,
            key_echo: cfg.key_echo,
            read_rung: cfg.read_rung,
        },
        EngineParams {
            mass_floor: cfg.mass_floor,
            slots: cfg.slots,
        },
        cfg.structure_seed,
        &mut rng,
    );

    runtime.set_bypass(cfg.bypass);
    runtime.set_turnover(!cfg.frozen_topology);
    runtime.set_blind_turnover(cfg.blind_turnover);
    runtime.set_mode(if cfg.overlapped {
        Mode::Overlapped
    } else {
        Mode::Serial
    });
    runtime.set_ingress_mode(cfg.ingress);
    // The needle geometry below reads assembled vectors, so capture them
    // whenever needles are on even if the probes are not wanted.
    runtime.capture_representations(cfg.probe || cfg.needles > 0);

    // The sources get their own generator. Drawing them from the one the model
    // construction just used would make the data depend on the architecture:
    // `Ingress::new` consumes a number of draws that varies with `slots`, so a
    // sweep over slot counts silently swept over Markov chains too. The printed
    // entropy rate is the guard — it must not move across a sweep.
    //
    // With `domains > 1` the stream cycles through independent chains, which is
    // the continual-learning test: does returning to a domain cost less the
    // second time than the first, or has the model forgotten it?
    let mut source_rng = Rng::new(cfg.seed ^ 0x50_17_CE_50_17_CE_00);
    // Needle symbols are carved off the top of the vocabulary so the source can
    // never emit one; see `Config::needles`.
    // Real text, if asked for. Done before anything sizes itself on `vocab`,
    // because the corpus decides what the vocabulary actually is.
    let corpus_stream = match (&cfg.corpus, &cfg.tokenizer) {
        (Some(c), Some(t)) => {
            let st = crate::corpus::stream(c, t, cfg.tokens, cfg.vocab)?;
            println!("corpus: {}", c.display());
            println!(
                "  {} documents, {} tokens, {} distinct pieces before truncation",
                st.documents,
                st.tokens.len(),
                st.distinct_pieces
            );
            println!(
                "  kept the {} most frequent; {:.2}% of tokens fell into the catch-all id",
                cfg.vocab - 1,
                100.0 * st.unknown_share
            );
            println!("  commonest pieces: {:?}", st.commonest);
            println!();
            Some(st)
        }
        (Some(_), None) | (None, Some(_)) => {
            panic!("--corpus and --tokenizer go together")
        }
        (None, None) => None,
    };
    if corpus_stream.is_some() {
        assert!(cfg.needles == 0, "needles are for the synthetic source");
        assert!(
            cfg.domains <= 1,
            "domain cycling is for the synthetic source"
        );
    }

    let plan = Needles::plan(cfg.needles, cfg.needle_key_symbols);
    let needles = plan.len();
    assert!(
        plan.reserved < cfg.vocab,
        "the needle layout reserves {} of the {} symbols, leaving nothing for the source; \
         raise --vocab so that vocab minus the reserved count is the source size you want",
        plan.reserved,
        cfg.vocab
    );
    let source_vocab = cfg.vocab - plan.reserved;
    let base = source_vocab as u32;
    let mut sources: Vec<MarkovSource> = (0..cfg.domains.max(1))
        .map(|_| MarkovSource::new(source_vocab, cfg.order, cfg.fanout, &mut source_rng))
        .collect();
    let entropy_rate = sources.iter().map(|s| s.entropy_rate()).sum::<f64>() / sources.len() as f64;

    let started = std::time::Instant::now();
    let mut scored: Vec<Scored> = Vec::with_capacity(cfg.tokens);
    let mut stream_rng = Rng::new(cfg.seed ^ 0x5EED);
    let d_model = cfg.slots * cfg.d_head;
    let mut probes_prev: Vec<Probe> = (0..3).map(|_| Probe::new(cfg.vocab, d_model)).collect();
    let mut probes_cur: Vec<Probe> = (0..3).map(|_| Probe::new(cfg.vocab, d_model)).collect();
    let mut probe_previous: std::collections::BTreeMap<u32, u32> = Default::default();
    // Lags chosen to straddle the two scales the cursor can express: 1, the
    // grid side, and a spread either way.
    let lags: Vec<usize> = if cfg.lag_probe {
        let g = cfg.grid_side;
        let mut v = vec![1usize, 2, 3, 4, 6, 8, 12, 16, g, g + 1, 2 * g, 4 * g];
        v.sort_unstable();
        v.dedup();
        v
    } else {
        Vec::new()
    };
    let mut lag_probes: Vec<Probe> = lags
        .iter()
        .map(|_| Probe::new(cfg.vocab, d_model))
        .collect();
    let mut history: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut readout_linear = Probe::new(cfg.vocab, d_model);
    // The absolute yardsticks are tables over the vocabulary and grow as
    // `V^(order+1)`. That is nothing at `V = 16` and impossible on real text —
    // an order-2 counting coder at `V = 4096` would want half a terabyte — so
    // each one is built only if it fits a fixed budget and is reported as
    // unavailable rather than silently skipped.
    const CELL_BUDGET: usize = 1 << 27;
    let affords = |cells: usize| cells <= CELL_BUDGET;
    let oracle_cells = cfg.vocab.saturating_pow(3);
    let mut oracle = affords(oracle_cells)
        .then(|| OracleCoder::new(cfg.vocab, Probe::new(cfg.vocab, 1).rate_count()));
    let mut decoded: Vec<u32> = Vec::new();
    let mut readout_nonlinear = NonlinearProbe::new(
        cfg.vocab,
        d_model,
        d_model,
        &mut Rng::new(cfg.seed ^ 0x9A_7C_11_05),
    );
    let mut additive = affords(cfg.vocab.saturating_pow(2)).then(|| AdditiveCoder::new(cfg.vocab));
    let mut previous_token: Option<u32> = None;
    let mut coders: Vec<(usize, CountingCoder)> = [1, cfg.order]
        .into_iter()
        .filter(|k| affords(cfg.vocab.saturating_pow(*k as u32 + 1)))
        .map(|k| (k, CountingCoder::new(cfg.vocab, k)))
        .collect();
    let mut coder_nats = vec![0.0f64; coders.len()];
    let mut coder_tail = vec![0.0f64; coders.len()];
    // Order of a coder -> its slot, so a missing yardstick is a lookup miss
    // rather than an index into the wrong table.
    let coder_slot: std::collections::HashMap<usize, usize> = coders
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (*k, i))
        .collect();
    let domains = cfg.domains.max(1);
    let span = cfg.domain_span.max(1);
    let swap_at = if cfg.fresh_domain_at > 0.0 && cfg.fresh_domain_at < 1.0 {
        Some((cfg.tokens as f64 * cfg.fresh_domain_at) as usize)
    } else {
        None
    };
    let mut swapped = false;
    let mut fresh_reentry = (0.0f64, 0u64);
    let mut familiar_reentry = (0.0f64, 0u64);
    // Loss on the opening fifth of each visit to a domain, indexed by which
    // pass through the cycle it is. Retention shows as this falling across
    // passes; catastrophic forgetting shows as it staying flat.
    let mut reentry: Vec<(f64, u64)> = vec![(0.0, 0); cfg.tokens / (span * domains) + 2];
    // Where each needle is taught, and where it is asked. Teaching is spread
    // evenly over the body of the run; asking happens in one block at the end,
    // in a shuffled order so that position within the block is uncorrelated
    // with how long ago the needle was taught.
    let stride = plan.stride();
    let query_len = needles * stride;
    let body = cfg.tokens.saturating_sub(query_len);
    let mut forced: HashMap<usize, u32> = HashMap::new();
    // Scored slot of a query -> which needle is being asked.
    let mut asked: HashMap<usize, usize> = HashMap::new();
    let mut taught_at: Vec<usize> = vec![0; needles];
    // 0 for the matched half, 1 for the null half; assigned by teaching
    // position rather than by needle index, see below.
    let mut role: Vec<u8> = vec![0; needles];
    // Assembled vector at each needle's last teaching, and at its query.
    //
    // This is the learner-free half of the diagnosis. A probe that fails to
    // read something out is only a lower bound (§23), so asking whether the
    // binding survives at all is asked as a geometric question instead: among
    // the needles sharing a second key symbol, is a query vector closest to its
    // own teaching vector? Chance is `1 / P`. At chance no readout of any kind
    // could recover the binding and the gap is in the network; above chance the
    // information is present and separable and the gap is in the readout.
    let mut taught_vec: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut asked_vec: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut last_teach: HashMap<usize, usize> = HashMap::new();
    // Which needle's value each query actually asks for: the needle's own for
    // the matched half, another needle's for the null half.
    let mut answer: HashMap<usize, usize> = HashMap::new();
    if needles > 0 {
        let repeats = cfg.needle_repeats.max(1);
        let mut order: Vec<usize> = (0..needles).collect();
        for k in (1..order.len()).rev() {
            order.swap(k, stream_rng.next_below(k as u64 + 1) as usize);
        }
        let teach = |forced: &mut HashMap<usize, u32>, at: usize, i: usize, value: usize| {
            for (o, t) in plan.prompt[i].iter().enumerate() {
                forced.insert(at + o, base + *t);
            }
            forced.insert(at + stride - 1, base + plan.value[value]);
        };
        // Teaching order is a permutation of the needles, not their index.
        //
        // With the paired layout, needle `i` uses key symbols `(i / P, i % P)`,
        // so the needles sharing a second symbol are `i, i+P, i+2P, ...`. Laying
        // them out in index order puts the *last* write for every shared key in
        // the newest band, which makes band membership the same thing as "am I
        // the most recent binding for my key". The gradient measured that way is
        // last-write-wins — trivially expected, and nothing to do with distance.
        let mut layout: Vec<usize> = (0..needles).collect();
        for k in (1..layout.len()).rev() {
            layout.swap(k, stream_rng.next_below(k as u64 + 1) as usize);
        }
        for (slot, &i) in layout.iter().enumerate() {
            let at = (slot + 1) * body / (needles + 2);
            for r in 0..repeats {
                teach(&mut forced, at + r * stride, i, i);
            }
            taught_at[i] = at;
            // Scored slot of the final teaching repetition.
            last_teach.insert(at + (repeats - 1) * stride + stride - 2, i);
            // Matched and foil alternate along *position*, so the two halves
            // still span the same distances after the permutation.
            role[i] = (slot % 2) as u8;
        }
        // Half the needles are asked for their own value, half for another
        // needle's. The mismatched half is the null: those values were taught
        // exactly as often and are exactly as rare, so the only thing separating
        // the halves is whether the binding is the right one.
        //
        // Uniform-over-the-vocabulary is *not* the null. Needle symbols are rare,
        // so the model gives them far less than 1/V and a score above `ln V` can
        // still be recall. Reading the curve against that line would waste the
        // whole experiment.
        let odd: Vec<usize> = (0..needles).filter(|i| role[*i] == 1).collect();
        for (slot, &i) in order.iter().enumerate() {
            let at = body + slot * stride;
            let value = if role[i] == 0 {
                i
            } else {
                let k = odd.iter().position(|&x| x == i).expect("odd index");
                odd[(k + 1) % odd.len()]
            };
            teach(&mut forced, at, i, value);
            asked.insert(at + stride - 2, i);
            answer.insert(i, value);
        }
    }
    // Where a token's mass went, and whether that depends on the token.
    //
    // The visit count alone cannot separate two questions. `coverage` is the
    // share of the grid one token touches: near one and the reassembled vector
    // is a network-wide average, which cannot be specific to anything, and no
    // change of geometry makes it specific while the token still walks
    // everywhere. `selectivity` is paired — overlap between the node sets of
    // two occurrences of the *same* token, minus overlap between two
    // *different* tokens at a matched time distance. Same grid, same recency,
    // differing only in content, so geometry cancels and what is left is
    // whether the trajectory is content-driven at all.
    //
    // Neither reads without the other: at saturated coverage, selectivity is
    // zero by arithmetic rather than by dynamics.
    struct Trail {
        position: u32,
        token: u32,
        nodes: Vec<u32>,
    }
    let mut trails: std::collections::VecDeque<Trail> = std::collections::VecDeque::new();
    let (mut cover_sum, mut cover_n) = (0.0f64, 0u64);
    let (mut same_sum, mut same_n) = (0.0f64, 0u64);
    let (mut other_sum, mut other_n) = (0.0f64, 0u64);
    fn jaccard(a: &[u32], b: &[u32]) -> f64 {
        let (mut i, mut j, mut hit) = (0usize, 0usize, 0usize);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    hit += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        let union = a.len() + b.len() - hit;
        if union == 0 {
            0.0
        } else {
            hit as f64 / union as f64
        }
    }

    let mut needle_loss: Vec<f64> = vec![f64::NAN; needles];
    for i in 0..cfg.tokens {
        if let Some(at) = swap_at {
            // Swap at a domain boundary so the fresh chain gets a whole visit.
            if !swapped && i >= at && i % span == 0 {
                sources[0] =
                    MarkovSource::new(source_vocab, cfg.order, cfg.fanout, &mut source_rng);
                swapped = true;
            }
        }
        let domain = (i / span) % domains;
        let token = match corpus_stream.as_ref() {
            // Real text is already segmented; the synthetic paths below are what
            // the needle and domain machinery attach to.
            Some(st) => st.tokens[i],
            None => match forced.get(&i) {
                // Spliced in, not substituted: the chain is not advanced, so the
                // source sequence the model sees is still a valid sample of it.
                Some(t) => *t,
                None => sources[domain].next(&mut stream_rng),
            },
        };
        if let (Some(prev), Some(a)) = (previous_token, additive.as_mut()) {
            a.observe(prev, token, i * 10 >= cfg.tokens * 9);
        }
        previous_token = Some(token);
        for (k, (_, coder)) in coders.iter_mut().enumerate() {
            let nats = coder.observe(token);
            coder_nats[k] += nats;
            if i * 10 >= cfg.tokens * 9 {
                coder_tail[k] += nats;
            }
        }
        let pass = (i / (span * domains)).min(reentry.len() - 1);
        let within = i % span;
        for mut s in runtime.advance(Some(token)) {
            {
                let nodes = std::mem::take(&mut s.touched);
                cover_sum += nodes.len() as f64 / (cfg.grid_side * cfg.grid_side) as f64;
                cover_n += 1;
                if let Some(prev) = trails.iter().rev().find(|t| t.token == s.token) {
                    let gap = s.position - prev.position;
                    same_sum += jaccard(&nodes, &prev.nodes);
                    same_n += 1;
                    // The control: a different token as close to the same time
                    // distance as the ring allows. Matching the gap matters
                    // because trails drift as the topology rewires, so an
                    // unmatched control would be measuring drift, not content.
                    if let Some(other) = trails
                        .iter()
                        .filter(|t| t.token != s.token)
                        .min_by_key(|t| s.position.abs_diff(t.position).abs_diff(gap))
                    {
                        other_sum += jaccard(&nodes, &other.nodes);
                        other_n += 1;
                    }
                }
                if trails.len() >= 512 {
                    trails.pop_front();
                }
                trails.push_back(Trail {
                    position: s.position,
                    token: s.token,
                    nodes,
                });
            }
            if !lags.is_empty()
                && let Some(reps) = s.reps.as_ref()
            {
                for (probe, lag) in lag_probes.iter_mut().zip(&lags) {
                    // `history` holds tokens before this one, most recent last.
                    if history.len() >= *lag {
                        let target = history[history.len() - *lag];
                        let in_tail = s.position as usize * 10 >= cfg.tokens * 9;
                        probe.observe(&reps.assembled, target, in_tail);
                    }
                }
                let cap = lags.last().copied().unwrap_or(1) + 1;
                if history.len() >= cap {
                    history.pop_front();
                }
                history.push_back(s.token);
            }
            if let (Some(&n), Some(reps)) =
                (last_teach.get(&(s.position as usize)), s.reps.as_ref())
            {
                taught_vec.insert(n, reps.assembled.clone());
            }
            if let Some(&n) = asked.get(&(s.position as usize)) {
                if let Some(reps) = s.reps.as_ref() {
                    asked_vec.insert(n, reps.assembled.clone());
                }
                // Guard the alignment rather than trusting it: `Scored` charges
                // for the token *after* `position`, so scoring the key's slot is
                // only recall of the value if these line up.
                assert_eq!(
                    s.token,
                    base + *plan.prompt[n].last().expect("a needle has a prompt"),
                    "needle key misaligned"
                );
                assert_eq!(
                    s.target,
                    base + plan.value[answer[&n]],
                    "needle value misaligned"
                );
                needle_loss[n] = s.loss;
            }
            if domains > 1 && within < span / 5 {
                reentry[pass].0 += s.loss;
                reentry[pass].1 += 1;
                // After the swap, domain 0 is a chain the model has never seen
                // while the others are ones it has met many times. Both are read
                // in the same passes, so the model's general warm-up is matched
                // and only domain-specific retention can separate them. That
                // matters: comparing the fresh domain against pass 0 would not
                // control for warm-up, and pass 0 is measured on an untrained
                // model.
                if swapped {
                    let side = if domain == 0 {
                        &mut fresh_reentry
                    } else {
                        &mut familiar_reentry
                    };
                    side.0 += s.loss;
                    side.1 += 1;
                }
            }
            if let Some(reps) = s.reps.take() {
                let in_tail = s.position as usize * 10 >= cfg.tokens * 9;
                if let Some(prev) = probe_previous.get(&s.position) {
                    for (probe, x) in probes_prev.iter_mut().zip([
                        &reps.ingress,
                        &reps.after_one_hop,
                        &reps.assembled,
                    ]) {
                        probe.observe(x, *prev, in_tail);
                    }
                }
                for (probe, x) in
                    probes_cur
                        .iter_mut()
                        .zip([&reps.ingress, &reps.after_one_hop, &reps.assembled])
                {
                    probe.observe(x, s.token, in_tail);
                }
                // What any readout of the assembled vector could reach, split
                // into the part better optimisation would recover and the part
                // only a different shape can.
                readout_linear.observe(&reps.assembled, s.target, in_tail);
                readout_nonlinear.observe(&reps.assembled, s.target, in_tail);
                if let Some(prev) = probe_previous.get(&s.position) {
                    decoded.clear();
                    for r in 0..probes_prev[2].rate_count() {
                        decoded.push(probes_prev[2].argmax(&reps.assembled, r));
                    }
                    if let Some(o) = oracle.as_mut() {
                        o.observe(s.token, &decoded, *prev, s.target, in_tail);
                    }
                }
            }
            probe_previous.insert(s.position + 1, s.token);
            probe_previous.remove(&s.position);
            scored.push(s);
        }
    }
    let tail_count = (cfg.tokens - cfg.tokens * 9 / 10).max(1) as f64;
    scored.extend(runtime.drain(100_000));
    let elapsed = started.elapsed();

    let uniform = (cfg.vocab as f64).ln();
    let nats_to_bits = 1.0 / std::f64::consts::LN_2;

    // Name the source that actually ran. A header that says "synthetic Markov
    // source" over a run on real text is the same failure as a comment that
    // describes what the code was going to do.
    println!(
        "run — {}, {} tokens{}",
        match &cfg.corpus {
            Some(path) => format!("corpus {}", path.display()),
            None => "synthetic Markov source".to_string(),
        },
        scored.len(),
        if cfg.bypass { "  [BYPASS]" } else { "" }
    );
    println!("  absorb rule: {:?}", cfg.absorb);
    println!(
        "  protocol: {}",
        if cfg.overlapped {
            "OVERLAPPED — leaks future tokens, loss is NOT a compression bound"
        } else {
            "serial — one token in flight, loss is a valid compression bound"
        }
    );
    // Order and fanout describe the synthetic chain and mean nothing over a
    // corpus, so they are printed only when they are real.
    let source_shape = match &cfg.corpus {
        Some(_) => String::new(),
        None => format!(" order={} fanout={}", cfg.order, cfg.fanout),
    };
    println!(
        "  vocab={}{source_shape} d_head={} slots={} grid={}x{} deg={} floor={}",
        cfg.vocab,
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
    for (label, from, to) in [
        ("network, first decile", 0.0, 0.1),
        ("network, last decile", 0.9, 1.0),
    ] {
        println!(
            "  {:<26} {:>8.4}",
            label,
            window(&scored, from, to, |s| s.loss) * nats_to_bits
        );
    }
    // A yardstick that did not fit the budget is said to be missing, never
    // quietly replaced by a different one.
    let yardstick = |k: usize| coder_slot.get(&k).map(|i| coder_tail[*i] / tail_count);
    match yardstick(1) {
        Some(v) => println!(
            "  {:<26} {:>8.4}   <- ceiling on any context-free model",
            "order-1 counting coder",
            v * nats_to_bits
        ),
        None => println!("  order-1 counting coder      unavailable at this vocabulary"),
    }
    match yardstick(cfg.order) {
        Some(v) => println!(
            "  {:<26} {:>8.4}   <- ceiling on anything",
            format!("order-{} counting coder", cfg.order),
            v * nats_to_bits
        ),
        None => println!(
            "  order-{} counting coder      unavailable at this vocabulary",
            cfg.order
        ),
    }
    if let Some(a) = additive.as_ref() {
        println!(
            "  {:<26} {:>8.4}   <- ceiling on additive (non-interacting) models",
            "log-linear order-2",
            a.best_tail() / tail_count * nats_to_bits
        );
    }
    if corpus_stream.is_none() {
        println!(
            "  {:<26} {:>8.4}",
            "source entropy rate",
            entropy_rate * nats_to_bits
        );
    }
    if let (Some(a), Some(b)) = (yardstick(1), yardstick(cfg.order)) {
        println!(
            "  context is worth {:.4} bits/token: everything between the two coders",
            (a - b) * nats_to_bits
        );
    }
    println!(
        "  prequential total: {:.1} bits over {} tokens   (bigram coder: {:.1})",
        scored.iter().map(|s| s.loss).sum::<f64>() * nats_to_bits,
        scored.len(),
        coder_slot
            .get(&1)
            .map(|i| coder_nats[*i])
            .unwrap_or(f64::NAN)
            * nats_to_bits
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

    if cfg.probe {
        println!("linear probes, bits to name a token from a representation");
        println!(
            "  chance is {:.4}; lower means the information is linearly present",
            (cfg.vocab as f64).ln() * nats_to_bits
        );
        println!(
            "  {:<20} {:>12} {:>12}",
            "representation", "previous", "current"
        );
        for (i, label) in ["ingress embedding", "after one hop", "assembled"]
            .iter()
            .enumerate()
        {
            println!(
                "  {:<20} {:>12.4} {:>12.4}",
                label,
                probes_prev[i].best_tail() / tail_count * nats_to_bits,
                probes_cur[i].best_tail() / tail_count * nats_to_bits
            );
        }
        println!();
        println!("readout ceiling on the assembled vector, bits per token");
        println!(
            "  {:<26} {:>8.4}",
            "the model itself",
            window(&scored, 0.9, 1.0, |s| s.loss) * nats_to_bits
        );
        println!(
            "  {:<26} {:>8.4}   <- what the existing linear head could reach",
            "linear probe",
            readout_linear.best_tail() / tail_count * nats_to_bits
        );
        println!(
            "  {:<26} {:>8.4}   <- what a nonlinear readout could reach",
            "one-hidden-layer probe",
            readout_nonlinear.best_tail() / tail_count * nats_to_bits
        );
        if let Some(o) = oracle.as_ref() {
            let (oracle_tail, accuracy) = o.best();
            println!(
                "  {:<26} {:>8.4}   <- lookup on (current, previous decoded from it)",
                "decode-then-look-up",
                oracle_tail / tail_count * nats_to_bits
            );
            println!(
                "  previous token decoded correctly {:.1}% of the time",
                100.0 * accuracy
            );
        }
        println!();
    }

    if domains > 1 {
        println!("continual learning: {domains} domains, {span} tokens each, cycled");
        println!("  loss over the opening fifth of each visit, by pass through the cycle");
        println!("  {:<8} {:>10} {:>10}", "pass", "bits", "tokens");
        for (pass, (sum, n)) in reentry.iter().enumerate() {
            if *n > 0 {
                println!(
                    "  {:<8} {:>10.4} {:>10}",
                    pass,
                    sum / *n as f64 * nats_to_bits,
                    n
                );
            }
        }
        println!(
            "  falling across passes means the domain is being retained; flat means it is not"
        );
        if fresh_reentry.1 > 0 && familiar_reentry.1 > 0 {
            let fresh = fresh_reentry.0 / fresh_reentry.1 as f64 * nats_to_bits;
            let familiar = familiar_reentry.0 / familiar_reentry.1 as f64 * nats_to_bits;
            println!("  after swapping one domain for an unseen chain, over the same passes:");
            println!(
                "    fresh domain     {:>10.4} bits over {} tokens",
                fresh, fresh_reentry.1
            );
            println!(
                "    familiar domains {:>10.4} bits over {} tokens",
                familiar, familiar_reentry.1
            );
            println!("    gap              {:>10.4} bits", fresh - familiar);
            println!(
                "  a positive gap means what was retained is domain-specific, not general skill"
            );
        }
        println!();
    }

    if needles > 0 && cfg.needle_key_symbols >= 2 {
        // Nearest-neighbour retrieval among the needles that share a second key
        // symbol. Learner-free: it asks whether the binding is *there*, not
        // whether something managed to learn it.
        let p = cfg.needle_key_symbols;
        let cos = |a: &[f64], b: &[f64]| {
            let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
            for (x, y) in a.iter().zip(b) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            if na <= 0.0 || nb <= 0.0 {
                0.0
            } else {
                d / (na.sqrt() * nb.sqrt())
            }
        };
        // Retrieval is also run on vectors with the group mean removed. Within a
        // group every needle shares the second key symbol, so the group mean *is*
        // the component determined by the current token, and subtracting it
        // leaves exactly the part that depends on the first symbol. If accuracy
        // jumps, the binding is present but buried under a token-determined
        // component; if it does not, burial is the wrong explanation.
        let centre = |v: &HashMap<usize, Vec<f64>>, group: &[usize]| -> Vec<f64> {
            let mut m = vec![0.0; v.values().next().map_or(0, |x| x.len())];
            let mut n = 0.0;
            for j in group {
                if let Some(x) = v.get(j) {
                    for (a, b) in m.iter_mut().zip(x) {
                        *a += b;
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                for a in m.iter_mut() {
                    *a /= n;
                }
            }
            m
        };
        let less =
            |x: &[f64], m: &[f64]| -> Vec<f64> { x.iter().zip(m).map(|(a, b)| a - b).collect() };
        let (mut chits, mut cmargin) = (0usize, 0.0f64);
        let (mut hits, mut total, mut margin) = (0usize, 0usize, 0.0f64);
        for i in 0..needles {
            let Some(q) = asked_vec.get(&i) else { continue };
            // Needles sharing this second key symbol are the confusable set.
            let group: Vec<usize> = (0..needles).filter(|j| j % p == i % p).collect();
            let mut best = (f64::NEG_INFINITY, usize::MAX);
            let mut own = f64::NEG_INFINITY;
            for &j in &group {
                let Some(t) = taught_vec.get(&j) else {
                    continue;
                };
                let c = cos(q, t);
                if j == i {
                    own = c;
                }
                if c > best.0 {
                    best = (c, j);
                }
            }
            if best.1 == usize::MAX || own == f64::NEG_INFINITY {
                continue;
            }
            let runner = group
                .iter()
                .filter(|&&j| j != i)
                .filter_map(|j| taught_vec.get(j).map(|t| cos(q, t)))
                .fold(f64::NEG_INFINITY, f64::max);
            total += 1;
            margin += own - runner;
            if best.1 == i {
                hits += 1;
            }

            let tm = centre(&taught_vec, &group);
            let qm = centre(&asked_vec, &group);
            let qc = less(q, &qm);
            let mut cbest = (f64::NEG_INFINITY, usize::MAX);
            let (mut cown, mut crunner) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &j in &group {
                let Some(t) = taught_vec.get(&j) else {
                    continue;
                };
                let c = cos(&qc, &less(t, &tm));
                if j == i {
                    cown = c;
                } else if c > crunner {
                    crunner = c;
                }
                if c > cbest.0 {
                    cbest = (c, j);
                }
            }
            if cbest.1 == i {
                chits += 1;
            }
            if cown > f64::NEG_INFINITY && crunner > f64::NEG_INFINITY {
                cmargin += cown - crunner;
            }
        }
        if total > 0 {
            println!("needle geometry: is a query vector closest to its own teaching vector?");
            println!(
                "  {hits}/{total} correct, chance is 1/{p} = {:.1}%; mean margin over the runner-up {:+.4}",
                100.0 / p as f64,
                margin / total as f64
            );
            println!(
                "  at chance the binding is not in the representation and no readout can recover it"
            );
            println!(
                "  with the token-determined group mean removed: {chits}/{total} correct, margin {:+.4}",
                cmargin / total as f64
            );
            println!();
        }
    }

    if needles > 0 {
        let mut matched: Vec<(usize, f64)> = Vec::new();
        let mut foil: Vec<(usize, f64)> = Vec::new();
        for (i, (at, l)) in taught_at.iter().zip(&needle_loss).enumerate() {
            if !l.is_finite() {
                continue;
            }
            let row = (body - at, l * nats_to_bits);
            if role[i] == 0 {
                matched.push(row)
            } else {
                foil.push(row)
            }
        }
        // Taught in permuted order, so sort back to oldest-first before banding.
        matched.sort_by_key(|(d, _)| std::cmp::Reverse(*d));
        foil.sort_by_key(|(d, _)| std::cmp::Reverse(*d));
        println!(
            "lost in the middle: {} needles taught {} time(s) each, all queried at the end",
            taught_at.len(),
            cfg.needle_repeats.max(1)
        );
        println!("  matched asks for the needle's own value, foil asks with the wrong key;");
        let floor = plan.order_one_floor(cfg.needle_key_symbols) * nats_to_bits;
        if floor > 0.0 {
            println!(
                "  an order-1 model cannot get matched below {floor:.4} bits, so anything under \
                 that carried the first key symbol forward"
            );
        }
        println!("  the gap between them is the recall, and a flat gap is the claim");
        println!(
            "  {:<10} {:>9} {:>9} {:>9}   range",
            "taught ago", "matched", "foil", "gap"
        );
        // An odd needle count leaves the halves uneven; band over the common
        // length so a band is never silently dropped.
        let paired = matched.len().min(foil.len());
        let third = (paired / 3).max(1);
        let mean_of = |v: &[(usize, f64)]| v.iter().map(|(_, l)| l).sum::<f64>() / v.len() as f64;
        for (label, a, b) in [
            ("oldest", 0, third),
            ("middle", third, paired.saturating_sub(third)),
            ("newest", paired.saturating_sub(third), paired),
        ] {
            if a >= b {
                continue;
            }
            let (m, f) = (mean_of(&matched[a..b]), mean_of(&foil[a..b]));
            let (lo, hi) = (matched[b - 1].0, matched[a].0);
            println!(
                "  {label:<10} {m:>9.4} {f:>9.4} {:>9.4}   ({lo} to {hi} back)",
                f - m
            );
        }
        println!();
    }

    if let Some(c) = runtime.readout_coverage() {
        println!(
            "readout: {:.1}% of components ever won gate mass",
            100.0 * c
        );
        println!("  below 100% with --head-top-k means slots are frozen out of the competition");
        println!();
    }

    if !lags.is_empty() {
        println!("how far back the assembled vector still names a token");
        println!(
            "  chance is {:.4} bits; the cursor can express lag 1 and lag {} and aliases the rest",
            (cfg.vocab as f64).ln() * nats_to_bits,
            cfg.grid_side
        );
        println!("  {:>5} {:>10}", "lag", "bits");
        for (probe, lag) in lag_probes.iter().zip(&lags) {
            println!(
                "  {lag:>5} {:>10.4}",
                probe.best_tail() / tail_count * nats_to_bits
            );
        }
        println!();
    }

    if cover_n > 0 {
        println!("where a token's mass goes");
        println!(
            "  coverage, share of the grid one token touches  {:.4}",
            cover_sum / cover_n as f64
        );
        if same_n > 0 && other_n > 0 {
            let (same, other) = (same_sum / same_n as f64, other_sum / other_n as f64);
            println!("  node-set overlap, same token again             {same:.4}");
            println!("  node-set overlap, a different token            {other:.4}");
            println!(
                "  selectivity, the difference                   {:+.4}",
                same - other
            );
        }
        println!("  coverage near 1 means the assembled vector is a network-wide average,");
        println!("  and selectivity then reads zero by arithmetic rather than by dynamics");
        println!();
    }

    if let Some((surprise, added)) = runtime.visit_stats() {
        println!("what a visit adds to the payload, per visit");
        println!("  magnitude of tanh(Wq)      {added:>8.4}   (the payload has norm 1)");
        println!("  surprise, how wrong it was {surprise:>8.4}");
        println!("  large and wrong is what would make more visits cost rather than pay");
        println!();
    }

    if let Some((mean, hits)) = runtime.key_echo() {
        println!("key echo: does a node's context key ever come back?");
        println!(
            "  closest previous key, mean cosine {mean:.4}; {:.1}% of visits above 0.5",
            100.0 * hits
        );
        println!("  a node's memory is addressed by this key, so a low echo means");
        println!("  the write went somewhere the read never returns to");
        println!();
    }

    let mut visits = runtime.engine().visits().to_vec();
    let total: u64 = visits.iter().sum();
    visits.sort_unstable();
    let idle = visits.iter().filter(|c| **c == 0).count();
    let decile = visits.len() * 9 / 10;
    println!(
        "load across {} nodes, ingress {:?}",
        visits.len(),
        cfg.ingress
    );
    println!(
        "  never visited      {idle} ({:.1}%)",
        100.0 * idle as f64 / visits.len() as f64
    );
    println!(
        "  busiest decile     {:.1}% of all visits (10% would be uniform)",
        100.0 * visits[decile..].iter().sum::<u64>() as f64 / total as f64
    );
    println!(
        "  busiest single     {:.2}% of all visits",
        100.0 * visits[visits.len() - 1] as f64 / total as f64
    );
    println!();

    println!("topology");
    println!(
        "  {}",
        if cfg.frozen_topology {
            "frozen".to_string()
        } else {
            format!("{} rewirings", runtime.rewirings())
        }
    );
    println!();

    let (known, distinct, mean_move) = runtime.model().ingress().drift();
    println!("ingress: {:?}", cfg.ingress);
    println!("  tokens with a remembered anchor  {known} / {}", cfg.vocab);
    println!("  distinct cells occupied          {distinct}");
    println!("  mean move per observation        {mean_move:.2} (torus L1)");
    println!("  anchor movement over the run (a settling map decays, a jittering one does not)");
    for (label, from, to) in [
        ("first tenth", 0.0, 0.1),
        ("middle tenth", 0.45, 0.55),
        ("last tenth", 0.9, 1.0),
    ] {
        println!(
            "    {:<16} {:>6.2}",
            label,
            window(&scored, from, to, |s| s.anchor_move as f64)
        );
    }
    println!();

    let mut csv =
        String::from("position,token,target,loss_nats,passthrough_nats,visits,mean_hops\n");
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
            assert!(
                source.entropy_rate() < 1e-12,
                "order {order}: {}",
                source.entropy_rate()
            );
        }
    }

    #[test]
    fn entropy_rate_is_bounded_by_the_fanout() {
        let mut rng = Rng::new(2);
        for order in [1usize, 2] {
            for fanout in [2usize, 4, 8] {
                let source = MarkovSource::new(16, order, fanout, &mut rng);
                let h = source.entropy_rate();
                assert!(
                    h > 0.0 && h < (fanout as f64).ln(),
                    "order {order} fanout {fanout}: {h}"
                );
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
            assert!(
                allowed.contains(&next),
                "context {context} -> {next} is not an edge"
            );
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
        assert!(
            n2 < n1 * 0.8,
            "order-2 coder {n2} did not clearly beat order-1 {n1}"
        );
    }
}
