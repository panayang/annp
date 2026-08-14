//! Learned baselines on the same stream, under the same protocol.
//!
//! Everything this project has compared against so far is either an ablation of
//! itself (`--bypass`) or a counting table (the order-k coders). Neither is a
//! model anyone would choose. DESIGN.md §0 said E1 would align FLOPs against a
//! Transformer and that was never done, so the question "how far behind is
//! this" has no answer rather than a bad one.
//!
//! The comparison has to be like for like or it says nothing:
//!
//! * **Same stream.** Same corpus, same tokenizer, same vocabulary truncation,
//!   same token order.
//! * **Same protocol.** Prequential and single-pass — predict the next token,
//!   pay the code length, then update. No epochs, no held-out split, because
//!   the architecture under test has no train/run separation to give it one.
//! * **Same budget.** Parameter counts are reported so the comparison is at
//!   matched size rather than matched wall clock, which would flatter whichever
//!   side happened to be better optimised.
//! * **Tuning goes to the baseline.** Several learning rates run in parallel and
//!   the best is reported, the same courtesy the probes get in `run.rs`. A
//!   baseline that lost because nobody tuned it would prove nothing.
//!
//! The window model here is deliberately the weakest interesting opponent: a
//! fixed window of the last few tokens, embedded, concatenated and pushed
//! through one hidden layer. §40.6 measured the architecture's effective window
//! at two or three tokens, so a window of three is the matched question — is
//! this an expensive way to use a short window, or does the mechanism earn
//! something a trivial model with the same view cannot get?

use std::fmt::Write as _;
use std::path::Path;

use annp_core::rng::Rng;

/// A fixed-window model: embed the last `window` tokens, concatenate, one
/// hidden layer, then logits.
///
/// Weights are held once per learning rate so several can be raced. Each rate
/// is a complete independent model, which costs memory rather than correctness.
pub struct WindowMlp {
    vocab: usize,
    window: usize,
    d_model: usize,
    hidden: usize,
    rates: Vec<f64>,
    /// `[rate][token * d_model + j]`
    embed: Vec<Vec<f64>>,
    /// `[rate][h * (window * d_model) + j]`
    w1: Vec<Vec<f64>>,
    b1: Vec<Vec<f64>>,
    /// `[rate][token * hidden + h]`
    w2: Vec<Vec<f64>>,
    b2: Vec<Vec<f64>>,
    /// Accumulated code length per rate, in nats, and over the final tenth.
    nats: Vec<f64>,
    tail: Vec<f64>,
    /// Scratch, reused so a step allocates nothing.
    x: Vec<f64>,
    h: Vec<f64>,
    logits: Vec<f64>,
    gh: Vec<f64>,
    history: std::collections::VecDeque<u32>,
}

impl WindowMlp {
    pub fn new(vocab: usize, window: usize, d_model: usize, hidden: usize, rng: &mut Rng) -> Self {
        assert!(window >= 1 && d_model >= 1 && hidden >= 1);
        let rates = vec![0.003, 0.01, 0.03, 0.1];
        let n = rates.len();
        let fan_in = window * d_model;
        // He-style for the tanh layer and small for the rest; the exact scale
        // matters little because the rates are raced anyway.
        let draw = |count: usize, sigma: f64, rng: &mut Rng| -> Vec<f64> {
            (0..count).map(|_| rng.next_normal() * sigma).collect()
        };
        let embed = (0..n)
            .map(|_| draw(vocab * d_model, 1.0 / (d_model as f64).sqrt(), rng))
            .collect();
        let w1 = (0..n)
            .map(|_| draw(hidden * fan_in, 1.0 / (fan_in as f64).sqrt(), rng))
            .collect();
        let w2 = (0..n)
            .map(|_| draw(vocab * hidden, 1.0 / (hidden as f64).sqrt(), rng))
            .collect();
        Self {
            vocab,
            window,
            d_model,
            hidden,
            embed,
            w1,
            b1: vec![vec![0.0; hidden]; n],
            w2,
            b2: vec![vec![0.0; vocab]; n],
            nats: vec![0.0; n],
            tail: vec![0.0; n],
            x: vec![0.0; fan_in],
            h: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            gh: vec![0.0; hidden],
            rates,
            history: std::collections::VecDeque::new(),
        }
    }

    pub fn parameters(&self) -> usize {
        self.vocab * self.d_model
            + self.hidden * self.window * self.d_model
            + self.hidden
            + self.vocab * self.hidden
            + self.vocab
    }

    /// Charges `target` against the current weights, then updates. Returns the
    /// best rate's nats for this token, so the caller can report a stream that
    /// is comparable token by token.
    pub fn observe(&mut self, target: u32, in_tail: bool) -> f64 {
        let mut best = f64::INFINITY;
        for r in 0..self.rates.len() {
            let nats = self.step(r, target);
            self.nats[r] += nats;
            if in_tail {
                self.tail[r] += nats;
            }
            best = best.min(nats);
        }
        // The window advances after every rate has seen the same input.
        self.history.push_back(target);
        while self.history.len() > self.window {
            self.history.pop_front();
        }
        best
    }

    /// Forward and backward for one rate. Split out so the gradient can be
    /// checked against finite differences without going through `observe`.
    fn step(&mut self, r: usize, target: u32) -> f64 {
        let (w, d, hid, v) = (self.window, self.d_model, self.hidden, self.vocab);
        // Positions with no history yet read a zero block, which is what a
        // model at the start of a stream legitimately knows.
        self.x.iter_mut().for_each(|x| *x = 0.0);
        for (slot, &tok) in self.history.iter().enumerate() {
            let lo = slot * d;
            let e = &self.embed[r][tok as usize * d..(tok as usize + 1) * d];
            self.x[lo..lo + d].copy_from_slice(e);
        }
        for j in 0..hid {
            let row = &self.w1[r][j * (w * d)..(j + 1) * (w * d)];
            let z: f64 = row.iter().zip(&self.x).map(|(a, b)| a * b).sum::<f64>() + self.b1[r][j];
            self.h[j] = z.tanh();
        }
        for n in 0..v {
            let row = &self.w2[r][n * hid..(n + 1) * hid];
            self.logits[n] =
                row.iter().zip(&self.h).map(|(a, b)| a * b).sum::<f64>() + self.b2[r][n];
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
        for n in 0..v {
            let g = self.logits[n];
            if g == 0.0 {
                continue;
            }
            let row = &mut self.w2[r][n * hid..(n + 1) * hid];
            for (j, (weight, &hv)) in row.iter_mut().zip(&self.h).enumerate() {
                self.gh[j] += g * *weight;
                *weight -= lr * g * hv;
            }
            self.b2[r][n] -= lr * g;
        }
        // Through the tanh, then into the first layer and the embeddings that
        // fed it. The embedding gradient is what makes this a learned model
        // rather than a fixed featuriser.
        for j in 0..hid {
            let gz = self.gh[j] * (1.0 - self.h[j] * self.h[j]);
            if gz == 0.0 {
                continue;
            }
            let row = &mut self.w1[r][j * (w * d)..(j + 1) * (w * d)];
            for (k, (weight, &xv)) in row.iter_mut().zip(&self.x).enumerate() {
                // Accumulate into the embedding before the weight moves, so the
                // gradient is the one belonging to this forward pass.
                let slot = k / d;
                if let Some(&tok) = self.history.get(slot) {
                    self.embed[r][tok as usize * d + (k % d)] -= lr * gz * *weight;
                }
                *weight -= lr * gz * xv;
            }
            self.b1[r][j] -= lr * gz;
        }
        nats
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

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub window: usize,
    pub d_model: usize,
    pub hidden: usize,
    pub order: usize,
    pub fanout: usize,
    pub seed: u64,
    pub corpus: Option<std::path::PathBuf>,
    pub tokenizer: Option<std::path::PathBuf>,
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    crate::write_manifest(out_dir, "baseline", cfg);
    std::fs::create_dir_all(out_dir)?;
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

    let mut model = WindowMlp::new(cfg.vocab, cfg.window, cfg.d_model, cfg.hidden, &mut rng);
    let started = std::time::Instant::now();
    let mut per_token = Vec::with_capacity(stream.len());
    for (i, &tok) in stream.iter().enumerate() {
        per_token.push(model.observe(tok, i * 10 >= stream.len() * 9));
    }
    let elapsed = started.elapsed();
    let nats_to_bits = std::f64::consts::LOG2_E;
    let (total, rate) = model.best();
    let tail_count = (stream.len() / 10).max(1) as f64;

    println!();
    println!(
        "baseline — window {} over the last tokens, {} tokens",
        cfg.window,
        stream.len()
    );
    println!(
        "  vocab={} d_model={} hidden={} parameters={}",
        cfg.vocab,
        cfg.d_model,
        cfg.hidden,
        model.parameters()
    );
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        stream.len() as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("loss, bits per token");
    println!(
        "  {:<26} {:>8.4}   <- best of the raced learning rates ({rate})",
        "baseline, last decile",
        model.best_tail() / tail_count * nats_to_bits
    );
    println!(
        "  prequential total: {:.1} bits over {} tokens",
        total * nats_to_bits,
        stream.len()
    );

    let mut csv = String::from("position,nats\n");
    for (i, n) in per_token.iter().enumerate() {
        let _ = writeln!(csv, "{i},{n}");
    }
    std::fs::write(out_dir.join("baseline.csv"), csv)?;
    println!();
    println!("  wrote {}", out_dir.join("baseline.csv").display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite differences against the analytic gradient.
    ///
    /// A hand-written backward pass that is subtly wrong produces a baseline
    /// that merely learns badly, and a baseline that learns badly makes the
    /// system under test look good. That failure is silent, which is why this
    /// test exists before any number from this file is quoted.
    #[test]
    fn the_backward_pass_matches_finite_differences() {
        let mut rng = Rng::new(7);
        let (vocab, window, d, hid) = (11usize, 3usize, 5usize, 4usize);
        let mut m = WindowMlp::new(vocab, window, d, hid, &mut rng);
        m.rates = vec![0.0]; // freeze: measure the gradient, do not take a step
        m.b1 = vec![vec![0.0; hid]];
        m.b2 = vec![vec![0.0; vocab]];
        m.embed.truncate(1);
        m.w1.truncate(1);
        m.w2.truncate(1);
        m.nats = vec![0.0];
        m.tail = vec![0.0];
        for t in [2u32, 5, 9] {
            m.history.push_back(t);
        }

        let target = 4u32;
        let eps = 1e-6;
        // A weight in each layer, plus one embedding entry that the window
        // actually reads, since an embedding gradient that never fires would
        // pass a test that only poked the dense layers.
        let probes: Vec<(&str, usize)> = vec![("w2", 3), ("w1", 2), ("embed", 5 * d + 1)];
        for (which, idx) in probes {
            let loss_at = |m: &mut WindowMlp, delta: f64| -> f64 {
                match which {
                    "w2" => m.w2[0][idx] += delta,
                    "w1" => m.w1[0][idx] += delta,
                    _ => m.embed[0][idx] += delta,
                }
                let l = m.step(0, target);
                match which {
                    "w2" => m.w2[0][idx] -= delta,
                    "w1" => m.w1[0][idx] -= delta,
                    _ => m.embed[0][idx] -= delta,
                }
                l
            };
            let up = loss_at(&mut m, eps);
            let down = loss_at(&mut m, -eps);
            let numeric = (up - down) / (2.0 * eps);

            // The analytic gradient, read off by taking a step of known size.
            m.rates = vec![1.0];
            let before = match which {
                "w2" => m.w2[0][idx],
                "w1" => m.w1[0][idx],
                _ => m.embed[0][idx],
            };
            m.step(0, target);
            let after = match which {
                "w2" => m.w2[0][idx],
                "w1" => m.w1[0][idx],
                _ => m.embed[0][idx],
            };
            let analytic = before - after;
            m.rates = vec![0.0];
            // Undo the step so the next probe starts from the same weights.
            match which {
                "w2" => m.w2[0][idx] = before,
                "w1" => m.w1[0][idx] = before,
                _ => m.embed[0][idx] = before,
            }
            assert!(
                (analytic - numeric).abs() < 1e-6 * numeric.abs().max(1e-3),
                "{which}[{idx}]: analytic {analytic:e} against numeric {numeric:e}"
            );
        }
    }

    #[test]
    fn a_predictable_stream_is_learned() {
        // If the next token is a fixed function of the previous one, the loss
        // has to fall well below uniform. A baseline that cannot do this is
        // broken, and would understate whatever it is being compared with.
        let mut rng = Rng::new(3);
        let vocab = 8;
        let mut m = WindowMlp::new(vocab, 2, 8, 16, &mut rng);
        let mut last = 0u32;
        let mut early = 0.0;
        let mut late = 0.0;
        for i in 0..6000 {
            let next = (last * 3 + 1) % vocab as u32;
            let nats = m.observe(next, false);
            if i < 500 {
                early += nats;
            }
            if i >= 5500 {
                late += nats;
            }
            last = next;
        }
        assert!(
            late / 500.0 < 0.25 * (early / 500.0),
            "did not learn: {} then {}",
            early / 500.0,
            late / 500.0
        );
    }
}
