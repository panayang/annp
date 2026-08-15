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

use crate::head::Head;
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
    /// `[rate][token * d_model + j]`. The embedding is the only learned thing
    /// outside the shared head, and it is what makes this a learned baseline
    /// rather than a fixed featuriser.
    embed: Vec<Vec<f64>>,
    head: Head,
    /// Scratch, reused so a step allocates nothing.
    x: Vec<f64>,
    history: std::collections::VecDeque<u32>,
}

impl WindowMlp {
    pub fn new(vocab: usize, window: usize, d_model: usize, hidden: usize, rng: &mut Rng) -> Self {
        assert!(window >= 1 && d_model >= 1 && hidden >= 1);
        let fan_in = window * d_model;
        let head = Head::new(fan_in, hidden, vocab, rng);
        let embed = (0..head.num_rates())
            .map(|_| {
                (0..vocab * d_model)
                    .map(|_| rng.next_normal() / (d_model as f64).sqrt())
                    .collect()
            })
            .collect();
        Self {
            vocab,
            window,
            d_model,
            embed,
            head,
            x: vec![0.0; fan_in],
            history: std::collections::VecDeque::new(),
        }
    }

    pub fn parameters(&self) -> usize {
        self.vocab * self.d_model + self.head.parameters()
    }

    /// Charges `target` against the current weights, then updates. Returns the
    /// best rate's nats for this token, so the caller can report a stream that
    /// is comparable token by token.
    pub fn observe_at(&mut self, target: u32, in_tail: bool, decile: usize) -> f64 {
        let mut best = f64::INFINITY;
        for r in 0..self.head.num_rates() {
            let nats = self.step(r, target);
            self.head.charge(r, nats, in_tail, decile);
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
        let d = self.d_model;
        // Positions with no history yet read a zero block, which is what a
        // model at the start of a stream legitimately knows.
        self.x.iter_mut().for_each(|x| *x = 0.0);
        for (slot, &tok) in self.history.iter().enumerate() {
            let lo = slot * d;
            let e = &self.embed[r][tok as usize * d..(tok as usize + 1) * d];
            self.x[lo..lo + d].copy_from_slice(e);
        }
        let nats = self.head.step(r, &self.x, target);
        // The head returns `dL/dx` against pre-update `w1`, which is the
        // gradient belonging to this forward pass. Route it into the
        // embeddings the window actually read.
        let lr = self.head.rate(r);
        for k in 0..self.head.gx.len() {
            let slot = k / d;
            if let Some(&tok) = self.history.get(slot) {
                self.embed[r][tok as usize * d + (k % d)] -= lr * self.head.gx[k];
            }
        }
        nats
    }

    /// Best accumulated code length across rates, and which rate won.
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
        per_token.push(model.observe_at(tok, i * 10 >= stream.len() * 9, i * 10 / stream.len()));
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
        // Freeze: measure the gradient, do not take a step.
        m.head.keep_one_rate(0.0);
        m.embed.truncate(1);
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
                    "w2" => m.head.w2_mut(0)[idx] += delta,
                    "w1" => m.head.w1_mut(0)[idx] += delta,
                    _ => m.embed[0][idx] += delta,
                }
                let l = m.step(0, target);
                match which {
                    "w2" => m.head.w2_mut(0)[idx] -= delta,
                    "w1" => m.head.w1_mut(0)[idx] -= delta,
                    _ => m.embed[0][idx] -= delta,
                }
                l
            };
            let up = loss_at(&mut m, eps);
            let down = loss_at(&mut m, -eps);
            let numeric = (up - down) / (2.0 * eps);

            // The analytic gradient, read off by taking a step of known size.
            m.head.set_rate(1.0);
            let before = match which {
                "w2" => m.head.w2_mut(0)[idx],
                "w1" => m.head.w1_mut(0)[idx],
                _ => m.embed[0][idx],
            };
            m.step(0, target);
            let after = match which {
                "w2" => m.head.w2_mut(0)[idx],
                "w1" => m.head.w1_mut(0)[idx],
                _ => m.embed[0][idx],
            };
            let analytic = before - after;
            m.head.set_rate(0.0);
            // Undo the step so the next probe starts from the same weights.
            match which {
                "w2" => m.head.w2_mut(0)[idx] = before,
                "w1" => m.head.w1_mut(0)[idx] = before,
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
            let nats = m.observe_at(next, false, 0);
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
