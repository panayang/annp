//! Synthetic Zipf source, the growing tree, and a continual-learning control.
//!
//! `DESIGN-TREE.md` §0 sets the rules this file exists to follow.
//!
//! The source is synthetic and its shape is verified in every run rather than
//! assumed. The corpus that preceded it was BPE-tokenised, which flattens the
//! head and deletes the tail, leaving about 1% of the mass in the region the
//! ladder exists to protect — a mechanism was being judged on data with its
//! subject removed.
//!
//! The control is online EWC, not a feedforward network. A memoryless learner
//! on a stationary stream is being judged on its home ground; the question here
//! is retention across shifts, so the comparison has to be against something
//! that also tries to retain. Both arms share the same write, so the difference
//! between them is the penalty and the topology, not two implementations.

use std::path::Path;

use annp_core::linalg::linear_fit;
use annp_core::rng::Rng;
use annp_core::tree::Tree;

/// Order-1 chain with a **Zipf marginal** and an informative conditional.
///
/// The first version made each state's successor ranking an independent random
/// permutation. That gives every context a Zipf conditional and a nearly
/// uniform marginal, because the heavy mass lands on a different symbol from
/// every state -- measured slope -0.169 with the top ten symbols holding 5.7%.
/// E0-b's result is about frequently revisited patterns crowding out rare ones,
/// which is a property of the *marginal*, so that source could not have tested
/// it.
///
/// Here the conditional is the Zipf marginal tilted by a state-dependent
/// factor: `P(j | c) ∝ pi(j) * exp(±eps)`. The marginal stays heavy-tailed and
/// the current symbol still carries information, so the current symbol remains
/// a sufficient statistic and any failure stays attributable to the learner.
pub struct ZipfChain {
    /// `[state][j]`, cumulative, width x width.
    cdf: Vec<Vec<f64>>,
    width: usize,
    offset: usize,
    vocab: usize,
    state: usize,
}

impl ZipfChain {
    pub fn new(width: usize, offset: usize, vocab: usize, s: f64, eps: f64, rng: &mut Rng) -> Self {
        assert!(width >= 2);
        let pi: Vec<f64> = (1..=width).map(|r| (r as f64).powf(-s)).collect();
        let cdf = (0..width)
            .map(|_| {
                let mut p: Vec<f64> = pi
                    .iter()
                    .map(|w| w * (if rng.next_f64() < 0.5 { -eps } else { eps }).exp())
                    .collect();
                let z: f64 = p.iter().sum();
                let mut acc = 0.0;
                for x in p.iter_mut() {
                    acc += *x / z;
                    *x = acc;
                }
                p
            })
            .collect();
        Self {
            cdf,
            width,
            offset,
            vocab,
            state: 0,
        }
    }

    pub fn next(&mut self, rng: &mut Rng) -> u32 {
        let u = rng.next_f64();
        self.state = self.cdf[self.state]
            .partition_point(|&c| c < u)
            .min(self.width - 1);
        ((self.state + self.offset) % self.vocab) as u32
    }
}

/// A linear softmax layer with an online EWC penalty, task free.
///
/// Same write as the node — `W += eta (onehot - softmax(Wk)) k^T` — plus a
/// quadratic pull toward a slowly moving anchor, weighted by a running Fisher
/// estimate. No task boundaries are supplied, because the stream does not
/// announce them.
pub struct OnlineEwc {
    w: Vec<Vec<f64>>,
    anchor: Vec<Vec<f64>>,
    fisher: Vec<Vec<f64>>,
    lambdas: Vec<f64>,
    nats: Vec<f64>,
    vocab: usize,
    d: usize,
    eta: f64,
    /// Rate at which the anchor and the Fisher follow the weights.
    trail: f64,
    logits: Vec<f64>,
}

impl OnlineEwc {
    pub fn new(vocab: usize, d: usize, eta: f64, trail: f64) -> Self {
        let lambdas = vec![0.0, 1.0, 10.0, 100.0];
        let n = lambdas.len();
        Self {
            w: vec![vec![0.0; vocab * d]; n],
            anchor: vec![vec![0.0; vocab * d]; n],
            fisher: vec![vec![0.0; vocab * d]; n],
            nats: vec![0.0; n],
            lambdas,
            vocab,
            d,
            eta,
            trail,
            logits: vec![0.0; vocab],
        }
    }

    pub fn observe(&mut self, key: &[f64], target: u32) -> f64 {
        let mut best = f64::INFINITY;
        for l in 0..self.lambdas.len() {
            let (v, d) = (self.vocab, self.d);
            for n in 0..v {
                let row = &self.w[l][n * d..(n + 1) * d];
                self.logits[n] = row.iter().zip(key).map(|(a, b)| a * b).sum();
            }
            let peak = self
                .logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0;
            for x in self.logits.iter_mut() {
                *x = (*x - peak).exp();
                z += *x;
            }
            for x in self.logits.iter_mut() {
                *x /= z;
            }
            let nats = -self.logits[target as usize].max(f64::MIN_POSITIVE).ln();
            self.nats[l] += nats;
            best = best.min(nats);

            let lam = self.lambdas[l];
            for n in 0..v {
                let g_out = if n == target as usize {
                    1.0 - self.logits[n]
                } else {
                    -self.logits[n]
                };
                let base = n * d;
                for (j, &k) in key.iter().enumerate() {
                    let idx = base + j;
                    let g = g_out * k;
                    let pull = lam * self.fisher[l][idx] * (self.w[l][idx] - self.anchor[l][idx]);
                    self.w[l][idx] += self.eta * (g - pull);
                    self.fisher[l][idx] += self.trail * (g * g - self.fisher[l][idx]);
                    self.anchor[l][idx] += self.trail * (self.w[l][idx] - self.anchor[l][idx]);
                }
            }
        }
        best
    }

    /// Best accumulated code length and the lambda that won.
    pub fn best(&self) -> (f64, f64) {
        let mut b = (f64::INFINITY, self.lambdas[0]);
        for (l, &n) in self.nats.iter().enumerate() {
            if n < b.0 {
                b = (n, self.lambdas[l]);
            }
        }
        b
    }

    pub fn lambda_at_boundary(&self) -> bool {
        let (_, l) = self.best();
        l == self.lambdas[0] || l == self.lambdas[self.lambdas.len() - 1]
    }

    pub fn parameters(&self) -> usize {
        self.vocab * self.d
    }

    /// Weights, anchor and Fisher are all held.
    pub fn state_held(&self) -> usize {
        3 * self.vocab * self.d
    }

    pub fn cost(&self) -> usize {
        self.lambdas.len() * 4 * self.vocab * self.d
    }
}

enum Arm {
    Tree(Box<Tree>),
    Ewc(Box<OnlineEwc>),
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tokens: usize,
    pub vocab: usize,
    pub d_model: usize,
    pub domains: usize,
    pub domain_span: usize,
    pub domain_width: f64,
    pub zipf_s: f64,
    /// State-dependent tilt on the Zipf marginal. Zero makes the conditional
    /// independent of the state and the task uninformative.
    pub tilt: f64,
    pub fanout: usize,
    pub depth: usize,
    pub rungs: usize,
    pub ladder_r: f64,
    pub eta: f64,
    pub ewc: bool,
    /// Print the source diagnostic and stop. The tilt trade-off is a property
    /// of the source alone, so measuring it should not cost a model run.
    pub source_only: bool,
    pub seed: u64,
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    crate::write_manifest(out_dir, "grow", cfg);
    std::fs::create_dir_all(out_dir)?;
    let mut rng = Rng::new(cfg.seed);
    let (d_count, span) = (cfg.domains.max(1), cfg.domain_span.max(1));

    // Overlapping alphabet windows: the overlap is where domains collide, so
    // there is interference to forget, and the rest is what tells them apart,
    // so routing has something to separate.
    let stride = (cfg.vocab / d_count).max(1);
    let width = ((stride as f64 * cfg.domain_width).round() as usize).clamp(2, cfg.vocab);
    let mut chains: Vec<ZipfChain> = (0..d_count)
        .map(|k| ZipfChain::new(width, k * stride, cfg.vocab, cfg.zipf_s, cfg.tilt, &mut rng))
        .collect();
    let mut draw = Rng::new(cfg.seed ^ 0x5EED);
    let stream: Vec<u32> = (0..cfg.tokens)
        .map(|i| chains[(i / span) % d_count].next(&mut draw))
        .collect();

    // Verify the shape instead of assuming it.
    let mut counts_by_id = vec![0.0f64; cfg.vocab];
    for &t in &stream {
        counts_by_id[t as usize] += 1.0;
    }
    let mut ranked: Vec<f64> = counts_by_id.iter().copied().filter(|c| *c > 0.0).collect();
    ranked.sort_by(|a, b| b.total_cmp(a));
    let total: f64 = ranked.iter().sum();
    let upper = ranked.len().min(1000);
    let (xs, ys): (Vec<f64>, Vec<f64>) = (1..=upper)
        .map(|r| ((r as f64).ln(), ranked[r - 1].ln()))
        .unzip();
    let (slope, _) = linear_fit(&xs, &ys);
    // A power law has a flat local exponent; a single fitted slope hides bends.
    let half = upper / 2;
    let (s_head, _) = linear_fit(&xs[..half.max(2)], &ys[..half.max(2)]);
    let (s_tail, _) = linear_fit(&xs[half..], &ys[half..]);
    println!(
        "source: {} domains, span {span}, window {width}, stride {stride}",
        d_count
    );
    println!(
        "  types {}  top-10 {:.1}%  top-100 {:.1}%  slope {:.3}  (head {:.3} / tail {:.3})",
        ranked.len(),
        100.0 * ranked.iter().take(10).sum::<f64>() / total,
        100.0 * ranked.iter().take(100).sum::<f64>() / total,
        slope,
        s_head,
        s_tail
    );
    if (s_head - s_tail).abs() > 0.35 {
        println!("  !! head and tail exponents disagree -- this is not a clean power law");
    }

    // The other half of the tilt trade-off. A tilt small enough to leave the
    // marginal a clean power law also leaves the conditional uninformative, and
    // then the current symbol says nothing about the next one -- a source with
    // a beautiful exponent and no task in it. Both numbers have to be read
    // together, so both are printed together.
    {
        let v = cfg.vocab;
        let mut uni = vec![0.0f64; v];
        let mut joint = vec![0.0f64; v * v];
        for w in stream.windows(2) {
            uni[w[0] as usize] += 1.0;
            joint[w[0] as usize * v + w[1] as usize] += 1.0;
        }
        let n: f64 = uni.iter().sum();
        let mut marg = vec![0.0f64; v];
        for &tok in &stream[1..] {
            marg[tok as usize] += 1.0;
        }
        let h1: f64 = marg
            .iter()
            .filter(|c| **c > 0.0)
            .map(|c| {
                let p = c / n;
                -p * p.log2()
            })
            .sum();
        let mut h2 = 0.0;
        for a in 0..v {
            if uni[a] <= 0.0 {
                continue;
            }
            let pa = uni[a] / n;
            let row = &joint[a * v..(a + 1) * v];
            let mut inner = 0.0;
            for c in row.iter().filter(|c| **c > 0.0) {
                let p = c / uni[a];
                inner -= p * p.log2();
            }
            h2 += pa * inner;
        }
        println!(
            "  entropy: marginal {h1:.3} bits, conditional {h2:.3} bits, mutual information {:.3}",
            h1 - h2
        );
    }
    if cfg.source_only {
        return Ok(());
    }

    let mut codes = vec![0.0; cfg.vocab * cfg.d_model];
    for t in 0..cfg.vocab {
        rng.fill_unit_vector(&mut codes[t * cfg.d_model..(t + 1) * cfg.d_model]);
    }

    let mut arm = if cfg.ewc {
        Arm::Ewc(Box::new(OnlineEwc::new(
            cfg.vocab,
            cfg.d_model,
            cfg.eta,
            1.0 / span as f64,
        )))
    } else {
        Arm::Tree(Box::new(Tree::new(annp_core::tree::Spec {
            vocab: cfg.vocab,
            d: cfg.d_model,
            fanout: cfg.fanout,
            depth_max: cfg.depth,
            rungs: cfg.rungs,
            r: cfg.ladder_r,
            g1: 1.0 / span as f64,
            eta: cfg.eta,
        })))
    };

    // Re-entry cost against the settled cost of the same visit. Absolute
    // re-entry alone cannot be read: it falls as the model improves at
    // everything, which looks like retention and is not.
    // Loss stratified by the target's global rank.
    //
    // A Zipf marginal puts most of the mass in the head, so an aggregate bits
    // per token is very nearly a measurement of the head alone. The ladder's
    // mechanism is protecting rare items from being overwritten by frequent
    // ones, which is a tail effect by construction, and E0-b's win was reported
    // as a tail metric -- usable rank 205 against 96, recall over ranks 22-45 of
    // 99% against 61%. Judging it on an average was judging it where it does
    // not act. Bands are log spaced because the distribution is.
    let mut rank_of = vec![0usize; cfg.vocab];
    {
        let mut order: Vec<(usize, f64)> = counts_by_id.iter().copied().enumerate().collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (r, (id, _)) in order.into_iter().enumerate() {
            rank_of[id] = r + 1;
        }
    }
    let band_of = |id: u32| -> usize {
        match rank_of[id as usize] {
            0..=4 => 0,
            5..=16 => 1,
            17..=64 => 2,
            _ => 3,
        }
    };
    const BANDS: usize = 4;
    const BAND_NAME: [&str; BANDS] = ["rank 1-4", "rank 5-16", "rank 17-64", "rank 65+"];
    let (mut band_s, mut band_n) = ([0.0f64; BANDS], [0.0f64; BANDS]);
    let (mut band_re, mut band_re_n) = ([0.0f64; BANDS], [0.0f64; BANDS]);
    let (mut band_se, mut band_se_n) = ([0.0f64; BANDS], [0.0f64; BANDS]);

    let probe = (span / 50).clamp(1, 64);
    let visits = cfg.tokens / (span * d_count) + 1;
    let (mut reentry, mut re_n) = (vec![0.0f64; visits], vec![0.0f64; visits]);
    let (mut settled, mut se_n) = (vec![0.0f64; visits], vec![0.0f64; visits]);
    let started = std::time::Instant::now();
    let mut prev: Option<u32> = None;
    for (i, &tok) in stream.iter().enumerate() {
        let key: &[f64] = match prev {
            Some(p) => &codes[p as usize * cfg.d_model..(p as usize + 1) * cfg.d_model],
            None => &codes[..cfg.d_model],
        };
        let nats = match &mut arm {
            Arm::Tree(t) => t.observe(key, tok),
            Arm::Ewc(e) => e.observe(key, tok),
        };
        prev = Some(tok);
        let b = band_of(tok);
        band_s[b] += nats;
        band_n[b] += 1.0;
        let visit = i / (span * d_count);
        if visit < visits {
            if i % span < probe {
                reentry[visit] += nats;
                re_n[visit] += 1.0;
                band_re[b] += nats;
                band_re_n[b] += 1.0;
            } else if i % span >= span - probe {
                settled[visit] += nats;
                se_n[visit] += 1.0;
                band_se[b] += nats;
                band_se_n[b] += 1.0;
            }
        }
    }
    let elapsed = started.elapsed();
    let bits = std::f64::consts::LOG2_E;

    println!();
    match &arm {
        Arm::Tree(t) => {
            println!(
                "=== growing tree, fanout {} depth {} rungs {} r {} ===",
                cfg.fanout, cfg.depth, cfg.rungs, cfg.ladder_r
            );
            println!(
                "  nodes {} / capacity {}   width by depth {:?}",
                t.live_nodes(),
                t.capacity(),
                t.width_by_depth()
            );
            println!(
                "  live parameters {}   state held {}   MACs/token {}",
                t.live_parameters(),
                t.state_held(),
                t.cost()
            );
            let per = t.per_level_bits();
            println!(
                "  bits by depth (the division-of-labour check):\n    {}",
                per.iter()
                    .map(|b| format!("{b:.3}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            );
            if per.len() > 1 && per[1] >= per[0] - 0.01 {
                println!(
                    "  !! level 1 does not improve on level 0 -- the tree is not dividing labour"
                );
            }
        }
        Arm::Ewc(e) => {
            let (_, lam) = e.best();
            println!("=== online EWC (continual-learning control) ===");
            println!("  best lambda {lam}");
            println!(
                "  live parameters {}   state held {}   MACs/token {}",
                e.parameters(),
                e.state_held(),
                e.cost()
            );
            if e.lambda_at_boundary() {
                println!("  !! the winning lambda is an endpoint of the swept set");
            }
        }
    }
    println!(
        "  {:.2} s, {:.0} tokens/s",
        elapsed.as_secs_f64(),
        stream.len() as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("  band          share   overall   re-entry   settled       gap");
    for b in 0..BANDS {
        if band_n[b] < 1.0 {
            continue;
        }
        let share = 100.0 * band_n[b] / stream.len() as f64;
        let overall = band_s[b] / band_n[b] * bits;
        let (r, s) = (
            if band_re_n[b] > 0.0 {
                band_re[b] / band_re_n[b] * bits
            } else {
                f64::NAN
            },
            if band_se_n[b] > 0.0 {
                band_se[b] / band_se_n[b] * bits
            } else {
                f64::NAN
            },
        );
        println!(
            "  {:<12} {:>5.1}%  {:>8.4}   {:>8.4}  {:>8.4}  {:>+8.4}",
            BAND_NAME[b],
            share,
            overall,
            r,
            s,
            r - s
        );
    }
    println!();
    println!("  visit   re-entry   settled       gap  <- read the trajectory");
    for v in 0..visits {
        if re_n[v] < 1.0 || se_n[v] < 1.0 {
            continue;
        }
        let r = reentry[v] / re_n[v] * bits;
        let s = settled[v] / se_n[v] * bits;
        println!("  {:>5}   {:>8.4}  {:>8.4}  {:>+8.4}", v + 1, r, s, r - s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marginal_is_heavy_tailed() {
        // The marginal is what E0-b's result depends on. A conditional-only
        // Zipf leaves the marginal flat and cannot test the mechanism.
        let mut rng = Rng::new(5);
        let mut c = ZipfChain::new(128, 0, 128, 1.0, 1.0, &mut rng);
        let mut draw = Rng::new(6);
        let mut counts = vec![0.0f64; 128];
        for _ in 0..400_000 {
            counts[c.next(&mut draw) as usize] += 1.0;
        }
        counts.sort_by(|a, b| b.total_cmp(a));
        let total: f64 = counts.iter().sum();
        let top10 = counts.iter().take(10).sum::<f64>() / total;
        assert!(
            top10 > 0.25,
            "top ten symbols hold {:.1}%, that is not heavy tailed",
            100.0 * top10
        );
        assert!(
            counts[0] / counts[1] > 1.3,
            "rank1/rank2 = {:.2}",
            counts[0] / counts[1]
        );
    }

    #[test]
    fn ewc_penalty_actually_restrains_the_weights() {
        // With a large lambda the weights must move less than with none, or the
        // control is not a control.
        let (v, d) = (16usize, 8usize);
        let mut rng = Rng::new(3);
        let mut codes = vec![0.0; v * d];
        for t in 0..v {
            rng.fill_unit_vector(&mut codes[t * d..(t + 1) * d]);
        }
        let mut e = OnlineEwc::new(v, d, 0.3, 0.01);
        let mut s = Rng::new(4);
        for _ in 0..4000 {
            let a = s.next_below(v as u64) as u32;
            let b = s.next_below(v as u64) as u32;
            e.observe(&codes[a as usize * d..(a as usize + 1) * d], b);
        }
        let free: f64 = e.w[0].iter().map(|x| x * x).sum();
        let held: f64 = e.w[3].iter().map(|x| x * x).sum();
        assert!(
            held < free,
            "lambda 100 did not restrain anything: {held} vs {free}"
        );
    }
}
