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

use annp_core::rng::Rng;

/// `in_dim -> hidden (tanh) -> vocab`, trained online, several learning rates
/// raced in parallel with the best reported.
pub struct Head {
    pub(crate) in_dim: usize,
    pub(crate) hidden: usize,
    pub(crate) vocab: usize,
    pub(crate) rates: Vec<f64>,
    /// `[rate][h * in_dim + k]`
    pub(crate) w1: Vec<Vec<f64>>,
    pub(crate) b1: Vec<Vec<f64>>,
    /// `[rate][token * hidden + h]`
    pub(crate) w2: Vec<Vec<f64>>,
    pub(crate) b2: Vec<Vec<f64>>,
    /// Accumulated code length per rate, in nats, and over the final tenth.
    pub(crate) nats: Vec<f64>,
    pub(crate) tail: Vec<f64>,
    h: Vec<f64>,
    logits: Vec<f64>,
    gh: Vec<f64>,
    /// `dL/dx` for the last `step`, taken against the pre-update `w1`. A front
    /// end with learned inputs applies this; one with fixed codes ignores it.
    pub(crate) gx: Vec<f64>,
}

impl Head {
    pub fn new(in_dim: usize, hidden: usize, vocab: usize, rng: &mut Rng) -> Self {
        assert!(in_dim >= 1 && hidden >= 1 && vocab >= 1);
        let rates = vec![0.003, 0.01, 0.03, 0.1];
        let n = rates.len();
        let draw = |count: usize, sigma: f64, rng: &mut Rng| -> Vec<f64> {
            (0..count).map(|_| rng.next_normal() * sigma).collect()
        };
        let w1 = (0..n)
            .map(|_| draw(hidden * in_dim, 1.0 / (in_dim as f64).sqrt(), rng))
            .collect();
        let w2 = (0..n)
            .map(|_| draw(vocab * hidden, 1.0 / (hidden as f64).sqrt(), rng))
            .collect();
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
            h: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            gh: vec![0.0; hidden],
            gx: vec![0.0; in_dim],
            rates,
        }
    }

    pub fn num_rates(&self) -> usize {
        self.rates.len()
    }

    pub fn rate(&self, r: usize) -> f64 {
        self.rates[r]
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
        for j in 0..hid {
            let row = &self.w1[r][j * n_in..(j + 1) * n_in];
            let z: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + self.b1[r][j];
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
        self.gx.iter_mut().for_each(|g| *g = 0.0);
        for j in 0..hid {
            let gz = self.gh[j] * (1.0 - self.h[j] * self.h[j]);
            if gz == 0.0 {
                continue;
            }
            let row = &mut self.w1[r][j * n_in..(j + 1) * n_in];
            for (k, (weight, &xv)) in row.iter_mut().zip(x).enumerate() {
                // Read the weight into `gx` before it moves, so the reported
                // input gradient belongs to this forward pass.
                self.gx[k] += gz * *weight;
                *weight -= lr * gz * xv;
            }
            self.b1[r][j] -= lr * gz;
        }
        nats
    }

    pub fn charge(&mut self, r: usize, nats: f64, in_tail: bool) {
        self.nats[r] += nats;
        if in_tail {
            self.tail[r] += nats;
        }
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
