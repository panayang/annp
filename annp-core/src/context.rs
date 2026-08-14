//! Rotation-addressed context, persisted by a consolidation ladder.
//!
//! `DESIGN-NEXT.md` §4: long context is persistence times addressability. The
//! ladder supplies the first factor and only the first. The architecture that
//! `DESIGN.md` records hung a 65536-event ladder on every node and reached
//! three tokens — an expensive proof that one factor is not the product.
//!
//! The second factor comes from a group. Encode stream position `t` as a
//! block-diagonal rotation `R(t) = ⊕_j R(θ_j t)`. Then
//!
//! ```text
//!     R(t)⁻¹ R(t − ℓ) = R(−ℓ)
//! ```
//!
//! and the right-hand side does not mention `t`. **A lag is a fixed group
//! element**: addressing ℓ steps back is one rotation, at a cost independent of
//! ℓ. Reach stops being whatever falls out of a stochastic process and becomes
//! a rotation you choose to apply — design invariant ① of `DESIGN-NEXT.md` §3.
//!
//! The accumulator is one recurrence and needs no window:
//!
//! ```text
//!     c_t = R(θ) c_{t−1} + v(x_t)
//! ```
//!
//! A token written at time `s` sits at phase `R(θ(t−s))` at time `t` no matter
//! which rung it has since diffused into, because every rung is rotated
//! together and the diffusion is coordinate-wise. **Phase carries the address,
//! the ladder carries the amplitude, and the two do not interfere** — see
//! `rotation_commutes_with_diffusion`.
//!
//! Nothing here is trained. The write codes are fixed, so no gradient crosses
//! the recurrence and no backpropagation-through-time is implied: the readout
//! is the only thing that learns.

use crate::ladder::{Ladder, Schedule};
use crate::linalg::Mat;
use crate::rng::Rng;

/// A block-diagonal rotation on `d` coordinates, `d/2` planes.
///
/// Frequencies are derived, never tuned. The periods run geometrically from 2
/// to `horizon`: below 2 nothing is resolvable — one step per sample is the
/// Nyquist limit and a period-2 plane already flips sign every token — and
/// above `horizon` a plane cannot complete a turn inside the window we claim to
/// cover. Because the periods are geometrically spread and mutually
/// incommensurate, the joint phase identifies a lag over a range set by their
/// product, which is how `d/2` planes address exponentially many lags.
#[derive(Clone, Debug)]
pub struct Rotation {
    theta: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

/// How the `d/2` frequencies are allocated. This is a real fork, not a
/// preference, and it is left measurable rather than guessed.
///
/// Unbinding at a lag error `Δ` leaves correlation `(1/J) Σ_j cos(θ_j Δ)`, so
/// resolution is decided entirely by how the `θ_j` are spread:
///
/// - `Geometric` spreads them uniformly in log-period from 2 to `horizon`.
///   Range is the product of the periods — exponential in `J` — but planes
///   whose period far exceeds `Δ` all contribute ≈1, so nearby lags stay
///   confusable. Resolution is constant in `Δ/ℓ`, not in `Δ`.
/// - `Linear` spreads them uniformly in frequency, making the sum a Dirichlet
///   kernel: sharp at every lag, but unambiguous only out to about `J`.
///
/// Real text puts most of its mutual information at lags 1–3, which argues for
/// `Linear`; the long-context claim argues for `Geometric`. Measure it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spacing {
    Geometric,
    Linear,
}

impl Rotation {
    pub fn for_horizon(d: usize, horizon: f64) -> Self {
        Self::allocate(d, horizon, Spacing::Geometric)
    }

    pub fn allocate(d: usize, horizon: f64, spacing: Spacing) -> Self {
        assert!(d >= 2 && d % 2 == 0, "d must be even and at least 2");
        assert!(horizon > 2.0, "horizon must exceed the Nyquist period of 2");
        let blocks = d / 2;
        let theta: Vec<f64> = (0..blocks)
            .map(|j| {
                let f = if blocks == 1 {
                    0.0
                } else {
                    j as f64 / (blocks - 1) as f64
                };
                match spacing {
                    // Periods geometric from the Nyquist period of 2 up to the
                    // horizon: nothing is resolvable faster than one step per
                    // sample, and nothing slower than the horizon can complete
                    // a turn inside the window we claim to cover.
                    Spacing::Geometric => {
                        std::f64::consts::TAU / (2.0 * (horizon / 2.0).powf(f))
                    }
                    // Frequencies uniform on (0, pi]. Index j = blocks - 1 lands
                    // exactly on Nyquist; j = 0 is the slowest that still turns.
                    Spacing::Linear => {
                        std::f64::consts::PI * (j + 1) as f64 / blocks as f64
                    }
                }
            })
            .collect();
        Self::from_theta(theta)
    }

    /// The identity, for the ablation that removes addressing entirely.
    pub fn identity(d: usize) -> Self {
        assert!(d >= 2 && d % 2 == 0, "d must be even and at least 2");
        Self::from_theta(vec![0.0; d / 2])
    }

    fn from_theta(theta: Vec<f64>) -> Self {
        let cos = theta.iter().map(|t| t.cos()).collect();
        let sin = theta.iter().map(|t| t.sin()).collect();
        Self { theta, cos, sin }
    }

    #[inline]
    pub fn planes(&self) -> usize {
        self.theta.len()
    }

    /// Advance one step: `v ← R(θ) v`.
    #[inline]
    pub fn step(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), 2 * self.theta.len());
        for (j, (&c, &s)) in self.cos.iter().zip(&self.sin).enumerate() {
            let (a, b) = (v[2 * j], v[2 * j + 1]);
            v[2 * j] = c * a - s * b;
            v[2 * j + 1] = s * a + c * b;
        }
    }

    /// Apply `R(θ·k)` for any integer `k`, positive or negative. `k = -ℓ` is
    /// the group element that addresses lag ℓ.
    pub fn by(&self, k: i64, v: &mut [f64]) {
        debug_assert_eq!(v.len(), 2 * self.theta.len());
        for (j, &t) in self.theta.iter().enumerate() {
            let ang = t * k as f64;
            let (s, c) = ang.sin_cos();
            let (a, b) = (v[2 * j], v[2 * j + 1]);
            v[2 * j] = c * a - s * b;
            v[2 * j + 1] = s * a + c * b;
        }
    }
}

/// How the accumulator persists between tokens.
#[derive(Clone, Debug)]
enum Persist {
    /// One vector with per-token multiplicative decay: exponential forgetting
    /// with characteristic time `1/(1 − decay)`. This is the control that
    /// isolates the ladder — `DESIGN-NEXT.md` §5 candidate C.
    Single { u: Mat, decay: f64 },
    /// The consolidation ladder: power-law forgetting, `t^-1/2` measured over
    /// six decades.
    Ladder(Ladder),
}

/// Fixed write codes, a rotation, and a persistence rule. Nothing is trained.
#[derive(Clone, Debug)]
pub struct Context {
    rot: Rotation,
    persist: Persist,
    /// `vocab * d`, unit-norm, fixed at construction.
    codes: Vec<f64>,
    d: usize,
}

impl Context {
    /// `horizon` sizes both halves: it fixes the slowest rotation period and,
    /// through `rungs_for_horizon`, the number of ladder rungs. One number
    /// drives persistence and addressing together, which is the whole claim.
    pub fn new(
        vocab: usize,
        d: usize,
        horizon: f64,
        ladder: bool,
        spacing: Spacing,
        rng: &mut Rng,
    ) -> Self {
        let rot = Rotation::allocate(d, horizon, spacing);
        Self::with_rotation(vocab, d, horizon, ladder, rot, rng)
    }

    /// As `new`, but with addressing removed. Used for the 2×2 ablation.
    pub fn without_addressing(vocab: usize, d: usize, horizon: f64, ladder: bool, rng: &mut Rng) -> Self {
        Self::with_rotation(vocab, d, horizon, ladder, Rotation::identity(d), rng)
    }

    fn with_rotation(
        vocab: usize,
        d: usize,
        horizon: f64,
        ladder: bool,
        rot: Rotation,
        rng: &mut Rng,
    ) -> Self {
        assert!(vocab >= 1 && d >= 2 && d % 2 == 0);
        let schedule = Schedule::Geometric { r: 2.0, g1: 0.25 };
        let persist = if ladder {
            let m = schedule.rungs_for_horizon(horizon);
            Persist::Ladder(Ladder::new(schedule, m, d, 1))
        } else {
            // Matched control: the same nominal horizon, an exponential kernel
            // instead of a power-law one. Characteristic time `1/(1 − decay)`
            // is set to `horizon` so the arms differ in kernel shape, not span.
            Persist::Single {
                u: Mat::zeros(d, 1),
                decay: 1.0 - 1.0 / horizon,
            }
        };
        let mut codes = vec![0.0; vocab * d];
        for tok in 0..vocab {
            let slot = &mut codes[tok * d..(tok + 1) * d];
            for x in slot.iter_mut() {
                *x = rng.next_normal();
            }
            let n = slot.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in slot.iter_mut() {
                *x /= n;
            }
        }
        Self {
            rot,
            persist,
            codes,
            d,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.d
    }

    pub fn rungs(&self) -> usize {
        match &self.persist {
            Persist::Single { .. } => 1,
            Persist::Ladder(l) => l.num_rungs(),
        }
    }

    #[inline]
    pub fn code(&self, tok: u32) -> &[f64] {
        let t = tok as usize;
        &self.codes[t * self.d..(t + 1) * self.d]
    }

    /// Rotate the whole state one step, let it consolidate, then write the
    /// token. Rotating before the write is what puts a token observed now at
    /// phase zero and a token observed ℓ steps ago at phase `R(θℓ)`.
    pub fn observe(&mut self, tok: u32) {
        let d = self.d;
        match &mut self.persist {
            Persist::Single { u, decay } => {
                let s = u.as_mut_slice();
                self.rot.step(s);
                for x in s.iter_mut() {
                    *x *= *decay;
                }
                let code = &self.codes[tok as usize * d..(tok as usize + 1) * d];
                for (x, c) in s.iter_mut().zip(code) {
                    *x += c;
                }
            }
            Persist::Ladder(l) => {
                for rung in l.rungs_mut() {
                    self.rot.step(rung.as_mut_slice());
                }
                l.relax();
                let code = &self.codes[tok as usize * d..(tok as usize + 1) * d];
                l.inject(code, &[1.0], 1.0);
            }
        }
    }

    /// Advance persistence by one token without writing anything. Only the
    /// impulse-response measurements use this; the forward pass always writes.
    pub fn idle(&mut self) {
        match &mut self.persist {
            Persist::Single { u, decay } => {
                let s = u.as_mut_slice();
                self.rot.step(s);
                for x in s.iter_mut() {
                    *x *= *decay;
                }
            }
            Persist::Ladder(l) => {
                for rung in l.rungs_mut() {
                    self.rot.step(rung.as_mut_slice());
                }
                l.relax();
            }
        }
    }

    /// The vector the readout sees: rung 1, the same one the ladder's forward
    /// pass has always read.
    pub fn read(&self) -> &[f64] {
        match &self.persist {
            Persist::Single { u, .. } => u.as_slice(),
            Persist::Ladder(l) => l.rung(0).as_slice(),
        }
    }

    /// Unbind lag `lag` from the current state into `out`: apply `R(−θ·lag)`.
    /// Used by the lag probe, not by the forward pass.
    pub fn unbind(&self, lag: i64, out: &mut [f64]) {
        out.copy_from_slice(self.read());
        self.rot.by(-lag, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dot, least_squares};

    #[test]
    fn a_lag_is_the_same_group_element_wherever_you_stand() {
        // R(t)^-1 R(t - l) must not depend on t. This is the property the whole
        // design rests on, so it is pinned directly rather than inferred.
        let rot = Rotation::for_horizon(32, 1024.0);
        let lag = 37;
        let mut reference = vec![0.0; 32];
        reference[0] = 1.0;
        reference[5] = -0.5;
        let mut expected = reference.clone();
        rot.by(-lag, &mut expected);
        for t in [0i64, 1, 9, 100, 5000] {
            let mut v = reference.clone();
            rot.by(t - lag, &mut v);
            rot.by(-t, &mut v);
            for (a, b) in v.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-9, "t = {t} changed the lag operator");
            }
        }
    }

    #[test]
    fn rotation_preserves_length() {
        let rot = Rotation::for_horizon(16, 500.0);
        let mut v: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).sin()).collect();
        let before = dot(&v, &v);
        for _ in 0..1000 {
            rot.step(&mut v);
        }
        assert!((dot(&v, &v) - before).abs() < 1e-9);
    }

    #[test]
    fn rotation_commutes_with_diffusion() {
        // The claim in the module header: because `relax` couples rungs
        // coordinate by coordinate with the same conductances at every
        // coordinate, rotating all rungs and then relaxing is the same as
        // relaxing and then rotating. If this failed, phase would depend on
        // which rung a write had diffused into and addressing would be wrong.
        let schedule = Schedule::Geometric { r: 2.0, g1: 0.25 };
        let rot = Rotation::for_horizon(8, 256.0);
        let mut rng = Rng::new(3);
        let seed: Vec<f64> = (0..8).map(|_| rng.next_normal()).collect();

        let mut a = Ladder::new(schedule, 4, 8, 1);
        a.inject(&seed, &[1.0], 1.0);
        let mut b = a.clone();

        for rung in a.rungs_mut() {
            rot.step(rung.as_mut_slice());
        }
        a.relax();

        b.relax();
        for rung in b.rungs_mut() {
            rot.step(rung.as_mut_slice());
        }

        for k in 0..a.num_rungs() {
            for (x, y) in a.rung(k).as_slice().iter().zip(b.rung(k).as_slice()) {
                assert!((x - y).abs() < 1e-12, "rung {k} differs");
            }
        }
    }

    #[test]
    fn unbinding_recovers_a_token_at_a_known_lag_against_a_foil() {
        // Exposure-matched: the foil is another token written into the same
        // stream, so a hit cannot come from one code being seen and the other
        // not. DIAGNOSIS.md §5.1 — uniform is the wrong null.
        let mut rng = Rng::new(11);
        let mut c = Context::new(64, 256, 4096.0, true, Spacing::Geometric, &mut rng);
        let target: u32 = 3;
        let foil: u32 = 4;
        let lag = 12;

        let mut written = Vec::new();
        for step in 0..40u32 {
            let tok = if step == 40 - 1 - lag as u32 {
                target
            } else if step == 5 {
                foil
            } else {
                8 + step % 40
            };
            written.push(tok);
            c.observe(tok);
        }
        assert_eq!(written[written.len() - 1 - lag as usize], target);

        let mut probe = vec![0.0; c.width()];
        c.unbind(lag as i64, &mut probe);
        let hit = dot(&probe, c.code(target));
        let miss = dot(&probe, c.code(foil));
        assert!(
            hit > 4.0 * miss.abs(),
            "unbinding did not separate target {hit} from foil {miss}"
        );
    }

    /// Writes a stream whose only occurrence of token 3 is at `lag`, then
    /// returns (unbind at the true lag, unbind at a wrong lag).
    fn lag_discrimination(mut c: Context, lag: usize, wrong: usize) -> (f64, f64) {
        for step in 0..40u32 {
            let tok = if step as usize == 40 - 1 - lag {
                3
            } else {
                8 + step % 40
            };
            c.observe(tok);
        }
        let mut probe = vec![0.0; c.width()];
        c.unbind(lag as i64, &mut probe);
        let right = dot(&probe, c.code(3));
        c.unbind(wrong as i64, &mut probe);
        (right, dot(&probe, c.code(3)))
    }

    /// Fits a linear decoder from the accumulator state to the code of the
    /// token `lag` steps back, then reports held-out top-1 accuracy against all
    /// other tokens as foils. This is what the readout head actually does, so
    /// it is the honest question: *is the lag content linearly decodable*.
    fn decode_at_lag(mut c: Context, lag: usize, vocab: u32) -> f64 {
        let (train, test) = (800usize, 200usize);
        let mut rng = Rng::new(4242);
        let (mut states, mut targets) = (Vec::new(), Vec::new());
        let mut seen: Vec<u32> = Vec::new();
        for _ in 0..(train + test + lag + 64) {
            let tok = rng.next_below(vocab as u64) as u32;
            c.observe(tok);
            seen.push(tok);
            if seen.len() > lag {
                states.push(c.read().to_vec());
                targets.push(seen[seen.len() - 1 - lag]);
            }
        }
        let d = c.width();
        // Ridge by data augmentation: appending `sqrt(lambda) * e_i` rows with
        // zero targets turns the normal equations into `A^T A + lambda I`. The
        // accumulator's coordinates are strongly correlated by construction --
        // it is a smooth recurrence -- so the unregularised system is singular
        // to the pivot tolerance. Both arms get the same lambda, so this cannot
        // favour either.
        let lambda: f64 = 1e-6;
        let mut design: Vec<Vec<f64>> = states[..train].to_vec();
        for i in 0..d {
            let mut row = vec![0.0; d];
            row[i] = lambda.sqrt();
            design.push(row);
        }
        // One linear score per token, argmax over them. This *is* a readout
        // head with the nonlinearity removed, so it measures the thing the real
        // head will have to do.
        let weights: Vec<Vec<f64>> = (0..vocab)
            .map(|v| {
                let mut y: Vec<f64> = targets[..train]
                    .iter()
                    .map(|&t| if t == v { 1.0 } else { 0.0 })
                    .collect();
                y.resize(y.len() + d, 0.0);
                least_squares(&design, &y).expect("ridge system is nonsingular")
            })
            .collect();

        let mut hits = 0.0;
        for i in train..train + test {
            let score = |w: &Vec<f64>| -> f64 {
                w.iter().zip(&states[i]).map(|(a, b)| a * b).sum()
            };
            let best = (0..vocab as usize)
                .max_by(|&x, &y| score(&weights[x]).total_cmp(&score(&weights[y])))
                .unwrap();
            if best as u32 == targets[i] {
                hits += 1.0;
            }
        }
        hits / test as f64
    }

    #[test]
    #[ignore = "measurement, not a gate; run with --ignored --nocapture"]
    fn capacity_curve() {
        println!("\n  d   horizon  lag  addressed  flat   (chance 0.125)");
        for d in [64usize, 128, 256] {
            for horizon in [8.0f64, 16.0, 64.0, 256.0] {
                for lag in [1usize, 5] {
                    let a = decode_at_lag(
                        Context::new(8, d, horizon, true, Spacing::Geometric, &mut Rng::new(11)),
                        lag,
                        8,
                    );
                    let f = decode_at_lag(
                        Context::without_addressing(8, d, horizon, true, &mut Rng::new(11)),
                        lag,
                        8,
                    );
                    println!("{d:5} {horizon:8} {lag:4}   {a:6.3}   {f:6.3}");
                }
            }
        }
    }

    #[test]
    fn a_linear_decoder_recovers_a_chosen_lag_only_when_addressing_is_on() {
        // Naive correlation unbinding is a poor decoder -- with geometric
        // periods spanning 2..horizon, most planes turn too slowly to separate
        // nearby lags and they swamp the sum. That says nothing about whether
        // the information is present, only that uniform weights cannot reach
        // it. A fitted linear map can, and a fitted linear map is exactly what
        // the readout head is.
        let vocab = 8;
        let chance = 1.0 / vocab as f64;
        let lag = 5;
        let addressed = decode_at_lag(
            Context::new(vocab as usize, 128, 64.0, true, Spacing::Geometric, &mut Rng::new(11)),
            lag,
            vocab,
        );
        let flat = decode_at_lag(
            Context::without_addressing(vocab as usize, 128, 64.0, true, &mut Rng::new(11)),
            lag,
            vocab,
        );
        assert!(
            addressed > 0.6,
            "addressed decoder should be well above chance {chance}, got {addressed}"
        );
        assert!(
            flat < addressed / 2.0,
            "without addressing the lag should be far less decodable: \
             {flat} vs {addressed}"
        );
    }

    #[test]
    fn the_ladder_trades_near_amplitude_for_a_far_tail() {
        // Same nominal horizon, different kernel shape, addressing off in both
        // so the only difference is persistence. The trade runs in BOTH
        // directions and the test pins both, because pinning only the flattering
        // half is how a design gets oversold:
        //
        //   t much less than the horizon -> the exponential is *higher*. A
        //     `t^-1/2` law has already given up 97% of its amplitude by t=1000
        //     while `exp(-t/4096)` still holds 78%.
        //   t much greater than the horizon -> the ladder is higher, and not by
        //     a slope but by a floor: the chain is closed, so it relaxes to the
        //     capacity-weighted mean rather than to zero.
        let horizon = 4096.0;
        let mut lad = Context::without_addressing(8, 64, horizon, true, &mut Rng::new(5));
        let mut exp = Context::without_addressing(8, 64, horizon, false, &mut Rng::new(5));
        lad.observe(1);
        exp.observe(1);

        for _ in 0..1000 {
            lad.idle();
            exp.idle();
        }
        let (near_l, near_e) = (dot(lad.read(), lad.code(1)), dot(exp.read(), exp.code(1)));
        assert!(
            near_e > 5.0 * near_l,
            "inside the horizon the exponential should lead: ladder {near_l} vs single {near_e}"
        );

        for _ in 0..39_000 {
            lad.idle();
            exp.idle();
        }
        let (far_l, far_e) = (dot(lad.read(), lad.code(1)), dot(exp.read(), exp.code(1)));
        assert!(
            far_l > 50.0 * far_e,
            "past the horizon the ladder should lead: ladder {far_l} vs single {far_e}"
        );
    }
}
