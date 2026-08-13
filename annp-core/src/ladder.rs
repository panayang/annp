//! Multi-timescale consolidation ladder, and the associative memory that sits
//! on top of it.
//!
//! A weight matrix is replaced by a chain of same-shaped variables `U_1..U_m`
//! coupled by diffusion:
//!
//! ```text
//!   C_1 dU_1/dt = input          + g_1 (U_2 - U_1)
//!   C_k dU_k/dt = g_{k-1}(U_{k-1} - U_k) + g_k (U_{k+1} - U_k)
//!   C_m dU_m/dt = g_{m-1}(U_{m-1} - U_m)
//! ```
//!
//! All updates enter `U_1`; the forward pass reads `U_1` only, so the ladder
//! costs memory but not forward compute. Consolidation is not a mechanism that
//! decides what to keep — it is diffusion. A one-off write is pulled back by
//! `U_2` and vanishes; a repeated write pushes `U_2` the same way each time and
//! survives. Repetition is the filter, and it is free.
//!
//! Time is discrete with `dt = 1` (one event), integrated with explicit Euler.
//! `Ladder::new` rejects schedules that are not stable at that step size rather
//! than integrating something subtly wrong.

use crate::linalg::{Mat, dot, norm, sub_into};

/// How capacities `C_k` and conductances `g_k` scale with rung index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Schedule {
    /// `C_k = 1`, `g_k = g`. This is the plain discretised 1-D diffusion
    /// equation, whose Green's function at the origin decays as `t^-1/2`. It
    /// reaches that exponent honestly but only covers timescales up to
    /// `~m^2/g`, so covering a long horizon costs `O(sqrt(horizon))` rungs.
    Uniform { g: f64 },
    /// `C_k = r^(k-1)`, `g_k = g1 * r^-(k-1)`, hence `tau_k ~ r^(2(k-1))/g1`.
    /// Covers `r^(2m)` of dynamic range with `m` rungs, against `m^2` for
    /// `Uniform`, and reaches the same `t^-1/2` exponent.
    ///
    /// Keep `r` in `2..=8`. E0-d measured the impulse response over six decades
    /// and found `-0.4991` at `r = 2` and `-0.4993` at `r = 8`, but `-0.5203` at
    /// `r = 16` and `-0.4597` at `r = 32`: past 8 the chain is no longer a good
    /// approximation to the diffusion. (An earlier version of this comment
    /// claimed a log-periodic ripple of period `ln r`. There is none — see
    /// DESIGN.md §8.4.)
    Geometric { r: f64, g1: f64 },
}

impl Schedule {
    /// Rungs needed to cover `horizon` events, from `m = 1 + ln(T)/(2 ln r)`.
    ///
    /// `m` is derived from the retention horizon, not tuned. For `Uniform` the
    /// diffusive scaling `tau ~ m^2/g` gives `m = sqrt(horizon * g)` instead.
    pub fn rungs_for_horizon(&self, horizon: f64) -> usize {
        assert!(horizon > 1.0, "horizon must exceed one event");
        let m = match *self {
            Schedule::Uniform { g } => (horizon * g).sqrt(),
            Schedule::Geometric { r, g1 } => {
                assert!(r > 1.0, "geometric ratio must exceed 1");
                1.0 + (horizon * g1).ln() / (2.0 * r.ln())
            }
        };
        (m.round() as usize).max(2)
    }

    fn capacity(&self, k: usize) -> f64 {
        match *self {
            Schedule::Uniform { .. } => 1.0,
            Schedule::Geometric { r, .. } => r.powi(k as i32),
        }
    }

    /// Conductance between rung `k` and rung `k+1` (both zero-based).
    fn conductance(&self, k: usize) -> f64 {
        match *self {
            Schedule::Uniform { g } => g,
            Schedule::Geometric { r, g1 } => g1 * r.powi(-(k as i32)),
        }
    }
}

/// A chain of coupled same-shaped matrices. `rung(0)` is the one that is read
/// and written; the rest are hidden state.
#[derive(Clone, Debug)]
pub struct Ladder {
    schedule: Schedule,
    rungs: Vec<Mat>,
    capacity: Vec<f64>,
    inv_capacity: Vec<f64>,
    /// `conductance[k]` couples rung `k` to rung `k+1`; length `m - 1`.
    conductance: Vec<f64>,
    n: usize,
    flux_prev: Vec<f64>,
    flux_cur: Vec<f64>,
}

impl Ladder {
    pub fn new(schedule: Schedule, m: usize, rows: usize, cols: usize) -> Self {
        assert!(m >= 2, "a ladder needs at least two rungs");
        let capacity: Vec<f64> = (0..m).map(|k| schedule.capacity(k)).collect();
        let conductance: Vec<f64> = (0..m - 1).map(|k| schedule.conductance(k)).collect();
        assert!(
            capacity.iter().all(|c| *c > 0.0),
            "capacities must be positive"
        );
        assert!(
            conductance.iter().all(|g| *g > 0.0),
            "conductances must be positive"
        );

        // Explicit Euler at dt = 1 is monotone only if each rung's total
        // outflow rate stays below 1. Above 1 the integration oscillates and
        // above 2 it diverges; either way the measurement is meaningless, so
        // refuse rather than produce numbers.
        for k in 0..m {
            let below = if k > 0 { conductance[k - 1] } else { 0.0 };
            let above = if k + 1 < m { conductance[k] } else { 0.0 };
            let rate = (below + above) / capacity[k];
            assert!(
                rate < 1.0,
                "unstable schedule {schedule:?}: rung {} has outflow rate {rate} >= 1 at dt = 1",
                k + 1
            );
        }

        let n = rows * cols;
        Self {
            schedule,
            rungs: (0..m).map(|_| Mat::zeros(rows, cols)).collect(),
            inv_capacity: capacity.iter().map(|c| 1.0 / c).collect(),
            capacity,
            conductance,
            n,
            flux_prev: vec![0.0; n],
            flux_cur: vec![0.0; n],
        }
    }

    #[inline]
    pub fn num_rungs(&self) -> usize {
        self.rungs.len()
    }

    #[inline]
    pub fn rung(&self, k: usize) -> &Mat {
        &self.rungs[k]
    }

    #[inline]
    pub fn capacity(&self, k: usize) -> f64 {
        self.capacity[k]
    }

    /// Characteristic time of rung `k` (zero-based).
    ///
    /// For `Geometric` this is `C_k / g_k`, the time for the rung to equilibrate
    /// with the next one. For `Uniform` every rung has the same local time
    /// constant, so the meaningful quantity is the diffusive time to reach depth
    /// `k` from the input, `~(k+1)^2 / g`.
    pub fn relaxation_time(&self, k: usize) -> f64 {
        match self.schedule {
            Schedule::Uniform { g } => ((k + 1) * (k + 1)) as f64 / g,
            Schedule::Geometric { .. } => {
                self.capacity[k] / self.conductance[k.min(self.conductance.len() - 1)]
            }
        }
    }

    /// `C`-weighted total, the conserved quantity of the diffusion.
    pub fn conserved_total(&self) -> f64 {
        self.rungs
            .iter()
            .zip(&self.capacity)
            .map(|(u, c)| c * u.as_slice().iter().sum::<f64>())
            .sum()
    }

    /// Writes `values` into every rung, leaving the chain at equilibrium.
    pub fn initialise(&mut self, values: &[f64]) {
        for rung in self.rungs.iter_mut() {
            assert_eq!(
                values.len(),
                rung.as_slice().len(),
                "initialise: size mismatch"
            );
            rung.as_mut_slice().copy_from_slice(values);
        }
    }

    /// `U_1 += (s / C_1) * a b^T`. Input is a flux into rung 1, so it is scaled
    /// by that rung's inverse capacity like every other term in its equation.
    pub fn inject(&mut self, a: &[f64], b: &[f64], s: f64) {
        self.rungs[0].add_outer(a, b, s * self.inv_capacity[0]);
    }

    /// Advances the diffusion by one event.
    pub fn relax(&mut self) {
        let Self {
            rungs,
            inv_capacity,
            conductance,
            n,
            flux_prev,
            flux_cur,
            ..
        } = self;
        let m = rungs.len();
        flux_prev.fill(0.0);

        for k in 0..m {
            // Flux from rung k+1 down into rung k, computed from pre-update
            // values throughout, which is what makes this explicit Euler.
            if k + 1 < m {
                let (lo, hi) = rungs.split_at_mut(k + 1);
                let cur = lo[k].as_slice();
                let next = hi[0].as_slice();
                let g = conductance[k];
                for e in 0..*n {
                    flux_cur[e] = g * (next[e] - cur[e]);
                }
            } else {
                flux_cur.fill(0.0);
            }

            let inv_c = inv_capacity[k];
            let cur = rungs[k].as_mut_slice();
            for e in 0..*n {
                cur[e] += inv_c * (flux_prev[e] + flux_cur[e]);
            }

            // What rung k received from above, rung k+1 gives up from below.
            std::mem::swap(flux_prev, flux_cur);
            for x in flux_prev.iter_mut() {
                *x = -*x;
            }
        }
    }
}

/// How a matrix's state persists between events.
#[derive(Clone, Debug)]
enum State {
    /// One matrix with per-event multiplicative decay. `decay == 1.0` is the
    /// no-forgetting baseline; `decay < 1.0` is exponential forgetting with
    /// characteristic time `1 / (1 - decay)`.
    Single {
        w: Mat,
        decay: f64,
    },
    Ladder(Ladder),
}

/// A delta-rule associative memory over a pluggable consolidation state.
///
/// The delta rule is written exactly once, here, so the baselines and the
/// ladder cannot accidentally differ in anything but their state dynamics.
#[derive(Clone, Debug)]
pub struct AssocMemory {
    state: State,
    pred: Vec<f64>,
    resid: Vec<f64>,
}

impl AssocMemory {
    /// Single matrix, no forgetting.
    pub fn plain(d: usize) -> Self {
        Self::single(d, 1.0)
    }

    /// Single matrix with per-event decay factor `decay` in `(0, 1]`.
    pub fn single(d: usize, decay: f64) -> Self {
        assert!(decay > 0.0 && decay <= 1.0, "decay must lie in (0, 1]");
        Self {
            state: State::Single {
                w: Mat::zeros(d, d),
                decay,
            },
            pred: vec![0.0; d],
            resid: vec![0.0; d],
        }
    }

    /// Rectangular single matrix, for measuring what the ladder costs.
    pub fn single_rect(rows: usize, cols: usize, decay: f64) -> Self {
        assert!(decay > 0.0 && decay <= 1.0, "decay must lie in (0, 1]");
        Self {
            state: State::Single {
                w: Mat::zeros(rows, cols),
                decay,
            },
            pred: vec![0.0; rows],
            resid: vec![0.0; rows],
        }
    }

    pub fn ladder(d: usize, schedule: Schedule, m: usize) -> Self {
        Self::ladder_rect(d, d, schedule, m)
    }

    /// Rectangular ladder. The output head's weight matrix is `vocab x d_model`
    /// and its gradient is a rank-one outer product, which `inject` already
    /// expresses, so it rides the same consolidation machinery as everything
    /// else with no special case.
    pub fn ladder_rect(rows: usize, cols: usize, schedule: Schedule, m: usize) -> Self {
        Self {
            state: State::Ladder(Ladder::new(schedule, m, rows, cols)),
            pred: vec![0.0; rows],
            resid: vec![0.0; rows],
        }
    }

    /// Writes `values` into **every** rung.
    ///
    /// Seeding only rung 1 would be a slow catastrophe: the chain conserves the
    /// capacity-weighted total, so an initial value in rung 1 alone diffuses
    /// outwards until every rung holds `1 / sum_k C_k` of it — a factor of 341
    /// for `r = 4, m = 5`. Initialising the chain at equilibrium means
    /// diffusion does nothing until real updates arrive.
    pub fn initialise(&mut self, values: &[f64]) {
        match &mut self.state {
            State::Single { w, .. } => {
                assert_eq!(
                    values.len(),
                    w.as_slice().len(),
                    "initialise: size mismatch"
                );
                w.as_mut_slice().copy_from_slice(values);
            }
            State::Ladder(l) => l.initialise(values),
        }
    }

    /// The matrix the forward pass sees: rung 1 for a ladder, the single matrix
    /// otherwise.
    pub fn read(&self) -> &Mat {
        self.read_at(0)
    }

    /// The rung the forward pass reads, clamped to what exists.
    ///
    /// Reading rung 0 only was a deliberate choice: the ladder was built as a
    /// *consolidation* mechanism, so the slow rungs exist to protect what was
    /// learned, not to be consulted. That decision was taken before anything
    /// measured how far back the assembled vector can still name a token, and
    /// §36 puts that at about three. Meanwhile `tau_k ~ r^(2(k-1))/g1` makes the
    /// rungs a geometric ladder of timescales — 2, 32, 512, 8192 writes at
    /// `r = 4, g1 = 0.5` — and a node takes roughly one write per token. The
    /// spectrum the reach is missing is therefore already being maintained and
    /// never read, which is worth measuring before it is worth arguing about.
    pub fn read_at(&self, rung: usize) -> &Mat {
        match &self.state {
            State::Single { w, .. } => w,
            State::Ladder(l) => l.rung(rung.min(l.num_rungs() - 1)),
        }
    }

    pub fn ladder_state(&self) -> Option<&Ladder> {
        match &self.state {
            State::Ladder(l) => Some(l),
            State::Single { .. } => None,
        }
    }

    /// `W += s * a b^T` into the fastest state. Used directly only by
    /// diagnostics that need a controlled input stream.
    pub fn inject(&mut self, a: &[f64], b: &[f64], s: f64) {
        match &mut self.state {
            State::Single { w, .. } => w.add_outer(a, b, s),
            State::Ladder(l) => l.inject(a, b, s),
        }
    }

    /// One delta-rule write, `W += eta * (value - W key) key^T`.
    ///
    /// Returns the surprise `||value - W key||`, the residual before the write.
    /// With a unit-norm key and `eta == 1` the write is exact in one shot.
    pub fn write(&mut self, key: &[f64], value: &[f64], eta: f64) -> f64 {
        let w = match &self.state {
            State::Single { w, .. } => w,
            State::Ladder(l) => l.rung(0),
        };
        w.mul_vec(key, &mut self.pred);
        sub_into(value, &self.pred, &mut self.resid);
        let surprise = norm(&self.resid);
        // `resid` is borrowed immutably here while `state` is borrowed mutably;
        // they are disjoint fields, so this is fine.
        match &mut self.state {
            State::Single { w, .. } => w.add_outer(&self.resid, key, eta),
            State::Ladder(l) => l.inject(&self.resid, key, eta),
        }
        surprise
    }

    /// Recall strength: `<value, W key>`.
    pub fn recall(&mut self, key: &[f64], value: &[f64]) -> f64 {
        let w = match &self.state {
            State::Single { w, .. } => w,
            State::Ladder(l) => l.rung(0),
        };
        w.mul_vec(key, &mut self.pred);
        dot(value, &self.pred)
    }

    /// Advances the state dynamics by one event.
    pub fn relax(&mut self) {
        match &mut self.state {
            State::Single { w, decay } => {
                if *decay != 1.0 {
                    w.scale(*decay);
                }
            }
            State::Ladder(l) => l.relax(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn geo(r: f64) -> Schedule {
        Schedule::Geometric { r, g1: 0.5 }
    }

    #[test]
    fn diffusion_conserves_the_capacity_weighted_total() {
        // The conserved quantity of the chain. If a sign or an index is wrong
        // in `relax`, this is what breaks first.
        let mut l = Ladder::new(geo(4.0), 6, 4, 4);
        let mut rng = Rng::new(3);
        let (mut a, mut b) = (vec![0.0; 4], vec![0.0; 4]);
        for _ in 0..5 {
            rng.fill_unit_vector(&mut a);
            rng.fill_unit_vector(&mut b);
            l.inject(&a, &b, 1.0);
        }
        let before = l.conserved_total();
        for _ in 0..10_000 {
            l.relax();
        }
        let after = l.conserved_total();
        assert!(
            (after - before).abs() < 1e-9 * before.abs().max(1.0),
            "{before} -> {after}"
        );
    }

    #[test]
    fn injection_adds_exactly_its_own_mass() {
        let mut l = Ladder::new(geo(2.0), 5, 3, 3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, -1.0, 2.0];
        // sum_ij (s a_i b_j) = s (sum a)(sum b), and C_1 = 1 so the flux enters
        // the conserved total undiminished.
        let expected = 0.75 * a.iter().sum::<f64>() * b.iter().sum::<f64>();
        l.inject(&a, &b, 0.75);
        assert!((l.conserved_total() - expected).abs() < 1e-12);
    }

    #[test]
    fn chain_equilibrates_to_the_capacity_weighted_mean() {
        let mut l = Ladder::new(geo(2.0), 4, 2, 2);
        l.inject(&[1.0, 0.0], &[1.0, 0.0], 1.0);
        let total = l.conserved_total();
        let c_sum: f64 = (0..l.num_rungs()).map(|k| l.capacity(k)).sum();
        // The slowest mode has a time constant of ~64 events here.
        for _ in 0..200_000 {
            l.relax();
        }
        // Every rung converges to the same value: total / sum_k C_k.
        let want = total / c_sum;
        for k in 0..l.num_rungs() {
            let got = l.rung(k).get(0, 0);
            assert!((got - want).abs() < 1e-9, "rung {k}: {got} != {want}");
        }
    }

    #[test]
    fn deep_rungs_actually_integrate() {
        // The f32 landmine, pinned exactly. The deepest rung of a geometric
        // ladder advances by g_{m-1}/C_m per event. For r = 4, m = 8 that is
        // ~7.5e-9, which is invisible against an O(1) rung in f32: the deep
        // rungs would freeze and the ladder would look inert. In f64 it moves.
        let (r, m) = (4.0f64, 8usize);
        let step = (0.5 * r.powi(-(m as i32 - 2))) / r.powi(m as i32 - 1);
        assert!(
            step < f32::EPSILON as f64,
            "test premise: {step} should be below f32 eps"
        );
        assert_eq!(
            1.0f32 + step as f32,
            1.0f32,
            "f32 cannot represent this increment"
        );

        let mut l = Ladder::new(geo(r), m, 4, 4);
        let mut rng = Rng::new(17);
        let (mut a, mut b) = (vec![0.0; 4], vec![0.0; 4]);
        rng.fill_unit_vector(&mut a);
        rng.fill_unit_vector(&mut b);
        l.inject(&a, &b, 1.0);

        let mut last = 0.0;
        for _ in 0..20_000 {
            l.relax();
            let deep = l.rung(m - 1).frob_sq();
            assert!(
                deep >= last,
                "deepest rung must fill monotonically from a single input"
            );
            last = deep;
        }
        assert!(last > 0.0, "deepest rung never integrated anything");
    }

    #[test]
    #[should_panic(expected = "unstable schedule")]
    fn unstable_schedule_is_rejected() {
        Ladder::new(Schedule::Uniform { g: 0.9 }, 4, 2, 2);
    }

    #[test]
    fn rung_count_follows_the_horizon_formula() {
        // m = round(1 + ln(T g1) / (2 ln r)), with g1 = 0.5 folded in because
        // tau_k = r^(2k) / g1, not r^(2k).
        let s = geo(4.0);
        let want = (1.0 + (15_000f64).ln() / (2.0 * 4f64.ln())).round() as usize;
        assert_eq!(s.rungs_for_horizon(30_000.0), want);
        // Larger ratio, fewer rungs for the same horizon.
        assert!(geo(8.0).rungs_for_horizon(30_000.0) < geo(2.0).rungs_for_horizon(30_000.0));
    }

    #[test]
    fn plain_memory_stores_in_one_shot_and_never_forgets() {
        let d = 12;
        let mut mem = AssocMemory::plain(d);
        let mut rng = Rng::new(41);
        let (mut k, mut v) = (vec![0.0; d], vec![0.0; d]);
        rng.fill_unit_vector(&mut k);
        rng.fill_unit_vector(&mut v);

        let surprise = mem.write(&k, &v, 1.0);
        assert!(
            (surprise - 1.0).abs() < 1e-12,
            "first write of a unit value has surprise 1"
        );
        assert!((mem.recall(&k, &v) - 1.0).abs() < 1e-12);
        for _ in 0..1_000 {
            mem.relax();
        }
        assert!(
            (mem.recall(&k, &v) - 1.0).abs() < 1e-12,
            "plain state must not decay"
        );
        assert!(
            mem.write(&k, &v, 1.0) < 1e-12,
            "a stored pattern is no longer surprising"
        );
    }

    #[test]
    fn exponential_memory_decays_at_its_closed_form_rate() {
        let d = 8;
        let decay = 0.99;
        let mut mem = AssocMemory::single(d, decay);
        let mut rng = Rng::new(42);
        let (mut k, mut v) = (vec![0.0; d], vec![0.0; d]);
        rng.fill_unit_vector(&mut k);
        rng.fill_unit_vector(&mut v);
        mem.write(&k, &v, 1.0);
        for _ in 0..200 {
            mem.relax();
        }
        assert!((mem.recall(&k, &v) - decay.powi(200)).abs() < 1e-12);
    }

    #[test]
    fn an_initialised_ladder_does_not_decay() {
        // Seeding rung 1 alone would bleed away to 1/sum(C_k) of its value.
        // Seeding every rung puts the chain at equilibrium, so an initialised
        // weight matrix survives.
        let d = 6;
        let mut mem = AssocMemory::ladder(d, geo(4.0), 5);
        let values: Vec<f64> = (0..d * d).map(|i| (i as f64 * 0.37).sin()).collect();
        mem.initialise(&values);
        for _ in 0..100_000 {
            mem.relax();
        }
        for (got, want) in mem.read().as_slice().iter().zip(&values) {
            assert!((got - want).abs() < 1e-12, "{got} drifted from {want}");
        }
    }

    #[test]
    fn a_rectangular_ladder_takes_rank_one_gradients() {
        let mut mem = AssocMemory::ladder_rect(5, 3, geo(2.0), 4);
        mem.inject(&[1.0, 0.0, 0.0, 0.0, -2.0], &[0.5, 1.0, 0.0], 1.0);
        assert_eq!(mem.read().shape(), (5, 3));
        assert_eq!(mem.read().get(0, 1), 1.0);
        assert_eq!(mem.read().get(4, 0), -1.0);
        assert_eq!(mem.read().get(2, 2), 0.0);
    }

    #[test]
    fn ladder_read_is_rung_one_only() {
        let d = 6;
        let mut mem = AssocMemory::ladder(d, geo(4.0), 5);
        let mut rng = Rng::new(43);
        let (mut k, mut v) = (vec![0.0; d], vec![0.0; d]);
        rng.fill_unit_vector(&mut k);
        rng.fill_unit_vector(&mut v);
        mem.write(&k, &v, 1.0);
        let rung0 = mem.ladder_state().unwrap().rung(0).clone();
        assert_eq!(mem.read(), &rung0);
        // The write landed only in rung 1; deeper rungs are still empty until
        // diffusion moves anything.
        assert_eq!(mem.ladder_state().unwrap().rung(1).frob_sq(), 0.0);
    }
}
