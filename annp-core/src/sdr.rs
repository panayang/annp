//! Continuous Sparse Distributed Representation (SDR) with dual input-weight
//! multi-timescale consolidation ladders, activity-gated event diffusion,
//! and zero-hyperparameter cosine-weighted projection.
//!
//! See `DESIGN-SDR.md` for full mathematical derivation and design rationales.
//!
//! Key components:
//! 1. `InputContextLadder`: Multi-timescale input context ladder where token embeddings
//!    diffuse across m_in rungs (micro-token, phrase, sentence, paragraph/domain),
//!    with independent unit-norm normalization across rungs before projection.
//! 2. `RandomProjection`: Fixed standard Gaussian random projection matrix Phi \in R^{D x d_in}
//!    with row unit normalization, paired with k-WTA (k-Winners-Take-All) sparse coordinate
//!    selection and direct non-negative cosine projection weighting.
//! 3. `SdrMemory`: Weight matrix W \in R^{V x D}, where each column hosts an independent
//!    consolidation ladder (m = 1, 2, 4, 8) or proximal EWC anchor.
//!    - Forward pass: Reads surface plastic conductance W = U_1.
//!    - Diffusion: Activity-Gated Event Diffusion (advances diffusion ONLY on active columns,
//!      quiescent/time-frozen on silent columns).

use crate::ladder::{Ladder, Schedule};
use crate::linalg::Mat;
use crate::rng::Rng;

/// Multi-timescale input context ladder that accumulates token embeddings across multiple timescales.
///
/// Replaces heuristic single-decay scalars (gamma) with an explicit Benna-Fusi diffusion ladder.
///
/// At step `t`, when token `x_{t-1}` is observed:
/// 1. Token unit-norm embedding `e(x_{t-1}) \in R^d` is injected into rung 0: `Z_0 += e(x_{t-1})`.
/// 2. Diffusion advances across all `m_in` rungs via `ladder.relax()`.
/// 3. Prior to projection, each rung `k` is independently unit-norm normalized:
///    `\hat{Z}_k = Z_k / ||Z_k||_2`.
/// 4. All normalized rungs are concatenated:
///    `\hat{Z}_t = [\hat{Z}_0; \hat{Z}_1; ...; \hat{Z}_{m_in - 1}] \in R^{m_in * d}`.
#[derive(Clone, Debug)]
pub struct InputContextLadder {
    d: usize,
    vocab: usize,
    m_in: usize,
    embeddings: Vec<f64>, // vocab x d (row v is unit-norm e(v))
    ladder: Ladder,       // m_in rungs of (1 x d)
    concat_buf: Vec<f64>, // m_in * d preallocated buffer
}

impl InputContextLadder {
    /// Creates a fresh input context ladder with randomly initialized, unit-norm token embeddings.
    pub fn new(d: usize, vocab: usize, m_in: usize, ladder_r: f64, rng: &mut Rng) -> Self {
        assert!(d > 0 && vocab > 0, "dimensions must be positive");
        assert!(m_in >= 2, "input ladder requires at least 2 rungs");
        assert!(ladder_r > 1.0, "geometric ratio must exceed 1.0");

        let mut embeddings = vec![0.0; vocab * d];
        for v in 0..vocab {
            let row = &mut embeddings[v * d..(v + 1) * d];
            rng.fill_unit_vector(row);
        }

        let schedule = Schedule::Geometric {
            r: ladder_r,
            g1: 0.25,
        };
        let ladder = Ladder::new(schedule, m_in, 1, d);
        let concat_buf = vec![0.0; m_in * d];

        Self {
            d,
            vocab,
            m_in,
            embeddings,
            ladder,
            concat_buf,
        }
    }

    /// Creates an input context ladder from pre-existing embedding weights.
    pub fn from_embeddings(
        d: usize,
        vocab: usize,
        m_in: usize,
        ladder_r: f64,
        embeddings: Vec<f64>,
    ) -> Self {
        assert_eq!(embeddings.len(), vocab * d, "embedding buffer size mismatch");
        assert!(m_in >= 2, "input ladder requires at least 2 rungs");

        let schedule = Schedule::Geometric {
            r: ladder_r,
            g1: 0.25,
        };
        let ladder = Ladder::new(schedule, m_in, 1, d);
        let concat_buf = vec![0.0; m_in * d];

        Self {
            d,
            vocab,
            m_in,
            embeddings,
            ladder,
            concat_buf,
        }
    }

    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    #[inline]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    #[inline]
    pub fn m_in(&self) -> usize {
        self.m_in
    }

    #[inline]
    pub fn total_dim(&self) -> usize {
        self.m_in * self.d
    }

    #[inline]
    pub fn embedding(&self, token: usize) -> &[f64] {
        assert!(token < self.vocab, "token index out of bounds");
        &self.embeddings[token * self.d..(token + 1) * self.d]
    }

    /// Resets the input context ladder back to zero (e.g. at sequence or domain boundary).
    pub fn reset(&mut self) {
        let zeros = vec![0.0; self.d];
        self.ladder.initialise(&zeros);
        self.concat_buf.fill(0.0);
    }

    /// Updates the input context ladder with the newly observed token `prev_token`.
    ///
    /// Strictly causal: incorporates token at step `t-1` before predicting target `y_t`.
    pub fn step(&mut self, prev_token: usize) {
        assert!(prev_token < self.vocab, "token index out of bounds");
        let emb = &self.embeddings[prev_token * self.d..(prev_token + 1) * self.d];

        // 1. Inject embedding flux into rung 0
        let r0 = self.ladder.rungs_mut()[0].as_mut_slice();
        for (z, &e) in r0.iter_mut().zip(emb) {
            *z += e;
        }

        // 2. Advance multi-timescale diffusion
        self.ladder.relax();

        // 3. Normalize each rung independently and concatenate into buffer
        let d = self.d;
        for (k, rung) in self.ladder.rungs_mut().iter().enumerate() {
            let slice = rung.as_slice();
            let mut sq = 0.0;
            for &val in slice {
                sq += val * val;
            }
            let norm = sq.sqrt();
            let out_chunk = &mut self.concat_buf[k * d..(k + 1) * d];
            if norm > 1e-12 {
                let inv_norm = 1.0 / norm;
                for (out, &val) in out_chunk.iter_mut().zip(slice) {
                    *out = val * inv_norm;
                }
            } else {
                out_chunk.fill(0.0);
            }
        }
    }

    /// Returns the concatenated, normalized multi-timescale state vector \hat{Z}_t \in R^{m_in * d}.
    #[inline]
    pub fn normalized_trace(&self) -> &[f64] {
        &self.concat_buf
    }
}

/// Fixed high-dimensional random projection with k-WTA sparse selection
/// and direct non-negative cosine projection weighting.
///
/// Matrix Phi \in R^{D x d_in}, where each row phi_i is a unit-norm standard Gaussian random vector.
/// Lifetime frozen: no gradients, no updates.
#[derive(Clone, Debug)]
pub struct RandomProjection {
    d_in: usize,
    d_sdr: usize,
    k: usize,
    phi: Mat, // d_sdr rows, d_in cols
}

impl RandomProjection {
    /// Initializes a random projection matrix with unit-norm Gaussian rows.
    pub fn new(d_in: usize, d_sdr: usize, k: usize, rng: &mut Rng) -> Self {
        assert!(d_in > 0 && d_sdr > 0, "dimensions must be positive");
        assert!(k > 0 && k <= d_sdr, "k must satisfy 1 <= k <= D");
        let mut phi = Mat::zeros(d_sdr, d_in);
        let slice = phi.as_mut_slice();
        for r in 0..d_sdr {
            let row = &mut slice[r * d_in..(r + 1) * d_in];
            rng.fill_unit_vector(row);
        }
        Self {
            d_in,
            d_sdr,
            k,
            phi,
        }
    }

    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    #[inline]
    pub fn d_sdr(&self) -> usize {
        self.d_sdr
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn phi(&self) -> &Mat {
        &self.phi
    }

    /// Projects normalized input z \in R^d_in to high-dimensional potential u \in R^D: u = Phi * z.
    pub fn project(&self, z: &[f64], out_u: &mut [f64]) {
        assert_eq!(z.len(), self.d_in, "input dimension mismatch");
        assert_eq!(out_u.len(), self.d_sdr, "output buffer size mismatch");
        self.phi.mul_vec(z, out_u);
    }

    /// Selects the top-k highest activation indices (k-WTA) and computes their
    /// non-negative cosine projection weights:
    /// `alpha_i = max(0, u_i) / \sum max(0, u_j)`.
    pub fn select_top_k_weighted(
        &self,
        u: &[f64],
        pairs_buf: &mut Vec<(usize, f64)>,
        out_active: &mut Vec<usize>,
        out_alphas: &mut Vec<f64>,
    ) {
        assert_eq!(u.len(), self.d_sdr, "projected dimension mismatch");
        out_active.clear();
        out_alphas.clear();
        pairs_buf.clear();

        pairs_buf.extend(u.iter().copied().enumerate());
        let k = self.k;
        pairs_buf.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top = &mut pairs_buf[..k];
        top.sort_unstable_by_key(|(idx, _)| *idx); // Sort for memory cache locality

        let mut sum_pos = 0.0;
        for &(idx, val) in top.iter() {
            out_active.push(idx);
            let pos = val.max(0.0);
            out_alphas.push(pos);
            sum_pos += pos;
        }

        if sum_pos > 1e-12 {
            let inv_sum = 1.0 / sum_pos;
            for alpha in out_alphas.iter_mut() {
                *alpha *= inv_sum;
            }
        } else {
            let uniform = 1.0 / k as f64;
            for alpha in out_alphas.iter_mut() {
                *alpha = uniform;
            }
        }
    }

    /// Projects input z to u, extracts top-k active neuron indices, and computes non-negative cosine weights.
    pub fn project_and_select(
        &self,
        z: &[f64],
        u_buf: &mut [f64],
        pairs_buf: &mut Vec<(usize, f64)>,
        out_active: &mut Vec<usize>,
        out_alphas: &mut Vec<f64>,
    ) {
        self.project(z, u_buf);
        self.select_top_k_weighted(u_buf, pairs_buf, out_active, out_alphas);
    }
}

/// Hierarchical Multi-Band Random Projection (8-Band Isomorphic Cortical SDR).
///
/// Divides input multi-timescale trace `\hat{Z}_t \in \mathbb{R}^{m_{\text{in}} \cdot d}` into `m_{\text{in}}` independent slices `Z_b \in \mathbb{R}^d`.
/// Each band `b \in [0, \text{num\_bands})` projects `Z_b` via dedicated `\Phi_b \in \mathbb{R}^{D_{\text{band}} \times d}` to `u_b \in \mathbb{R}^{D_{\text{band}}}`,
/// and selects local `k_{\text{band}}` active neurons independently, completely eliminating cross-timescale crowding.
#[derive(Clone, Debug)]
pub struct HierarchicalBandsProjection {
    num_bands: usize,
    d_rung: usize,
    d_band: usize,
    k_band: usize,
    d_sdr: usize,
    k_active: usize,
    phis: Vec<Mat>, // num_bands matrices of size d_band x d_rung
}

impl HierarchicalBandsProjection {
    /// Initializes an isomorphic multi-band cortical projection module.
    pub fn new(num_bands: usize, d_rung: usize, d_band: usize, k_band: usize, rng: &mut Rng) -> Self {
        assert!(num_bands > 0 && d_rung > 0 && d_band > 0, "dimensions must be positive");
        assert!(k_band > 0 && k_band <= d_band, "k_band must satisfy 1 <= k_band <= d_band");

        let mut phis = Vec::with_capacity(num_bands);
        for _ in 0..num_bands {
            let mut phi = Mat::zeros(d_band, d_rung);
            let slice = phi.as_mut_slice();
            for r in 0..d_band {
                let row = &mut slice[r * d_rung..(r + 1) * d_rung];
                rng.fill_unit_vector(row);
            }
            phis.push(phi);
        }

        Self {
            num_bands,
            d_rung,
            d_band,
            k_band,
            d_sdr: num_bands * d_band,
            k_active: num_bands * k_band,
            phis,
        }
    }

    #[inline]
    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    #[inline]
    pub fn d_sdr(&self) -> usize {
        self.d_sdr
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k_active
    }

    #[inline]
    pub fn d_in(&self) -> usize {
        self.num_bands * self.d_rung
    }

    /// Projects input `z \in \mathbb{R}^{m_{\text{in}} \cdot d}`, extracts local `k_{\text{band}}` active neurons per band,
    /// and populates global active indices and normalized cosine weights across all bands.
    pub fn project_and_select(
        &self,
        z: &[f64],
        u_buf: &mut [f64],
        pairs_buf: &mut Vec<(usize, f64)>,
        out_active: &mut Vec<usize>,
        out_alphas: &mut Vec<f64>,
    ) {
        assert_eq!(z.len(), self.d_in(), "input dimension mismatch");
        assert_eq!(u_buf.len(), self.d_sdr, "output buffer size mismatch");
        out_active.clear();
        out_alphas.clear();

        // 1. Identify which bands have active non-zero input energy (Energy-Gated Quiescence)
        let mut active_bands = Vec::with_capacity(self.num_bands);
        for b in 0..self.num_bands {
            let z_chunk = &z[b * self.d_rung..(b + 1) * self.d_rung];
            let mut energy = 0.0;
            for &val in z_chunk {
                energy += val * val;
            }
            if energy > 1e-12 {
                active_bands.push(b);
            }
        }

        if active_bands.is_empty() {
            return;
        }

        let inv_bands = 1.0 / active_bands.len() as f64;

        for &b in &active_bands {
            let z_chunk = &z[b * self.d_rung..(b + 1) * self.d_rung];
            let u_chunk = &mut u_buf[b * self.d_band..(b + 1) * self.d_band];
            self.phis[b].mul_vec(z_chunk, u_chunk);

            pairs_buf.clear();
            pairs_buf.extend(u_chunk.iter().copied().enumerate());
            let k_b = self.k_band;
            pairs_buf.select_nth_unstable_by(k_b - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top = &mut pairs_buf[..k_b];
            top.sort_unstable_by_key(|(idx, _)| *idx);

            let band_offset = b * self.d_band;
            let mut sum_pos = 0.0;
            let start_alpha_idx = out_alphas.len();

            for &(local_idx, val) in top.iter() {
                out_active.push(band_offset + local_idx);
                let pos = val.max(0.0);
                out_alphas.push(pos);
                sum_pos += pos;
            }

            if sum_pos > 1e-12 {
                let inv_sum = inv_bands / sum_pos;
                for alpha in &mut out_alphas[start_alpha_idx..] {
                    *alpha *= inv_sum;
                }
            } else {
                let uniform = inv_bands / k_b as f64;
                for alpha in &mut out_alphas[start_alpha_idx..] {
                    *alpha = uniform;
                }
            }
        }
    }
}

/// Underlying state backend for `SdrMemory`.
#[derive(Clone, Debug)]
pub enum MemoryBackend {
    /// Plain single matrix SGD (m = 1, no ladder, pure plastic weights).
    Plain {
        w: Mat, // vocab rows x d_sdr cols
    },
    /// Benna-Fusi multi-timescale consolidation ladder (m >= 2) with Activity-Gated Event Diffusion.
    Ladder {
        ladder: Ladder, // vocab rows x d_sdr cols
    },
    /// Online Proximal EWC with normalized Fisher diagonal estimate.
    ProximalEwc {
        w: Mat,      // vocab rows x d_sdr cols
        anchor: Mat, // w* anchor
        fisher: Mat, // running Fisher diagonal F
        lambda: f64, // penalty strength
        trail: f64,  // anchor & Fisher EMA follow rate alpha
    },
}

/// High-dimensional sparse associative memory with column-wise consolidation ladders
/// or online proximal EWC anchoring.
///
/// Weight matrix W \in R^{V x D}.
/// - Forward pass: Logits are computed by non-negative cosine weighted summation of surface
///   conductances U_1 for active columns:
///   `logits_t = \sum_{j=0}^{k-1} alpha_j * U_1(v, active[j]) \in R^V` (O(k * V) operations).
/// - Delta write: Active column `j` receives `eta * (k * alpha_j) * delta_t` into shallow rung 0,
///   and advances diffusion on active columns only via `ladder.relax_columns(active)`.
///   Silent columns undergo zero diffusion (frozen in time, zero quiescent dissipation).
#[derive(Clone, Debug)]
pub struct SdrMemory {
    vocab: usize,
    d_sdr: usize,
    backend: MemoryBackend,
    logits_buf: Vec<f64>,
    probs_buf: Vec<f64>,
    delta_buf: Vec<f64>,
}

impl SdrMemory {
    /// Creates an SDR memory backed by a Benna-Fusi consolidation ladder.
    ///
    /// For `m = 1`, automatically constructs a plain single-matrix memory.
    /// For `m >= 2`, initializes an m-rung consolidation ladder with activity-gated diffusion.
    pub fn new_ladder(vocab: usize, d_sdr: usize, m: usize, schedule: Schedule) -> Self {
        assert!(vocab > 0 && d_sdr > 0, "dimensions must be positive");
        let backend = if m == 1 {
            MemoryBackend::Plain {
                w: Mat::zeros(vocab, d_sdr),
            }
        } else {
            let ladder = Ladder::new(schedule, m, vocab, d_sdr);
            MemoryBackend::Ladder { ladder }
        };

        Self {
            vocab,
            d_sdr,
            backend,
            logits_buf: vec![0.0; vocab],
            probs_buf: vec![0.0; vocab],
            delta_buf: vec![0.0; vocab],
        }
    }

    /// Creates an SDR memory backed by a single plastic matrix (m = 1).
    pub fn new_plain(vocab: usize, d_sdr: usize) -> Self {
        Self::new_ladder(vocab, d_sdr, 1, Schedule::Uniform { g: 0.1 })
    }

    /// Creates an SDR memory with Online Proximal EWC anchoring.
    ///
    /// The proximal update rule is:
    /// `w <- (w + eta * alpha * g + stiff * w*) / (1 + stiff)`
    /// where `stiff = eta * alpha * lambda * (F / \bar{F})`.
    pub fn new_ewc(vocab: usize, d_sdr: usize, lambda: f64, trail: f64) -> Self {
        assert!(vocab > 0 && d_sdr > 0, "dimensions must be positive");
        assert!(lambda >= 0.0, "EWC lambda must be non-negative");
        assert!(trail > 0.0 && trail <= 1.0, "EWC trail must be in (0, 1]");
        Self {
            vocab,
            d_sdr,
            backend: MemoryBackend::ProximalEwc {
                w: Mat::zeros(vocab, d_sdr),
                anchor: Mat::zeros(vocab, d_sdr),
                fisher: Mat::zeros(vocab, d_sdr),
                lambda,
                trail,
            },
            logits_buf: vec![0.0; vocab],
            probs_buf: vec![0.0; vocab],
            delta_buf: vec![0.0; vocab],
        }
    }

    #[inline]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    #[inline]
    pub fn d_sdr(&self) -> usize {
        self.d_sdr
    }

    #[inline]
    pub fn backend(&self) -> &MemoryBackend {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut MemoryBackend {
        &mut self.backend
    }

    /// Returns the number of ladder rungs (1 for Plain and ProximalEwc).
    pub fn num_rungs(&self) -> usize {
        match &self.backend {
            MemoryBackend::Plain { .. } | MemoryBackend::ProximalEwc { .. } => 1,
            MemoryBackend::Ladder { ladder } => ladder.num_rungs(),
        }
    }

    /// Reads the fast surface weights (rung 0).
    pub fn read_fast_weights(&self) -> &Mat {
        match &self.backend {
            MemoryBackend::Plain { w } | MemoryBackend::ProximalEwc { w, .. } => w,
            MemoryBackend::Ladder { ladder } => ladder.rung(0),
        }
    }

    /// Reads rung `k` of the consolidation ladder. For single-matrix backends, returns rung 0.
    pub fn read_rung(&self, k: usize) -> &Mat {
        match &self.backend {
            MemoryBackend::Plain { w } | MemoryBackend::ProximalEwc { w, .. } => {
                assert_eq!(k, 0, "single-matrix backend only has rung 0");
                w
            }
            MemoryBackend::Ladder { ladder } => ladder.rung(k),
        }
    }

    /// Forward pass: computes logits via non-negative cosine weighted summation of surface conductances U_1:
    /// `logits[v] = \sum_{j=0}^{k-1} alphas[j] * U_1(v, active[j])`.
    pub fn forward(&self, active: &[usize], alphas: &[f64], out_logits: &mut [f64]) {
        assert_eq!(out_logits.len(), self.vocab, "output logits buffer size mismatch");
        assert_eq!(active.len(), alphas.len(), "active and alphas length mismatch");
        out_logits.fill(0.0);

        let d = self.d_sdr;
        let w_data = self.read_fast_weights().as_slice();

        for (v, logit) in out_logits.iter_mut().enumerate() {
            let row = &w_data[v * d..(v + 1) * d];
            let mut sum = 0.0;
            for (&col, &alpha) in active.iter().zip(alphas) {
                sum += alpha * row[col];
            }
            *logit = sum;
        }
    }

    /// Computes softmax probabilities, cross-entropy loss, and top-1 accuracy for `target`.
    ///
    /// Safe against overflow/underflow.
    pub fn score_logits(logits: &[f64], target: usize, out_probs: &mut [f64]) -> (f64, bool) {
        let v_len = logits.len();
        assert_eq!(out_probs.len(), v_len, "probs buffer size mismatch");
        assert!(target < v_len, "target index out of bounds");

        let peak = logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut sum_exp = 0.0;
        let mut best_v = 0;
        let mut best_val = f64::NEG_INFINITY;

        for (v, (&logit, prob)) in logits.iter().zip(out_probs.iter_mut()).enumerate() {
            let e = (logit - peak).exp();
            *prob = e;
            sum_exp += e;
            if logit > best_val {
                best_val = logit;
                best_v = v;
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for prob in out_probs.iter_mut() {
            *prob *= inv_sum;
        }

        let target_prob = out_probs[target].max(f64::MIN_POSITIVE);
        let loss = -target_prob.ln();
        let is_correct = best_v == target;

        (loss, is_correct)
    }

    /// Predicts target given active neuron indices and cosine weights without modifying state (eta = 0).
    pub fn predict(&mut self, active: &[usize], alphas: &[f64], target: usize) -> (f64, bool) {
        let mut logits = std::mem::take(&mut self.logits_buf);
        let mut probs = std::mem::take(&mut self.probs_buf);

        self.forward(active, alphas, &mut logits);
        let (loss, is_correct) = Self::score_logits(&logits, target, &mut probs);

        self.logits_buf = logits;
        self.probs_buf = probs;
        (loss, is_correct)
    }

    /// Performs sparse Delta write and activity-gated event diffusion.
    ///
    /// - Active columns `j`: `U_1(v, active[j]) += (eta * alphas[j]) * (1_{v = target} - probs[v])`,
    ///   followed by localized diffusion on active columns ONLY (`ladder.relax_columns(active)`).
    /// - Silent columns `i \notin active`: zero external injection, ZERO diffusion (time frozen).
    pub fn update_sparse(
        &mut self,
        active: &[usize],
        alphas: &[f64],
        target: usize,
        eta: f64,
        probs: &[f64],
    ) {
        assert_eq!(probs.len(), self.vocab, "probs size mismatch");
        assert_eq!(active.len(), alphas.len(), "active and alphas length mismatch");
        assert!(target < self.vocab, "target index out of bounds");

        if eta <= 0.0 {
            return;
        }

        let v_len = self.vocab;
        let d = self.d_sdr;

        // Compute error residual delta \in R^V
        self.delta_buf.resize(v_len, 0.0);
        for (v, (&p, delta)) in probs.iter().zip(self.delta_buf.iter_mut()).enumerate() {
            *delta = if v == target { 1.0 - p } else { -p };
        }

        let k_factor = active.len() as f64;

        match &mut self.backend {
            MemoryBackend::Plain { w } => {
                let slice = w.as_mut_slice();
                for (v, &delta_v) in self.delta_buf.iter().enumerate() {
                    let base = v * d;
                    for (&col, &alpha) in active.iter().zip(alphas) {
                        let eff_step = eta * (k_factor * alpha) * delta_v;
                        slice[base + col] += eff_step;
                    }
                }
            }
            MemoryBackend::Ladder { ladder, .. } => {
                // 1. Inject (eta * (k * alpha)) * delta into rung 0 for active columns only
                let r0 = ladder.rungs_mut()[0].as_mut_slice();
                for (v, &delta_v) in self.delta_buf.iter().enumerate() {
                    let base = v * d;
                    for (&col, &alpha) in active.iter().zip(alphas) {
                        let eff_step = eta * (k_factor * alpha) * delta_v;
                        r0[base + col] += eff_step;
                    }
                }
                // 2. Activity-Gated Event Diffusion: advance diffusion on ACTIVE columns only
                ladder.relax_columns(active);
            }
            MemoryBackend::ProximalEwc {
                w,
                anchor,
                fisher,
                lambda,
                trail,
            } => {
                let f_slice = fisher.as_slice();
                let f_mean = f_slice.iter().sum::<f64>() / f_slice.len() as f64;
                let f_bar = f_mean.max(1e-12);

                let lam = *lambda;
                let tr = *trail;

                for (v, &delta_v) in self.delta_buf.iter().enumerate() {
                    let g = delta_v;
                    let base = v * d;
                    for (&col, &alpha) in active.iter().zip(alphas) {
                        let eff_eta = eta * (k_factor * alpha);
                        let idx = base + col;
                        let f = fisher.as_slice()[idx];
                        let stiff = eff_eta * lam * (f / f_bar);

                        let w_cur = w.as_slice()[idx];
                        let w_star = anchor.as_slice()[idx];

                        // Proximal EWC update formula
                        let w_new = (w_cur + eff_eta * g + stiff * w_star) / (1.0 + stiff);

                        w.as_mut_slice()[idx] = w_new;
                        fisher.as_mut_slice()[idx] += tr * (g * g - f);
                        anchor.as_mut_slice()[idx] += tr * (w_new - w_star);
                    }
                }
            }
        }
    }

    /// Evaluates predictions, scores loss and accuracy, and applies sparse Delta updates.
    pub fn observe(
        &mut self,
        active: &[usize],
        alphas: &[f64],
        target: usize,
        eta: f64,
    ) -> (f64, bool) {
        let mut logits = std::mem::take(&mut self.logits_buf);
        let mut probs = std::mem::take(&mut self.probs_buf);

        self.forward(active, alphas, &mut logits);
        let (loss, is_correct) = Self::score_logits(&logits, target, &mut probs);

        self.update_sparse(active, alphas, target, eta, &probs);

        self.logits_buf = logits;
        self.probs_buf = probs;
        (loss, is_correct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_context_ladder_diffuses_and_normalizes() {
        let mut rng = Rng::new(20260817);
        let mut ladder = InputContextLadder::new(16, 10, 3, 2.0, &mut rng);

        assert_eq!(ladder.total_dim(), 48); // 3 rungs * 16 dim

        ladder.step(3);
        ladder.step(7);

        let z = ladder.normalized_trace();
        assert_eq!(z.len(), 48);

        // Check each chunk has unit norm
        for k in 0..3 {
            let chunk = &z[k * 16..(k + 1) * 16];
            let sq: f64 = chunk.iter().map(|x| x * x).sum();
            assert!((sq.sqrt() - 1.0).abs() < 1e-10, "rung {k} chunk must have unit norm");
        }
    }

    #[test]
    fn random_projection_produces_exact_k_weighted_neurons() {
        let mut rng = Rng::new(20260817);
        let proj = RandomProjection::new(48, 128, 8, &mut rng);

        let mut z = vec![0.0; 48];
        rng.fill_unit_vector(&mut z);

        let mut u = vec![0.0; 128];
        let mut pairs = Vec::new();
        let mut active = Vec::new();
        let mut alphas = Vec::new();

        proj.project_and_select(&z, &mut u, &mut pairs, &mut active, &mut alphas);

        assert_eq!(active.len(), 8);
        assert_eq!(alphas.len(), 8);

        // Sum of alphas must equal 1.0
        let alpha_sum: f64 = alphas.iter().sum();
        assert!((alpha_sum - 1.0).abs() < 1e-10, "alphas must sum to 1.0");

        // Top-k property verified
        for &idx in &active {
            assert!(idx < 128);
        }
        let min_active = active.iter().map(|&i| u[i]).fold(f64::INFINITY, f64::min);
        let max_silent = (0..128)
            .filter(|i| !active.contains(i))
            .map(|i| u[i])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min_active >= max_silent, "top-k property violated");
    }

    #[test]
    fn silent_columns_remain_frozen_in_time_under_activity_gating() {
        let sched = Schedule::Geometric { r: 2.0, g1: 0.05 };
        let mut mem = SdrMemory::new_ladder(8, 16, 4, sched);

        // Train column 0 on target 3 repeatedly
        let active1 = vec![0];
        let alphas1 = vec![1.0];
        for _ in 0..50 {
            mem.observe(&active1, &alphas1, 3, 0.2);
        }

        let r0_init = mem.read_rung(0).get(3, 0);
        let r1_init = mem.read_rung(1).get(3, 0);
        assert!(r0_init > 0.0 && r1_init > 0.0, "rungs should have accumulated charge");

        // Now train disjoint columns [5, 6, 7] on target 1 for 1000 steps
        // Column 0 is SILENT throughout this period
        let active2 = vec![5, 6, 7];
        let alphas2 = vec![0.33, 0.33, 0.34];
        for _ in 0..1000 {
            mem.observe(&active2, &alphas2, 1, 0.2);
        }

        let r0_after = mem.read_rung(0).get(3, 0);
        let r1_after = mem.read_rung(1).get(3, 0);

        // Under activity-gating, silent column 0 was FROZEN in time and experienced ZERO decay
        assert_eq!(
            r0_after, r0_init,
            "silent column 0 must remain perfectly frozen without quiescent dissipation"
        );
        assert_eq!(
            r1_after, r1_init,
            "silent column 0 rung 1 must remain perfectly frozen"
        );

        // Column 2 was never active, must be strictly 0.0 everywhere
        for k in 0..4 {
            for v in 0..8 {
                assert_eq!(
                    mem.read_rung(k).get(v, 2),
                    0.0,
                    "never-active column 2 must remain strictly 0.0"
                );
            }
        }
    }

    #[test]
    fn active_columns_consolidate_and_replenish_surface_weights() {
        let sched = Schedule::Geometric { r: 2.0, g1: 0.1 };
        let mut mem = SdrMemory::new_ladder(8, 16, 4, sched);

        let active = vec![2];
        let alphas = vec![1.0];

        // 1. Train column 2 heavily on target 5 to build deep consolidated charge
        for _ in 0..100 {
            mem.observe(&active, &alphas, 5, 0.5);
        }

        let deep_charge = mem.read_rung(2).get(5, 2);
        assert!(deep_charge > 0.0, "deep rung must have absorbed consolidated charge");

        // 2. Now apply conflicting gradient on target 1 into column 2 for a few steps
        for _ in 0..5 {
            mem.observe(&active, &alphas, 1, 0.2);
        }

        // Deep rung 2 still maintains positive memory for target 5
        let deep_after = mem.read_rung(2).get(5, 2);
        assert!(deep_after > 0.0, "deep memory must persist through short conflicting bursts");
    }

    #[test]
    fn ewc_proximal_anchoring_is_stable_and_restrains_weights() {
        let mut mem_free = SdrMemory::new_ewc(4, 8, 0.0, 0.1);
        let mut mem_stiff = SdrMemory::new_ewc(4, 8, 10.0, 0.1);

        let active = vec![2, 4];
        let alphas = vec![0.5, 0.5];
        for _ in 0..20 {
            mem_free.observe(&active, &alphas, 1, 0.1);
            mem_stiff.observe(&active, &alphas, 1, 0.1);
        }

        // Now change target to 2
        for _ in 0..10 {
            mem_free.observe(&active, &alphas, 2, 0.1);
            mem_stiff.observe(&active, &alphas, 2, 0.1);
        }

        let w_free = mem_free.read_fast_weights().get(2, 2);
        let w_stiff = mem_stiff.read_fast_weights().get(2, 2);

        // Under high EWC lambda, weights move less towards new target
        assert!(w_stiff < w_free, "EWC must restrain weights from rapid shifting");
        assert!(!w_stiff.is_nan() && !w_stiff.is_infinite(), "EWC proximal update must be stable");
    }

    #[test]
    fn forward_pass_sparsity_consistency() {
        let mut mem = SdrMemory::new_plain(4, 16);
        // Write into column 3 only
        let active = vec![3];
        let alphas = vec![1.0];
        mem.observe(&active, &alphas, 2, 0.5);

        let mut logits1 = vec![0.0; 4];
        mem.forward(&[3], &[1.0], &mut logits1);

        let mut logits2 = vec![0.0; 4];
        mem.forward(&[0, 1], &[0.5, 0.5], &mut logits2);

        assert!(logits1[2] > 0.0, "active column must contribute to logit");
        assert_eq!(logits2[2], 0.0, "silent columns must produce 0.0 logit");
    }

    #[test]
    fn surface_readout_reads_u1_and_diffuses_from_deep_rungs() {
        let sched = Schedule::Geometric { r: 2.0, g1: 0.1 };
        let mut mem = SdrMemory::new_ladder(4, 8, 4, sched);

        // Directly write a signal into deep rung 1 (U_2) while surface U_1 is zero
        match mem.backend_mut() {
            MemoryBackend::Ladder { ladder } => {
                ladder.rungs_mut()[1].set(2, 3, 4.0);
            }
            _ => unreachable!(),
        }

        let mut logits0 = vec![0.0; 4];
        mem.forward(&[3], &[1.0], &mut logits0);
        assert_eq!(logits0[2], 0.0, "surface readout U1 is initially 0.0");

        // Advance diffusion on active column 3: deep rung U_2 diffuses into surface U_1
        match mem.backend_mut() {
            MemoryBackend::Ladder { ladder } => {
                ladder.relax_columns(&[3]);
            }
            _ => unreachable!(),
        }

        let mut logits1 = vec![0.0; 4];
        mem.forward(&[3], &[1.0], &mut logits1);
        assert!(
            logits1[2] > 0.0,
            "diffusive flux from deep rung U_2 must replenish surface U_1"
        );
    }

    #[test]
    fn hierarchical_bands_projection_produces_exact_k_per_band() {
        let mut rng = Rng::new(20260817);
        let num_bands = 8;
        let d_rung = 16;
        let d_band = 16;
        let k_band = 1;
        let proj = HierarchicalBandsProjection::new(num_bands, d_rung, d_band, k_band, &mut rng);

        assert_eq!(proj.d_sdr(), 128);
        assert_eq!(proj.k(), 8);

        let z = vec![0.5; num_bands * d_rung];
        let mut u_buf = vec![0.0; 128];
        let mut pairs_buf = Vec::new();
        let mut active = Vec::new();
        let mut alphas = Vec::new();

        proj.project_and_select(&z, &mut u_buf, &mut pairs_buf, &mut active, &mut alphas);

        assert_eq!(active.len(), 8);
        assert_eq!(alphas.len(), 8);

        // Verify each band contributed exactly 1 active neuron in its own interval [b*16, (b+1)*16)
        for (b, &act) in active.iter().enumerate() {
            assert!(
                act >= b * 16 && act < (b + 1) * 16,
                "Band {} active column {} out of expected interval [{}, {})",
                b, act, b * 16, (b + 1) * 16
            );
        }

        // Verify sum of alphas is strictly 1.0
        let sum_alpha: f64 = alphas.iter().sum();
        assert!((sum_alpha - 1.0).abs() < 1e-6, "alphas must sum to 1.0");
    }

    #[test]
    fn hierarchical_bands_energy_gating_quiescence() {
        let mut rng = Rng::new(20260817);
        let num_bands = 8;
        let d_rung = 16;
        let d_band = 16;
        let k_band = 1;
        let proj = HierarchicalBandsProjection::new(num_bands, d_rung, d_band, k_band, &mut rng);

        // Only Band 0 and Band 1 have energy, Bands 2..8 are strictly 0.0
        let mut z = vec![0.0; num_bands * d_rung];
        for i in 0..2 * d_rung {
            z[i] = 1.0;
        }

        let mut u_buf = vec![0.0; 128];
        let mut pairs_buf = Vec::new();
        let mut active = Vec::new();
        let mut alphas = Vec::new();

        proj.project_and_select(&z, &mut u_buf, &mut pairs_buf, &mut active, &mut alphas);

        // Exactly 2 active neurons from the 2 active bands, zero noise from inactive bands
        assert_eq!(active.len(), 2, "only 2 bands with energy should activate");
        assert_eq!(alphas.len(), 2);
        assert!(active[0] < 16, "first neuron must be in Band 0");
        assert!(active[1] >= 16 && active[1] < 32, "second neuron must be in Band 1");

        // Alphas must sum to 1.0 and each be 0.5
        let sum_alpha: f64 = alphas.iter().sum();
        assert!((sum_alpha - 1.0).abs() < 1e-6, "alphas must sum to 1.0");
        assert!((alphas[0] - 0.5).abs() < 1e-6);
        assert!((alphas[1] - 0.5).abs() < 1e-6);
    }
}
