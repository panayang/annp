//! Embedding, shattering, content-addressed ingress, reassembly, output head.
//!
//! This is the layer that turns a token into particles and particles back into
//! a prediction. Everything in it is either parameter-free by construction or
//! learned from the single backward pass at the output head.
//!
//! **Shattering is a fixed orthogonal rotation.** A random sign flip followed by
//! a Walsh-Hadamard transform: orthogonal, so Parseval holds and the shatter
//! and reassembly boundary is gradient-isometric; `O(n log n)`; and carrying no
//! parameters at all, so it never appears on the hyperparameter table. The sign
//! flip matters — plain Hadamard has an all-ones first row, which would hand
//! chunk zero the mean of the embedding and leave the rest with the rest.
//!
//! **Direction goes in the payload, magnitude goes in the mass.** A chunk
//! becomes a unit-norm payload plus a mass equal to its share of the total. Two
//! consequences: every `d_head`-sized product downstream runs in a fixed
//! numerical range no matter how many hops it has taken, and a chunk carrying
//! little of the token's energy starts below everything else and is absorbed
//! early, so adaptive compute costs nothing to implement.
//!
//! **Ingress is content-addressed, and that is the point.** Uniform ingress is
//! the maximum-entropy input distribution: no topographic map can form, related
//! memories end up on unrelated nodes, and particles have to travel to meet.
//!
//! The seed position is the phase of the embedding in two fixed random planes
//! — scale-free, so no normalising constant. But a *random* projection of
//! `d_model` onto two angles preserves almost no structure, so on its own that
//! story is aspirational: the map exists but means nothing.
//!
//! So the anchor is **remembered rather than learned**: a token next enters
//! where its own mass last came to rest. That is a fixed-point iteration on
//! the dynamics that already exist, not a second learning rule — no rate to
//! choose, and structurally nothing to resonate with the adaptive topology of
//! §1.9, because it is a readout rather than an independent dynamical system.
//! Its failure mode is collapse (everything migrating to one region), which
//! routing homeostasis does not guard against, so `Ingress::drift` reports it.
//!
//! **The output head may be tied to the embedding table or separate.** Tying
//! was chosen so the table gets a learning signal without a gradient crossing
//! the graph, but it imposes a structural limit that took a while to notice:
//! with `logit_v(u) = <E_v, E_u>` the score matrix is `E E^T`, which is
//! symmetric and positive semi-definite. A Markov transition matrix is neither.
//! Symmetry forces `P(v|u)/P(u|v)` to factorise as `f(v)/f(u)`, and positive
//! semi-definiteness makes the diagonal dominate, permanently biasing the model
//! toward predicting the token it was just shown. A tied bypass therefore
//! *cannot* represent a bigram model, however much data it sees.
//!
//! Untied, the input embedding stays at its random initialisation — a perfectly
//! good fixed code — and the head is an unconstrained linear map, so any logit
//! matrix is reachable. `ModelParams::tied` selects between them.

use crate::graph::Ring;
use crate::ladder::{AssocMemory, Schedule};
use crate::linalg::Mat;
use crate::linalg::norm;
use crate::rng::Rng;

/// A fixed orthogonal rotation: sign flip, then normalised Walsh-Hadamard.
///
/// `H` is symmetric and, once scaled by `1/sqrt(n)`, its own inverse, and the
/// sign flip is its own inverse too, so the reverse transform is the same two
/// steps in the other order. No parameters, nothing to train.
#[derive(Clone, Debug)]
pub struct Rotation {
    n: usize,
    signs: Vec<f64>,
    scale: f64,
}

impl Rotation {
    pub fn new(n: usize, rng: &mut Rng) -> Self {
        assert!(
            n.is_power_of_two(),
            "the Hadamard transform needs a power-of-two width, got {n}"
        );
        let signs = (0..n)
            .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        Self {
            n,
            signs,
            scale: 1.0 / (n as f64).sqrt(),
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.n
    }

    /// In-place fast Walsh-Hadamard transform, unnormalised.
    fn hadamard(v: &mut [f64]) {
        let n = v.len();
        let mut span = 1;
        while span < n {
            for base in (0..n).step_by(span * 2) {
                for i in base..base + span {
                    let (a, b) = (v[i], v[i + span]);
                    v[i] = a + b;
                    v[i + span] = a - b;
                }
            }
            span *= 2;
        }
    }

    /// `v <- H D v`.
    pub fn forward(&self, v: &mut [f64]) {
        assert_eq!(v.len(), self.n, "rotation width mismatch");
        for (x, s) in v.iter_mut().zip(&self.signs) {
            *x *= s;
        }
        Self::hadamard(v);
        for x in v.iter_mut() {
            *x *= self.scale;
        }
    }

    /// `v <- D H v`, the exact inverse of `forward`.
    pub fn inverse(&self, v: &mut [f64]) {
        assert_eq!(v.len(), self.n, "rotation width mismatch");
        Self::hadamard(v);
        for (x, s) in v.iter_mut().zip(&self.signs) {
            *x *= s * self.scale;
        }
    }
}

/// How a token's entry point on the grid is chosen.
///
/// Content addressing was the original design: similar tokens start near each
/// other so their memories are co-located. It is measured to place
/// *consecutive* tokens at unrelated points, which is why no context flows —
/// sequence neighbours are not content neighbours.
///
/// `Constant` is the control that showed context can flow at all, at the cost
/// of piling every token onto the same nodes. `Cursor` keeps that sharing while
/// spreading the load. Note what `Cursor` does *not* address: it gives
/// recency, not similarity, and a source whose dependencies are all recent
/// flatters it. Real language needs both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngressMode {
    /// Phase of the embedding in two fixed random planes.
    Content,
    /// One fixed anchor for every token. Maximal sharing, no addressing.
    Constant,
    /// A cursor advancing one node per token. Recency becomes adjacency.
    Cursor,
    /// Where this token's own mass last came to rest. Measured worse.
    Readout,
}

/// Maps an embedding to the ring position its particles enter at.
#[derive(Clone, Debug)]
pub struct Ingress {
    len: usize,
    /// One orthonormal pair defining a plane whose phase gives the ring
    /// coordinate. Fixed at construction, never trained.
    planes: [Vec<f64>; 2],
    /// Per-slot ring offset: a random permutation of `P` consecutive cells.
    ///
    /// The 2-D version drew inside a disc of radius `sqrt(P/pi)`, chosen so the
    /// disc holds `P` nodes and one slot lands per node on average. On a ring
    /// the same requirement has an exact answer instead of an approximate one —
    /// `P` consecutive cells hold exactly `P` nodes — so the radius constant and
    /// the rejection sampling both disappear. Only the assignment of slots to
    /// cells stays random.
    offsets: Vec<u32>,
    /// Where each token's mass last came to rest, once it has been observed.
    remembered: Vec<Option<usize>>,
    mode: IngressMode,
    /// How far anchors have travelled in total, and over how many moves. Rising
    /// distinct-anchor counts with bounded drift is a map forming; drift that
    /// never settles, or a collapsing anchor count, is the failure mode.
    moves: u64,
    travelled: u64,
}

impl Ingress {
    pub fn new(len: usize, d_model: usize, slots: usize, vocab: usize, rng: &mut Rng) -> Self {
        assert!(
            slots <= len,
            "cannot give {slots} slots a cell each on a ring of {len}"
        );
        let mut planes: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; d_model]);
        for i in 0..2 {
            rng.fill_unit_vector(&mut planes[i]);
            // Gram-Schmidt against the previous one, so the plane is genuinely
            // two-dimensional and the phase is well defined.
            for j in 0..i {
                let proj = crate::linalg::dot(&planes[i], &planes[j]);
                let (done, rest) = planes.split_at_mut(i);
                for (x, b) in rest[0].iter_mut().zip(&done[j]) {
                    *x -= proj * b;
                }
            }
            let len_i = norm(&planes[i]);
            assert!(
                len_i > 1e-9,
                "degenerate ingress plane; is d_model at least 2?"
            );
            for x in planes[i].iter_mut() {
                *x /= len_i;
            }
        }

        // Centre the interval on the anchor so a token's fragments straddle it
        // rather than trailing off to one side, which would bias every entry.
        let half = (slots / 2) as i64;
        let mut offsets: Vec<u32> = (0..slots as i64)
            .map(|k| (k - half).rem_euclid(len as i64) as u32)
            .collect();
        for k in (1..offsets.len()).rev() {
            offsets.swap(k, rng.next_below(k as u64 + 1) as usize);
        }
        Self {
            len,
            planes,
            offsets,
            remembered: vec![None; vocab],
            moves: 0,
            travelled: 0,
            mode: IngressMode::Content,
        }
    }

    pub fn set_mode(&mut self, mode: IngressMode) {
        self.mode = mode;
    }

    /// Records where a token's mass came to rest. Returns how far its anchor
    /// moved around the ring.
    pub fn observe(&mut self, token: u32, resting: usize, embedding: &[f64]) -> usize {
        if self.mode != IngressMode::Readout {
            return 0;
        }
        let previous = self.anchor(token, 0, embedding);
        let d = (resting + self.len - previous) % self.len;
        let moved = d.min(self.len - d);
        self.remembered[token as usize] = Some(resting);
        self.moves += 1;
        self.travelled += moved as u64;
        moved
    }

    /// `(tokens with a remembered anchor, distinct cells occupied, mean move)`.
    pub fn drift(&self) -> (usize, usize, f64) {
        let known: Vec<usize> = self.remembered.iter().flatten().copied().collect();
        let distinct: std::collections::HashSet<usize> = known.iter().copied().collect();
        let mean = if self.moves > 0 {
            self.travelled as f64 / self.moves as f64
        } else {
            0.0
        };
        (known.len(), distinct.len(), mean)
    }

    /// Where this token enters.
    pub fn anchor(&self, token: u32, position: u64, embedding: &[f64]) -> usize {
        match self.mode {
            IngressMode::Content => self.phase_anchor(embedding),
            IngressMode::Constant => self.len / 2,
            // One cell per token straight along the ring, so ring distance is
            // stream lag with no remainder. That identity is the reason the
            // lattice is one-dimensional at all: on a torus the same map splits
            // lag across two axes and aliases every scale but two onto them.
            // Nothing here is tuned; the step is one because lag is one.
            IngressMode::Cursor => position as usize % self.len,
            IngressMode::Readout => match self.remembered[token as usize] {
                Some(a) => a,
                None => self.phase_anchor(embedding),
            },
        }
    }

    /// Ring coordinate of an embedding, as the phase in the fixed plane.
    fn phase_anchor(&self, embedding: &[f64]) -> usize {
        let angle = crate::linalg::dot(embedding, &self.planes[0])
            .atan2(crate::linalg::dot(embedding, &self.planes[1]));
        // atan2 is in (-pi, pi]; shift to [0, 1) then onto the ring.
        let unit = (angle + std::f64::consts::PI) / std::f64::consts::TAU;
        ((unit * self.len as f64) as usize).min(self.len - 1)
    }

    /// Entry node for one slot of a token with this embedding.
    pub fn node_for(
        &self,
        ring: &Ring,
        token: u32,
        position: u64,
        embedding: &[f64],
        slot: usize,
    ) -> u32 {
        let anchor = self.anchor(token, position, embedding);
        ring.shift(anchor as u32, self.offsets[slot] as usize)
    }
}

/// How the assembled vector becomes a distribution over the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// A single matrix. Whatever the network retrieves about earlier tokens
    /// enters the logits as a separate additive term, which caps it at the
    /// log-linear ceiling.
    Linear,
    /// A soft lookup over `slots` stored (key, distribution) pairs, mixed in
    /// **probability** space.
    ///
    /// §23 measured that what the readout lacks is a lookup, not a
    /// nonlinearity: a one-hidden-layer probe recovered 3.6% of the gap where
    /// decoding the context and indexing a table recovered 62%. Mixing
    /// distributions rather than logits is what makes the combination
    /// non-additive in log space, and matching a query against stored keys is
    /// the same operation routing already performs against node expectations.
    Lookup { slots: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct ModelParams {
    pub vocab: usize,
    pub d_head: usize,
    /// Particles per token, `P`. `d_model = slots * d_head`, so this is the
    /// width knob — and because nodes never see a slot index, it can change
    /// after training without touching a single node weight.
    pub slots: usize,
    /// Ring length: how many nodes there are. Reach is bounded by this, since
    /// stream positions `t` and `t + nodes` enter at the same cell and nothing
    /// downstream can separate them.
    pub nodes: usize,
    pub schedule: Schedule,
    /// Rungs behind the tied table. One means a plain matrix, which is the
    /// reference point for what consolidation costs the output head.
    pub embed_rungs: usize,
    pub learning_rate: f64,
    /// Whether the output head is the embedding table itself. Ignored for
    /// `HeadKind::Lookup`, which has weights of its own.
    pub tied: bool,
    pub head_kind: HeadKind,
    /// Deposit the payload and the node's answer into separate halves of the
    /// reassembly buffer rather than summing them.
    ///
    /// The emitted payload is `q + tanh(Wq)` renormalised, and the two terms
    /// have magnitudes 1 and about 0.4, so averaging emitted payloads is mostly
    /// averaging the input. Splitting them keeps an untrained node transparent —
    /// its answer is zero, and the payload half is unaffected — while letting
    /// the readout learn what the answers are worth.
    pub split_deposit: bool,
    /// Components the lookup readout evaluates per token. Zero keeps the ones
    /// above a uniform share, which needs no constant; a positive value keeps
    /// exactly that many. The full mixture is `O(M V)` a token and dominates the
    /// network's own cost once the vocabulary is real.
    pub head_top_k: usize,
    /// Subtract the running mean of the assembled vector for the current token
    /// before the readout.
    ///
    /// The network does bind context — retrieval among needles sharing a key
    /// symbol runs far above chance while a bypass control sits exactly at it
    /// (DESIGN.md §31.4) — but the binding rides at a cosine margin of a few
    /// thousandths on top of a component fixed by the current token, and the
    /// readout cannot pull it out from under that. Removing the token's own
    /// mean leaves the part that depends on everything else, which is the part
    /// worth reading. The mean is a plain prequential average, so this adds no
    /// constant to tune.
    pub centre_readout: bool,
}

impl ModelParams {
    #[inline]
    pub fn d_model(&self) -> usize {
        self.slots * self.d_head
    }

    /// Width the readout sees. Twice `d_model` when the payload and the nodes'
    /// answers are deposited apart, so the head can weigh them itself instead of
    /// being handed a fixed 1-to-0.4 sum of the two.
    #[inline]
    pub fn readout_width(&self) -> usize {
        self.d_model() * if self.split_deposit { 2 } else { 1 }
    }
}

/// One token's particles, ready for `Engine::inject`.
pub struct Shattered {
    pub seeds: Vec<(u32, u16, f64, Vec<f64>)>,
    /// Total chunk magnitude. Reassembly needs it to undo the direction and
    /// mass split, and only the caller that shattered the token knows it.
    pub scale: f64,
}

pub struct Model {
    params: ModelParams,
    /// Running mean of the assembled vector per current token, and how many
    /// observations are behind each row. `None` when centring is off.
    centre: Option<Mat>,
    centre_count: Vec<f64>,
    /// `vocab x d_model`, read for input. Also the output head when tied, in
    /// which case it is the only thing differentiated.
    embedding: AssocMemory,
    /// Separate output head when untied; `None` when tied.
    head: Option<AssocMemory>,
    lookup: Option<LookupHead>,
    rotation: Rotation,
    ingress: Ingress,
    embed_buf: Vec<f64>,
    logit_buf: Vec<f64>,
}

impl Model {
    pub fn new(params: ModelParams, rng: &mut Rng) -> Self {
        let d_model = params.d_model();
        assert!(params.vocab >= 2, "need at least two tokens to have a loss");
        assert!(d_model >= 4, "ingress needs four independent planes");

        let mut embedding = if params.embed_rungs <= 1 {
            AssocMemory::single_rect(params.vocab, d_model, 1.0)
        } else {
            AssocMemory::ladder_rect(params.vocab, d_model, params.schedule, params.embed_rungs)
        };
        // Unit-norm rows in expectation, so a fresh embedding is already in the
        // range the unit-norm payload invariant expects.
        let sigma = 1.0 / (d_model as f64).sqrt();
        let init: Vec<f64> = (0..params.vocab * d_model)
            .map(|_| rng.next_normal() * sigma)
            .collect();
        embedding.initialise(&init);

        // The readout is as wide as what it is handed, which is twice `d_model`
        // when the payload and the answers arrive separately. The embedding
        // stays `d_model` because it is what gets shattered into slots, so a
        // tied head cannot express a split readout at all.
        let readout = params.readout_width();
        assert!(
            !(params.tied && params.split_deposit),
            "a tied head is d_model wide and cannot read a split deposit"
        );
        let head = (!params.tied).then(|| {
            if params.embed_rungs <= 1 {
                AssocMemory::single_rect(params.vocab, readout, 1.0)
            } else {
                AssocMemory::ladder_rect(params.vocab, readout, params.schedule, params.embed_rungs)
            }
        });

        // Structure first, and only then the head, off a generator of its own.
        // Drawing the head from the shared stream made turning it on shift the
        // rotation signs and the ingress planes, so "linear head vs lookup
        // head" was never comparing two runs that differed in one thing. This
        // is the same defect as the source once had (§19): a configuration
        // change must not be able to perturb structure it has nothing to do
        // with.
        let rotation = Rotation::new(d_model, rng);
        let ingress = Ingress::new(params.nodes, d_model, params.slots, params.vocab, rng);
        // Drawn unconditionally so the shared stream advances the same way
        // whichever head is in use; a conditional draw would work today and
        // break the moment anything else is constructed after this.
        let mut head_rng = Rng::new(rng.next_u64() ^ 0x4E_AD_5E_ED_00_11_22_33);
        let lookup = match params.head_kind {
            HeadKind::Lookup { slots } => Some(LookupHead::new(
                params.vocab,
                readout,
                slots,
                params.head_top_k,
                params.centre_readout,
                &mut head_rng,
            )),
            HeadKind::Linear => None,
        };
        // Centring costs one `vocab x d_model` accumulator and nothing else;
        // built last so it cannot perturb any generator above it.
        let (centre, centre_count) = if params.centre_readout {
            (
                Some(Mat::zeros(params.vocab, readout)),
                vec![0.0; params.vocab],
            )
        } else {
            (None, Vec::new())
        };
        Self {
            params,
            centre,
            centre_count,
            embedding,
            head,
            lookup,
            rotation,
            ingress,
            embed_buf: vec![0.0; d_model],
            logit_buf: vec![0.0; params.vocab],
        }
    }

    /// Share of readout components that have ever won gate mass. Below one with
    /// `head_top_k` on means slots are frozen out of the competition and the
    /// effective capacity is smaller than it looks.
    pub fn readout_coverage(&self) -> Option<f64> {
        self.lookup.as_ref().map(|l| l.coverage())
    }

    #[inline]
    pub fn params(&self) -> &ModelParams {
        &self.params
    }

    #[inline]
    pub fn ingress(&self) -> &Ingress {
        &self.ingress
    }

    pub fn set_ingress_mode(&mut self, mode: IngressMode) {
        self.ingress.set_mode(mode);
    }

    /// Remembers where a token's mass came to rest, for its next entry.
    /// Returns how far that moved its anchor around the ring.
    pub fn observe_resting_place(&mut self, token: u32, resting: usize) -> usize {
        let d = self.params.d_model();
        let lo = token as usize * d;
        let embedding = self.embedding.read().as_slice()[lo..lo + d].to_vec();
        self.ingress.observe(token, resting, &embedding)
    }

    /// Row `token` of the tied embedding table.
    pub fn embedding_of(&self, token: u32) -> &[f64] {
        let d = self.params.d_model();
        let lo = token as usize * d;
        &self.embedding.read().as_slice()[lo..lo + d]
    }

    /// Turns a token into particles: rotate, slice, split each chunk into a
    /// unit direction and a share of the mass, and place it by content.
    pub fn shatter(&mut self, ring: &Ring, token: u32, position: u64) -> Shattered {
        let (d_head, slots) = (self.params.d_head, self.params.slots);
        let d_model = self.params.d_model();
        let lo = token as usize * d_model;
        let table = self.embedding.read().as_slice();
        let anchor_source = table[lo..lo + d_model].to_vec();
        self.embed_buf.copy_from_slice(&anchor_source);
        self.rotation.forward(&mut self.embed_buf);

        let magnitudes: Vec<f64> = (0..slots)
            .map(|s| norm(&self.embed_buf[s * d_head..(s + 1) * d_head]))
            .collect();
        let scale: f64 = magnitudes.iter().sum();
        assert!(scale > 1e-12, "token {token} has a degenerate embedding");

        let seeds = (0..slots)
            .map(|s| {
                let chunk = &self.embed_buf[s * d_head..(s + 1) * d_head];
                let mag = magnitudes[s];
                let payload: Vec<f64> = if mag > 1e-12 {
                    chunk.iter().map(|x| x / mag).collect()
                } else {
                    // Vanishing chunk: any unit direction will do, and its mass
                    // is zero, so it is absorbed on arrival regardless.
                    let mut v = vec![0.0; d_head];
                    v[0] = 1.0;
                    v
                };
                let node = self
                    .ingress
                    .node_for(ring, token, position, &anchor_source, s);
                (node, s as u16, mag / scale, payload)
            })
            .collect();
        Shattered { seeds, scale }
    }

    /// Undoes the shatter: rescale, concatenate, rotate back.
    pub fn assemble(&self, accumulated: &[f64], scale: f64) -> Vec<f64> {
        let d_model = self.params.d_model();
        assert_eq!(
            accumulated.len(),
            self.params.readout_width(),
            "accumulator width mismatch"
        );
        let mut v: Vec<f64> = accumulated.iter().map(|x| x * scale).collect();
        // The rotation is defined on one `d_model` block, so each half is
        // undone on its own; running it across the pair would mix the payload
        // into the answers and undo the point of separating them.
        self.rotation.inverse(&mut v[..d_model]);
        if self.params.split_deposit {
            self.rotation.inverse(&mut v[d_model..]);
        }
        v
    }

    /// Cross-entropy of the tied output head against `target`, plus the update.
    ///
    /// The gradient of the loss with respect to the tied table is exactly
    /// `(p - y) e^T`, a rank-one outer product — the same shape the
    /// consolidation ladder already accepts, so the output head needs no
    /// machinery of its own. This is the only backward pass in the system.
    /// `learn`, with the current token's running mean removed from `assembled`
    /// first — the readout entry point for the assembled pipeline.
    ///
    /// The mean is updated *after* the loss is charged, so the average behind
    /// the subtraction only ever contains earlier tokens and the loss stays a
    /// valid prequential code length.
    pub fn learn_centred(&mut self, assembled: &[f64], token: u32, target: u32) -> f64 {
        let Some(centre) = self.centre.as_ref() else {
            return self.learn(assembled, target);
        };
        let row = token as usize * assembled.len();
        let centred: Vec<f64> = assembled
            .iter()
            .zip(&centre.as_slice()[row..row + assembled.len()])
            .map(|(x, m)| x - m)
            .collect();
        let loss = self.learn_as(&centred, Some(token), target);
        self.observe_centre(assembled, token);
        loss
    }

    /// Folds one observation into the running mean for `token`.
    fn observe_centre(&mut self, assembled: &[f64], token: u32) {
        let Some(centre) = self.centre.as_mut() else {
            return;
        };
        let n = &mut self.centre_count[token as usize];
        *n += 1.0;
        let blend = 1.0 / *n;
        let row = token as usize * assembled.len();
        for (m, x) in centre.as_mut_slice()[row..row + assembled.len()]
            .iter_mut()
            .zip(assembled)
        {
            *m += blend * (x - *m);
        }
    }

    pub fn learn(&mut self, assembled: &[f64], target: u32) -> f64 {
        self.learn_as(assembled, None, target)
    }

    /// `learn`, telling the head which token the residual belongs to so the
    /// per-token expert can take part. `None` leaves the head exactly as it was.
    fn learn_as(&mut self, assembled: &[f64], token: Option<u32>, target: u32) -> f64 {
        if let Some(lookup) = self.lookup.as_mut() {
            let loss = lookup.learn(assembled, token, target, self.params.learning_rate);
            self.logit_buf.copy_from_slice(lookup.last_mixture());
            return loss;
        }
        let loss = self.cross_entropy_as(assembled, token, target);
        // `logit_buf` holds the softmax after `cross_entropy`; turn it into the
        // error signal in place.
        self.logit_buf[target as usize] -= 1.0;
        let grad = std::mem::take(&mut self.logit_buf);
        let head = self.head.as_mut().unwrap_or(&mut self.embedding);
        head.inject(&grad, assembled, -self.params.learning_rate);
        head.relax();
        self.logit_buf = grad;
        loss
    }

    /// Cross-entropy without updating anything. Leaves the softmax in
    /// `logit_buf` for `learn` to reuse.
    pub fn cross_entropy(&mut self, assembled: &[f64], target: u32) -> f64 {
        self.cross_entropy_as(assembled, None, target)
    }

    fn cross_entropy_as(&mut self, assembled: &[f64], token: Option<u32>, target: u32) -> f64 {
        assert_eq!(
            assembled.len(),
            self.params.readout_width(),
            "assembled width mismatch"
        );
        assert!(
            (target as usize) < self.params.vocab,
            "target out of vocabulary"
        );
        if let Some(lookup) = self.lookup.as_mut() {
            let loss = lookup.score(assembled, token, target);
            self.logit_buf.copy_from_slice(lookup.last_mixture());
            return loss;
        }
        let mut logits = std::mem::take(&mut self.logit_buf);
        let head = self.head.as_ref().unwrap_or(&self.embedding);
        head.read().mul_vec(assembled, &mut logits);

        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - max).exp();
            total += *l;
        }
        for l in logits.iter_mut() {
            *l /= total;
        }
        let loss = -logits[target as usize].max(f64::MIN_POSITIVE).ln();
        self.logit_buf = logits;
        loss
    }

    /// The softmax from the last `cross_entropy` call.
    #[inline]
    pub fn last_probabilities(&self) -> &[f64] {
        &self.logit_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centring_off_is_exactly_the_old_path() {
        // The flag has to be inert when off, or every number recorded before it
        // existed stops being comparable.
        let mut a = Model::new(params(), &mut Rng::new(7));
        let mut p = params();
        p.centre_readout = false;
        let mut b = Model::new(p, &mut Rng::new(7));
        let x: Vec<f64> = (0..a.params().d_model())
            .map(|i| (i as f64 * 0.37).sin())
            .collect();
        for i in 0..40 {
            let target = (i % 5) as u32;
            let la = a.learn(&x, target);
            let lb = b.learn_centred(&x, 3, target);
            assert_eq!(la.to_bits(), lb.to_bits(), "step {i}: {la} vs {lb}");
        }
    }

    #[test]
    fn centring_removes_what_the_token_alone_predicts() {
        // Feed one token a constant vector. Its running mean converges to that
        // vector, so what reaches the head goes to zero: exactly the component
        // the token fixes, and nothing else, is what gets subtracted.
        let mut p = params();
        p.centre_readout = true;
        let mut m = Model::new(p, &mut Rng::new(11));
        let x: Vec<f64> = (0..m.params().d_model())
            .map(|i| (i as f64 * 0.11).cos())
            .collect();
        for _ in 0..200 {
            m.learn_centred(&x, 4, 1);
        }
        let mean = m.centre.as_ref().expect("centring on").as_slice();
        let d = x.len();
        let gap: f64 = x
            .iter()
            .zip(&mean[4 * d..5 * d])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(gap < 1e-6, "running mean did not converge: {gap}");
    }

    #[test]
    fn the_per_token_expert_carries_what_centring_removed() {
        // Hold the assembled vector constant. Its running mean converges to it,
        // so the residual reaching the slots goes to zero and they can only
        // predict uniformly. Anything learned past uniform has to have come
        // through the per-token channel — which is the whole point of adding it,
        // since centring is what took that evidence away.
        let mut p = params();
        p.centre_readout = true;
        p.head_kind = HeadKind::Lookup { slots: 8 };
        p.learning_rate = 1.0;
        let mut m = Model::new(p, &mut Rng::new(5));
        let d = m.params().d_model();
        let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.29).sin()).collect();
        // Two tokens, two different targets: a bigram no context can supply.
        let pairs = [(2u32, 7u32), (3, 1)];
        let mut last = 0.0;
        for i in 0..600 {
            let (t, y) = pairs[i % 2];
            last = m.learn_centred(&x, t, y);
        }
        let uniform = (m.params().vocab as f64).ln();
        assert!(
            last < 0.25 * uniform,
            "per-token expert did not learn the bigram: {last} against uniform {uniform}"
        );
    }

    #[test]
    fn the_per_token_expert_wins_mass_against_real_competitors() {
        // The companion test above forces every slot to predict uniformly, which
        // is exactly why it could not see that a gate logit of zero left this
        // expert with `exp(-2 ln M)` of the softmax mass — four parts in a
        // million at 512 slots, too little responsibility for the gradient to
        // ever lift it. A guard for that bug has to leave the slots as genuine
        // competitors: varying, informative input that they can and do learn
        // from, with the token carrying evidence they cannot reach.
        let mut p = params();
        p.centre_readout = true;
        p.head_kind = HeadKind::Lookup { slots: 64 };
        p.learning_rate = 1.0;
        let mut m = Model::new(p, &mut Rng::new(23));
        let d = m.params().d_model();
        let vocab = m.params().vocab;
        let mut rng = Rng::new(91);
        let draw = |rng: &mut Rng, phase: usize| -> Vec<f64> {
            (0..d)
                .map(|i| ((i + phase) as f64 * 0.19).sin() + 0.05 * rng.next_normal())
                .collect()
        };
        for i in 0..3000 {
            // Context decides most of the target; the token shifts it. A slot
            // reading the residual can learn the first part and never the second.
            let phase = i % 4;
            let token = (i % 3) as u32;
            let target = ((phase * 3 + token as usize) % vocab) as u32;
            let x = draw(&mut rng, phase);
            m.learn_centred(&x, token, target);
        }
        let x = draw(&mut rng, 1);
        m.learn_centred(&x, 2, 0);
        let won = m.lookup.as_ref().expect("lookup head").token_weight;
        assert!(
            won > 0.01,
            "per-token expert won only {won} of the gate against live slots; \
             a zero gate logit would give about exp(-2 ln 64) = {:e}",
            (-2.0f64 * 64.0f64.ln()).exp()
        );
    }

    #[test]
    fn keeping_every_component_is_the_untouched_mixture() {
        // Truncation has to be inert when it keeps everything, or numbers taken
        // before it existed stop being comparable to ones taken after. What
        // makes it inert is that the renormalisation divides by one, so that is
        // what gets asserted rather than a downstream loss that could match by
        // luck.
        let mut p = params();
        p.head_kind = HeadKind::Lookup { slots: 16 };
        p.head_top_k = 16;
        let mut m = Model::new(p, &mut Rng::new(31));
        let d = m.params().d_model();
        let mut rng = Rng::new(77);
        for i in 0..200 {
            let x: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            m.learn(&x, (i % m.params().vocab) as u32);
        }
        let head = m.lookup.as_ref().expect("lookup head");
        assert_eq!(head.active.len(), head.slots, "a component was dropped");
        let mass: f64 = head.active.iter().map(|k| head.gate[*k]).sum();
        assert!(
            (mass - 1.0).abs() < 1e-12,
            "renormalisation was not the identity: {mass}"
        );
        assert_eq!(m.readout_coverage(), Some(1.0));
    }

    #[test]
    fn truncation_leaves_the_mixture_a_distribution() {
        // Dropping components loses their gate mass. If what is kept were not
        // renormalised the mixture would sum to less than one, every loss would
        // be inflated by the same silent factor, and nothing would ever fail.
        let mut p = params();
        p.head_kind = HeadKind::Lookup { slots: 32 };
        p.head_top_k = 4;
        let mut m = Model::new(p, &mut Rng::new(13));
        let d = m.params().d_model();
        let mut rng = Rng::new(101);
        for i in 0..200 {
            let x: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            m.learn(&x, (i % m.params().vocab) as u32);
            let head = m.lookup.as_ref().expect("lookup head");
            assert!(head.active.len() <= 4, "kept more than asked");
            let total: f64 = head.mixture.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "step {i}: mixture sums to {total}"
            );
        }
    }

    fn params() -> ModelParams {
        ModelParams {
            centre_readout: false,
            head_top_k: 0,
            split_deposit: false,
            vocab: 64,
            d_head: 8,
            slots: 8,
            nodes: 16,
            schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
            embed_rungs: 4,
            learning_rate: 0.05,
            tied: true,
            head_kind: HeadKind::Linear,
        }
    }

    #[test]
    fn rotation_round_trips_and_preserves_length() {
        let mut rng = Rng::new(1);
        let rot = Rotation::new(64, &mut rng);
        let mut v = vec![0.0; 64];
        rng.fill_unit_vector(&mut v);
        let original = v.clone();
        rot.forward(&mut v);
        assert!(
            (norm(&v) - 1.0).abs() < 1e-12,
            "Parseval violated: {}",
            norm(&v)
        );
        rot.inverse(&mut v);
        for (got, want) in v.iter().zip(&original) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn rotation_spreads_a_spike_evenly() {
        // The reason for the sign flip and the whole point of rotating before
        // slicing: no chunk may be handed the bulk of the energy.
        let mut rng = Rng::new(2);
        let rot = Rotation::new(64, &mut rng);
        let mut v = vec![0.0; 64];
        v[0] = 1.0;
        rot.forward(&mut v);
        for x in &v {
            assert!(
                (x.abs() - 1.0 / 8.0).abs() < 1e-12,
                "energy not spread: {x}"
            );
        }
    }

    #[test]
    fn shattering_produces_a_probability_measure_over_unit_payloads() {
        let mut rng = Rng::new(3);
        let grid = Ring::new(16);
        let mut model = Model::new(params(), &mut rng);
        for token in 0..64 {
            let s = model.shatter(&grid, token, 0);
            assert_eq!(s.seeds.len(), 8);
            let total: f64 = s.seeds.iter().map(|(_, _, m, _)| m).sum();
            assert!((total - 1.0).abs() < 1e-12, "masses sum to {total}");
            for (_, slot, _, payload) in &s.seeds {
                assert!((norm(payload) - 1.0).abs() < 1e-12);
                assert!((*slot as usize) < 8);
            }
        }
    }

    #[test]
    fn a_transparent_network_reassembles_the_embedding_exactly() {
        // Shatter, return every particle unchanged with all its mass, reassemble.
        // This is the whole ingress/egress path with the network removed, so it
        // isolates the split-and-rejoin arithmetic from everything else.
        let mut rng = Rng::new(4);
        let grid = Ring::new(16);
        let mut model = Model::new(params(), &mut rng);
        let d_head = model.params().d_head;

        for token in [0u32, 7, 41, 63] {
            let want = model.embedding_of(token).to_vec();
            let s = model.shatter(&grid, token, 0);
            let mut accumulated = vec![0.0; model.params().d_model()];
            for (_, slot, mass, payload) in &s.seeds {
                let lo = *slot as usize * d_head;
                for (a, p) in accumulated[lo..lo + d_head].iter_mut().zip(payload) {
                    *a += mass * p;
                }
            }
            let got = model.assemble(&accumulated, s.scale);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-12, "token {token}: {g} != {w}");
            }
        }
    }

    #[test]
    fn ingress_is_content_addressed() {
        // The same token always enters at the same place, and a token whose
        // embedding is perturbed slightly enters nearby. Without both, no
        // topographic map can form and related memories end up unrelated.
        let mut rng = Rng::new(5);
        let grid = Ring::new(32);
        let mut p = params();
        p.nodes = 32;
        let mut model = Model::new(p, &mut rng);

        let a = model.shatter(&grid, 11, 0).seeds[0].0;
        let b = model.shatter(&grid, 11, 0).seeds[0].0;
        assert_eq!(a, b, "ingress is not a function of the token");

        let base = model.embedding_of(11).to_vec();
        let mut near = base.clone();
        near[0] += 1e-4;
        let far = model.embedding_of(30).to_vec();
        let node_of = |t: u32, v: &[f64]| model.ingress.node_for(&grid, t, 0, v, 0);
        assert_eq!(
            node_of(11, &base),
            node_of(11, &near),
            "a tiny change moved the ingress node"
        );
        // Not a claim that unrelated tokens are always far, only that ingress is
        // not collapsing every token onto one node.
        let spread: std::collections::HashSet<u32> =
            (0..64).map(|t| node_of(t, model.embedding_of(t))).collect();
        assert!(
            spread.len() > 20,
            "only {} distinct entry nodes for 64 tokens",
            spread.len()
        );
        let _ = far;
    }

    #[test]
    fn slots_of_one_token_land_on_a_local_patch() {
        // Not one node, not the whole grid. Related fragments must start close
        // enough that their memories are co-located.
        let mut rng = Rng::new(6);
        let grid = Ring::new(32);
        let mut p = params();
        p.nodes = 32;
        let mut model = Model::new(p, &mut rng);
        let s = model.shatter(&grid, 3, 0);
        let nodes: Vec<u32> = s.seeds.iter().map(|(n, _, _, _)| *n).collect();
        let distinct: std::collections::HashSet<_> = nodes.iter().collect();
        assert!(
            distinct.len() >= 6,
            "slots piled up: {} distinct of 8",
            distinct.len()
        );
        for &a in &nodes {
            for &b in &nodes {
                assert!(grid.distance(a, b) <= 8, "slots scattered across the grid");
            }
        }
    }

    #[test]
    fn a_tied_head_cannot_express_an_asymmetric_bigram() {
        // The structural limit, stated as a test rather than an argument. With
        // ê(u) = E_u the score matrix is E E^T, so the unnormalised scores obey
        // logit_v(u) == logit_u(v) exactly. No amount of training moves that,
        // and a Markov transition matrix is not symmetric.
        let mut rng = Rng::new(77);
        let mut model = Model::new(params(), &mut rng);
        for _ in 0..500 {
            let e = model.embedding_of(3).to_vec();
            model.learn(&e, 9);
        }
        let d_model = model.params().d_model();
        let score = |u: u32, v: u32| {
            let (a, b) = (model.embedding_of(u), model.embedding_of(v));
            (0..d_model).map(|i| a[i] * b[i]).sum::<f64>()
        };
        assert!(
            (score(3, 9) - score(9, 3)).abs() < 1e-12,
            "tied scores must be symmetric however they were trained"
        );

        // Untied, the same training breaks the symmetry.
        let mut p = params();
        p.tied = false;
        let mut untied = Model::new(p, &mut Rng::new(77));
        for _ in 0..500 {
            let e = untied.embedding_of(3).to_vec();
            untied.learn(&e, 9);
        }
        // The untied head's score matrix is head * E^T, with no reason to be
        // symmetric: the score it gives 9 after seeing 3 is the one that was
        // trained, and the reverse was not.
        let head = untied.head.as_ref().expect("untied").read().clone();
        let raw = |u: u32, v: u32| {
            let e = untied.embedding_of(u);
            (0..d_model)
                .map(|i| head.get(v as usize, i) * e[i])
                .sum::<f64>()
        };
        assert!(
            (raw(3, 9) - raw(9, 3)).abs() > 1e-6,
            "an untied head should not be forced into symmetry"
        );
    }

    #[test]
    fn a_tied_table_starts_out_predicting_the_current_token() {
        // Not a defect, a consequence of tying, and worth stating because it
        // sets the baseline the language model has to beat. Rows have unit norm
        // and near-orthogonal neighbours, so feeding token X gives X a logit of
        // ~1 against ~1/sqrt(d_model) for everything else. A transparent
        // network therefore predicts "the token you just saw", and learning has
        // to move the prediction forward by that much before it gains anything.
        let mut rng = Rng::new(7);
        let mut model = Model::new(params(), &mut rng);
        let e5 = model.embedding_of(5).to_vec();

        let on_self = model.cross_entropy(&e5, 5);
        let on_other = model.cross_entropy(&e5, 20);
        let uniform = 64f64.ln();
        assert!(on_self < on_other, "tying should favour the current token");
        assert!(
            (on_other - uniform).abs() < 0.5,
            "off-diagonal loss {on_other} vs ln(64)"
        );
        assert!(
            on_self > 2.5,
            "self-preference is one logit, not a solved problem: {on_self}"
        );
    }

    #[test]
    fn the_output_update_reduces_the_loss_it_was_computed_from() {
        let mut rng = Rng::new(7);
        let mut model = Model::new(params(), &mut rng);
        let assembled = model.embedding_of(5).to_vec();

        let start = model.cross_entropy(&assembled, 20);
        let mut last = start;
        let mut worse = 0;
        for _ in 0..200 {
            let now = model.learn(&assembled, 20);
            if now > last {
                worse += 1;
            }
            last = now;
        }
        assert!(last < start, "loss went {start} -> {last}");
        assert_eq!(worse, 0, "the update should never increase its own loss");
    }

    #[test]
    fn consolidation_slows_the_output_head_down() {
        // Measured, not assumed. The ladder resists one-off changes by design,
        // and a gradient arriving at rung 1 leaks outwards, so the same number
        // of steps at the same rate moves the tied table less than a plain
        // matrix would. This is the cost side of DESIGN.md §1.8.3's injection
        // depth question, and it is why the embedding's rung count is a knob
        // rather than a copy of the node schedule.
        let after = |rungs: usize| {
            let mut rng = Rng::new(7);
            let mut p = params();
            p.embed_rungs = rungs;
            let mut model = Model::new(p, &mut rng);
            let assembled = model.embedding_of(5).to_vec();
            let start = model.cross_entropy(&assembled, 20);
            let mut last = start;
            for _ in 0..200 {
                last = model.learn(&assembled, 20);
            }
            (start, last)
        };
        let (plain_start, plain_end) = after(1);
        let (lad_start, lad_end) = after(4);
        assert!(
            (plain_start - lad_start).abs() < 1e-9,
            "same starting point"
        );
        let plain_drop = plain_start - plain_end;
        let ladder_drop = lad_start - lad_end;
        assert!(
            ladder_drop > 0.0 && ladder_drop < plain_drop,
            "{ladder_drop} vs {plain_drop}"
        );
        // Pin the order of magnitude so a change in the schedule shows up here.
        let ratio = ladder_drop / plain_drop;
        assert!(
            (0.1..0.7).contains(&ratio),
            "ladder retains {ratio} of the plain update"
        );
    }

    #[test]
    fn the_head_choice_does_not_perturb_the_network_structure() {
        // Changing the readout must leave the rotation and the ingress map
        // untouched, or an ablation of the head is silently also an ablation of
        // a random draw. This went wrong once here and once for the source
        // (§19), so it is pinned rather than argued.
        let structure = |kind: HeadKind| {
            let mut q = params();
            q.head_kind = kind;
            q.tied = false;
            let grid = Ring::new(q.nodes);
            let model = Model::new(q, &mut Rng::new(4242));
            let mut probe = vec![0.0; model.params().d_model()];
            probe[0] = 1.0;
            model.rotation.forward(&mut probe);
            let anchors: Vec<u32> = (0..8)
                .map(|t| {
                    model
                        .ingress
                        .node_for(&grid, t, 0, model.embedding_of(t), 0)
                })
                .collect();
            (probe, anchors)
        };
        let (rot_a, anchors_a) = structure(HeadKind::Linear);
        let (rot_b, anchors_b) = structure(HeadKind::Lookup { slots: 512 });
        assert_eq!(rot_a, rot_b, "the head changed the rotation");
        assert_eq!(anchors_a, anchors_b, "the head changed the ingress map");
    }

    #[test]
    fn the_lookup_gate_starts_with_a_clear_winner_at_every_size() {
        // The property the initialisation is derived for: the largest gate
        // weight must be O(1) rather than O(1/M), or responsibilities are
        // uniform, every component learns M times too slowly, and the gate can
        // never differentiate. It must hold across sizes without touching the
        // learning rate.
        for slots in [8usize, 64, 256, 1024] {
            let mut rng = Rng::new(1_000 + slots as u64);
            let mut p = params();
            p.head_kind = HeadKind::Lookup { slots };
            p.tied = false;
            let mut model = Model::new(p, &mut rng);
            let x = model.embedding_of(2).to_vec();
            model.cross_entropy(&x, 0);
            let head = model.lookup.as_ref().expect("lookup head");
            let peak = head.gate.iter().copied().fold(0.0, f64::max);
            assert!(
                peak > 0.1,
                "M = {slots}: top gate weight {peak} is near 1/M = {}",
                1.0 / slots as f64
            );
        }
    }

    #[test]
    fn a_lookup_head_mixes_distributions_not_logits() {
        // The property the whole change rests on. Two components, one favouring
        // token 0 and the other token 1: mixing their *distributions* puts mass
        // on both, where mixing their logits and then normalising would not
        // reproduce that mixture. Checked against the explicit mixture.
        let mut rng = Rng::new(91);
        let mut p = params();
        p.head_kind = HeadKind::Lookup { slots: 2 };
        p.tied = false;
        let mut model = Model::new(p, &mut rng);
        let x = model.embedding_of(3).to_vec();

        // Teach the two components apart by alternating targets.
        for i in 0..2_000 {
            model.learn(&x, if i % 2 == 0 { 0 } else { 1 });
        }
        model.cross_entropy(&x, 0);
        let mixed = model.last_probabilities().to_vec();
        let total: f64 = mixed.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "mixture is not a distribution: {total}"
        );
        assert!(
            mixed[0] > 0.2 && mixed[1] > 0.2,
            "mixture collapsed: {:?}",
            &mixed[..4]
        );
    }

    #[test]
    fn a_lookup_head_can_separate_what_a_linear_head_cannot() {
        // An XOR-shaped target: the same input feature must map to different
        // answers depending on a second feature, which is exactly the
        // interaction a linear head cannot express.
        let build = |kind: HeadKind| {
            let mut q = params();
            q.head_kind = kind;
            q.tied = false;
            q.learning_rate = 0.2;
            Model::new(q, &mut Rng::new(92))
        };
        let mut rng = Rng::new(93);
        let d = params().d_model();
        let axis: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                let mut v = vec![0.0; d];
                rng.fill_unit_vector(&mut v);
                v
            })
            .collect();
        let sample = |a: usize, b: usize| -> Vec<f64> {
            let s = |i: usize| if i == 0 { -1.0 } else { 1.0 };
            axis[0]
                .iter()
                .zip(&axis[1])
                .map(|(x, y)| s(a) * x + s(b) * y)
                .collect()
        };

        let mut losses = Vec::new();
        for kind in [HeadKind::Linear, HeadKind::Lookup { slots: 8 }] {
            let mut model = build(kind);
            let mut tail = 0.0;
            for i in 0..8_000 {
                let (a, b) = (i % 2, (i / 2) % 2);
                let target = (a ^ b) as u32;
                let loss = model.learn(&sample(a, b), target);
                if i >= 7_000 {
                    tail += loss;
                }
            }
            losses.push(tail / 1_000.0);
        }
        assert!(
            losses[1] < 0.5 * losses[0],
            "lookup {} did not beat linear {} on an interaction",
            losses[1],
            losses[0]
        );
    }

    #[test]
    fn probabilities_are_a_distribution() {
        let mut rng = Rng::new(8);
        let mut model = Model::new(params(), &mut rng);
        let assembled = model.embedding_of(2).to_vec();
        model.cross_entropy(&assembled, 2);
        let p = model.last_probabilities();
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn width_can_change_after_construction_without_touching_node_weights() {
        // d_head fixes the architecture; slots is a runtime width. Two models
        // differing only in slot count still emit unit payloads of the same
        // width, which is what makes nodes slot-agnostic.
        let grid = Ring::new(16);
        for slots in [4usize, 8, 16] {
            let mut rng = Rng::new(9);
            let mut p = params();
            p.slots = slots;
            let mut model = Model::new(p, &mut rng);
            let s = model.shatter(&grid, 1, 0);
            assert_eq!(s.seeds.len(), slots);
            assert!(s.seeds.iter().all(|(_, _, _, v)| v.len() == 8));
        }
    }
}

/// A soft lookup readout: stored `(key, distribution)` pairs, mixed in
/// probability space.
///
/// `p(next) = sum_m w_m p_m(next)` with `w = softmax(<x_hat, key_m>)` and
/// `p_m = softmax(value_m)`. The mixture is over distributions rather than
/// logits, so the combination is not additive in log space and the readout can
/// express interactions between whatever the assembled vector encodes.
///
/// Keys are left unnormalised on purpose: their magnitude is a learned inverse
/// temperature, so the gate can sharpen without a temperature parameter.
///
/// Their *initial* magnitude is derived rather than chosen. Scores are
/// `<x_hat, key_m>`, so with key entries drawn iid from `N(0, sigma^2)` the
/// scores are iid `N(0, sigma^2)` whatever the width. The largest of `M` such
/// scores sits about `sigma*sqrt(2 ln M)` above zero while the normaliser is
/// about `M exp(sigma^2/2)`, and requiring the winner to hold `O(1)` of the
/// mass gives `sigma*sqrt(2 ln M) - sigma^2/2 = ln M`, whose root is
///
/// ```text
///     sigma = sqrt(2 ln M)
/// ```
///
/// This matters because responsibilities scale the updates. Starting from a
/// flat gate every component has `r_m ~ 1/M`, so both keys and values learn `M`
/// times too slowly and the gate can never differentiate — a self-locking
/// state. Measured: at `M = 64` and a rate that works for `M = 8`, the head
/// scored 3.95 against a linear head's 3.42, and got monotonically worse with
/// more slots. Scaling the initialisation instead of the rate keeps `M` and the
/// learning rate independent, rather than tying two knobs together.
#[derive(Clone, Debug)]
pub struct LookupHead {
    vocab: usize,
    width: usize,
    slots: usize,
    keys: Vec<f64>,
    values: Vec<f64>,
    query: Vec<f64>,
    gate: Vec<f64>,
    component: Vec<f64>,
    mixture: Vec<f64>,
    responsibility: Vec<f64>,
    /// One extra expert per current token, present only when the readout is
    /// centred.
    ///
    /// Centring removes the component the current token fixes, and that
    /// component *is* the order-1 evidence — the most useful single thing the
    /// readout had. Subtracting it without giving it back makes the readout
    /// strictly poorer, which is what the first measurement showed. So it comes
    /// back on its own channel: `token_values[t]` is a distribution learned for
    /// token `t` alone, mixed in probability space alongside the slots with a
    /// gate of its own, which keeps §24's mixture intact rather than folding an
    /// additive term into log space.
    ///
    /// `vocab * vocab` here. Fine at benchmark scale; a real vocabulary would
    /// want this factored or sparse.
    token_values: Vec<f64>,
    token_gate: Vec<f64>,
    token_component: Vec<f64>,
    /// The token whose expert took part in the last `forward`.
    last_token: Option<usize>,
    /// Gate mass the per-token expert won in the last `forward`.
    token_weight: f64,
    /// Components that took part in the last `forward`.
    ///
    /// The mixture costs `O(M V)` a token because every component softmaxes over
    /// the whole vocabulary. At `V = 16` that is nothing; on real text it makes
    /// the readout cost an order of magnitude more than the network it is
    /// reading. Gate mass concentrates, so evaluating only the components that
    /// won any is a cheap approximation — but a component that is never chosen
    /// is never updated either, and a slot frozen out of the competition is the
    /// same structural exclusion that made the per-token expert inert, so
    /// `ever_active` is tracked and reported rather than assumed away.
    active: Vec<usize>,
    ever_active: Vec<bool>,
    top_k: usize,
}

impl LookupHead {
    fn new(
        vocab: usize,
        width: usize,
        slots: usize,
        top_k: usize,
        per_token: bool,
        rng: &mut Rng,
    ) -> Self {
        assert!(slots >= 1, "a lookup head needs at least one slot");
        let sigma = (2.0 * (slots as f64).max(2.0).ln()).sqrt();
        Self {
            vocab,
            width,
            slots,
            keys: (0..slots * width)
                .map(|_| rng.next_normal() * sigma)
                .collect(),
            // Zero values start every component at the uniform distribution, so
            // the head begins by predicting uniformly rather than arbitrarily.
            values: vec![0.0; slots * vocab],
            query: vec![0.0; width],
            gate: vec![0.0; slots],
            component: vec![0.0; slots * vocab],
            mixture: vec![0.0; vocab],
            responsibility: vec![0.0; slots],
            token_values: if per_token {
                vec![0.0; vocab * vocab]
            } else {
                Vec::new()
            },
            // Starts level with the *winning* slot, not at zero.
            //
            // Slot keys are drawn at `sigma = sqrt(2 ln M)` against a unit query,
            // so a slot logit is `N(0, 2 ln M)` and the largest of `M` of them
            // sits near `sigma sqrt(2 ln M) = 2 ln M`. A gate logit of zero
            // against that is `exp(-2 ln M)` of the mass — for 512 slots, four
            // parts in a million. That is not a weak start but a structural
            // exclusion: the expert can never win enough responsibility to learn
            // its way up. The derivation that fixed `sigma` in §24 fixes this
            // too, so it is a derived quantity and not a knob.
            token_gate: if per_token {
                vec![2.0 * (slots as f64).max(2.0).ln(); vocab]
            } else {
                Vec::new()
            },
            token_component: if per_token {
                vec![0.0; vocab]
            } else {
                Vec::new()
            },
            last_token: None,
            token_weight: 0.0,
            active: Vec::with_capacity(slots),
            ever_active: vec![false; slots],
            top_k,
        }
    }

    #[inline]
    fn per_token(&self) -> bool {
        !self.token_gate.is_empty()
    }

    /// The mixture from the last `score` or `learn`.
    pub fn last_mixture(&self) -> &[f64] {
        &self.mixture
    }

    fn forward(&mut self, x: &[f64], token: Option<u32>) {
        debug_assert_eq!(x.len(), self.width);
        let len = norm(x);
        let scale = if len > 1e-12 { 1.0 / len } else { 0.0 };
        for (q, &v) in self.query.iter_mut().zip(x) {
            *q = v * scale;
        }
        for m in 0..self.slots {
            let key = &self.keys[m * self.width..(m + 1) * self.width];
            self.gate[m] = crate::linalg::dot(&self.query, key);
        }
        // The per-token expert competes for the same gate mass as the slots, so
        // its logit joins the softmax rather than being blended in afterwards.
        self.last_token = match (self.per_token(), token) {
            (true, Some(t)) => Some(t as usize),
            _ => None,
        };
        let token_weight = match self.last_token {
            Some(t) => {
                self.gate.push(self.token_gate[t]);
                softmax(&mut self.gate);
                let w = self.gate.pop().expect("just pushed");
                let lo = t * self.vocab;
                self.token_component
                    .copy_from_slice(&self.token_values[lo..lo + self.vocab]);
                softmax(&mut self.token_component);
                w
            }
            None => {
                softmax(&mut self.gate);
                0.0
            }
        };
        // Choose the components worth evaluating. Zero means adaptive: keep the
        // ones that beat a uniform share, which is a derived line rather than a
        // tuned one — a component below `1/M` is contributing less than an
        // average component and cannot move the mixture much.
        self.active.clear();
        if self.top_k == 0 {
            let floor = 1.0 / self.slots as f64;
            self.active
                .extend((0..self.slots).filter(|m| self.gate[*m] >= floor));
            if self.active.is_empty() {
                let best = (0..self.slots)
                    .max_by(|a, b| self.gate[*a].total_cmp(&self.gate[*b]))
                    .unwrap_or(0);
                self.active.push(best);
            }
        } else if self.top_k >= self.slots {
            self.active.extend(0..self.slots);
        } else {
            let mut order: Vec<usize> = (0..self.slots).collect();
            order
                .select_nth_unstable_by(self.top_k, |a, b| self.gate[*b].total_cmp(&self.gate[*a]));
            order.truncate(self.top_k);
            order.sort_unstable();
            self.active.extend(order);
        }
        // Dropping components loses their gate mass, so what is kept is
        // renormalised: the mixture stays a distribution, and a responsibility
        // still sums to one over the components that were actually evaluated.
        let kept: f64 = self.active.iter().map(|m| self.gate[*m]).sum::<f64>() + token_weight;
        let renorm = if kept > 0.0 { 1.0 / kept } else { 0.0 };
        for m in &self.active {
            self.gate[*m] *= renorm;
            self.ever_active[*m] = true;
        }
        let token_weight = token_weight * renorm;

        self.mixture.fill(0.0);
        if self.last_token.is_some() {
            for (mix, &p) in self.mixture.iter_mut().zip(&self.token_component) {
                *mix += token_weight * p;
            }
        }
        self.token_weight = token_weight;
        for k in 0..self.active.len() {
            let m = self.active[k];
            let lo = m * self.vocab;
            let slice = &mut self.component[lo..lo + self.vocab];
            slice.copy_from_slice(&self.values[lo..lo + self.vocab]);
            softmax(slice);
            let w = self.gate[m];
            for (mix, &p) in self.mixture.iter_mut().zip(slice.iter()) {
                *mix += w * p;
            }
        }
    }

    /// Share of components that have ever won gate mass. Below one means slots
    /// are frozen out and the effective capacity is smaller than `M`.
    pub fn coverage(&self) -> f64 {
        let seen = self.ever_active.iter().filter(|x| **x).count();
        seen as f64 / self.ever_active.len().max(1) as f64
    }

    fn score(&mut self, x: &[f64], token: Option<u32>, target: u32) -> f64 {
        self.forward(x, token);
        -self.mixture[target as usize].max(f64::MIN_POSITIVE).ln()
    }

    fn learn(&mut self, x: &[f64], token: Option<u32>, target: u32, lr: f64) -> f64 {
        let loss = self.score(x, token, target);
        let y = target as usize;
        let total = self.mixture[y].max(f64::MIN_POSITIVE);
        // Posterior responsibility of each component for the observed target;
        // these sum to one.
        // Only the components that were evaluated hold a valid `component` row,
        // and only they carry gate mass, so the responsibilities are over the
        // active set and still sum to one.
        for k in 0..self.active.len() {
            let m = self.active[k];
            self.responsibility[m] = self.gate[m] * self.component[m * self.vocab + y] / total;
        }
        for k in 0..self.active.len() {
            let m = self.active[k];
            let r = self.responsibility[m];
            let lo = m * self.vocab;
            for v in 0..self.vocab {
                let g = r * (self.component[lo + v] - if v == y { 1.0 } else { 0.0 });
                self.values[lo + v] -= lr * g;
            }
            // Raise the score of components that explain the target better than
            // their own gate weight already assumed.
            let g = self.gate[m] - r;
            if g != 0.0 {
                let key = &mut self.keys[m * self.width..(m + 1) * self.width];
                for (k, &q) in key.iter_mut().zip(&self.query) {
                    *k -= lr * g * q;
                }
            }
        }
        // The per-token expert takes the identical pair of updates: its
        // distribution moves by its own responsibility, and its gate logit by
        // how much better it explained the target than its weight assumed. No
        // new rule, just one more component.
        if let Some(t) = self.last_token {
            let r = self.token_weight * self.token_component[y] / total;
            let lo = t * self.vocab;
            for v in 0..self.vocab {
                let g = r * (self.token_component[v] - if v == y { 1.0 } else { 0.0 });
                self.token_values[lo + v] -= lr * g;
            }
            self.token_gate[t] -= lr * (self.token_weight - r);
        }
        loss
    }
}

fn softmax(v: &mut [f64]) {
    let peak = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for x in v.iter_mut() {
        *x = (*x - peak).exp();
        total += *x;
    }
    for x in v.iter_mut() {
        *x /= total;
    }
}
