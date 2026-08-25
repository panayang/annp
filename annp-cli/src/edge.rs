//! Routing fixed, memory on the edges, class as one axis of the tensor.
//!
//! See `DESIGN-EDGE.md`. The short version of why this shape:
//!
//! Three earlier variants all died the same way. Their routing consistency --
//! the fraction of probed facts that come back to where training left them --
//! measured 0.078, 0.144 and 0.057 against a chance level of 1/16 = 0.0625.
//! Written knowledge was being looked up somewhere else. Every one of them had
//! routing that depended on adaptive state, and a hard argmax over drifting
//! state amplifies any difference into a different destination.
//!
//! So routing here is stateless. Each node's edge keys are drawn once at
//! construction and never change, and the payload is a deterministic function
//! of the fact's own tokens, so a fact always walks the same path. Consistency
//! is a property of the construction rather than something to be tuned toward.
//!
//! What that buys is the freedom to put the memory somewhere useful: on the
//! edges, sliced by a class index. Conflicting facts -- the same key needing
//! different values in different domains -- land in different slices and never
//! share parameters, so protection comes from where the write goes and not
//! from how slowly it goes. There is no ladder here, deliberately: the whole
//! point is that allocation does the job a slow timescale was failing to do.

use annp_core::rng::Rng;

/// Small-world ring. Fixed for the run: a topology that rewired while the
/// memory was being read would move a fact's address out from under it.
#[derive(Clone, Debug)]
pub struct Ring {
    edges: Vec<Vec<usize>>,
    /// Global index of each (node, slot) edge, so an edge can own parameters.
    edge_id: Vec<Vec<usize>>,
    n_edges: usize,
}

impl Ring {
    pub fn new(n: usize, shortcuts: usize, rng: &mut Rng) -> Self {
        assert!(n >= 4, "need at least four nodes for a ring");
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let mut e = vec![(i + n - 1) % n, (i + 1) % n];
            for _ in 0..shortcuts {
                let d = 1 + (rng.next_below((n / 2) as u64) as usize);
                let t = (i + d) % n;
                if t != i && !e.contains(&t) {
                    e.push(t);
                }
            }
            edges.push(e);
        }
        let mut edge_id = Vec::with_capacity(n);
        let mut next = 0usize;
        for e in &edges {
            let ids: Vec<usize> = (0..e.len())
                .map(|_| {
                    let id = next;
                    next += 1;
                    id
                })
                .collect();
            edge_id.push(ids);
        }
        Self {
            edges,
            edge_id,
            n_edges: next,
        }
    }

    #[inline]
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }
}

/// Memory on the edges, addressed by a fixed path and a class index.
pub struct EdgeMemory {
    ring: Ring,
    vocab: usize,
    d: usize,
    hops: usize,
    classes: usize,
    /// Per-node edge keys. Drawn once, never touched again -- this is the
    /// whole reason a fact can be found later.
    keys: Vec<Vec<Vec<f64>>>,
    /// Token embeddings, owned here: the memory reads tokens, not a globally
    /// computed context vector.
    emb: Vec<f64>,
    /// `E[edge, class]`, each a d x d associative memory. The third-order
    /// tensor whose first axis is the routing structure and whose second is
    /// the class -- not a stack of matrices indexed from outside, since which
    /// slice is live is decided by where the particle went and what regime
    /// the stream is in.
    edge_mem: Vec<f64>,
    /// Readout. Class-indexed when `class_readout` is set.
    ///
    /// A shared readout is the remaining place where domains still collide.
    /// The edge slices are allocated, so training domain 1 cannot touch the
    /// slices domain 0 wrote -- but every step writes densely over the whole
    /// vocabulary in the readout, so domain 1 overwrites exactly the rows
    /// domain 0 needs to be read back through. Measured with the readout
    /// shared: within-round forgetting grew every cycle (+3.73, +4.96,
    /// +5.35 bits), which is the opposite of the savings the allocation
    /// story predicts. Allocation protected the memory and left the readout
    /// unprotected.
    readout: Vec<f64>,
    class_readout: bool,
    /// Hidden reservoirs behind the readout, one Benna-Fusi ladder per class
    /// slice. Empty when `rungs <= 1`.
    ///
    /// The ladder was falsified earlier and this is not a reprieve for the
    /// version that failed. That one sat on a memory shared by every domain,
    /// so its deep rungs averaged values that genuinely conflicted -- the
    /// same key needing different answers per domain -- and produced a blend
    /// belonging to none of them. Here a slice is `R[class]`, and a class is
    /// one regime, so the deep rungs average a *consistent* quantity. The
    /// premise lesson 1 always claimed the mechanism needed, and never had,
    /// is supplied by the allocation.
    ///
    /// It is also the only place erosion actually happens. Edge slices are
    /// keyed by (edge, class) and diffusion is activity-gated, so while a
    /// domain is away its slices are neither written nor advanced. What the
    /// other domains overwrite is the readout.
    readout_rungs: Vec<Vec<f64>>,
    lad_cap: Vec<f64>,
    lad_cond: Vec<f64>,
    /// Slow global context, the only thing that can tell Mode B's domains
    /// apart: its entities are shared, so the payload is byte-identical
    /// across all four and the domain lives only in what has been passing by.
    ctx: Vec<f64>,
    ctx_rate: f64,
    /// Class prototypes. Only the first `n_active` are in use.
    ///
    /// Fixed random prototypes make it a per-seed lottery whether the domains
    /// land in different cells: two domains can each sit rock-steady in one
    /// class and still be sitting in the *same* class, which puts their
    /// conflicting targets in one slice. Growing instead of drawing removes
    /// the lottery -- a prototype is placed where a genuinely novel context
    /// appeared, so a new regime gets its own cell by construction rather
    /// than by luck.
    proto: Vec<f64>,
    n_active: usize,
    /// Running mean/variance of the best match against the live prototypes.
    /// Novelty is judged against this rather than a hardcoded similarity,
    /// so there is no magic constant and no scale to tune.
    sim_mean: f64,
    sim_var: f64,
    sim_n: f64,
    /// How many standard deviations below the running mean counts as novel.
    grow_k: f64,
    /// Recent context directions, used to check that adding a prototype does
    /// not move contexts that were already assigned. §16 promised this check:
    /// if an old context changes class, the knowledge written under the old
    /// one is stranded, and the growth rule is unsound.
    ctx_history: std::collections::VecDeque<Vec<f64>>,
    /// One context sample per domain that has already written knowledge.
    ///
    /// `ctx_history` cannot answer the stranding question: it holds the last
    /// 64 contexts, nearly all from the domain that just triggered the
    /// growth, and those are exactly the ones the new prototype is *supposed*
    /// to take. Measuring against them reported a 0.28-0.57 "steal" rate for
    /// what was almost certainly correct behaviour. Stranding is when some
    /// *other* domain -- one with knowledge already written under its old
    /// class -- changes class, so that is what has to be sampled.
    domain_ctx_sample: std::collections::HashMap<usize, Vec<f64>>,
    growth_events: f64,
    growth_steals: f64,
    growth_checked: f64,
    /// When true the class is a fixed hash of the fact's own tokens instead
    /// of the inferred context.
    ///
    /// This is the control that separates the two things the class axis
    /// could be doing. It multiplies the number of independent slices by
    /// `classes`, which reduces interference all by itself; and it is
    /// supposed to carry which regime the stream is in. A hash keeps the
    /// first and destroys the second -- identical slice count, zero context
    /// information. If the hash does as well, the class is address bits and
    /// nothing more, and the design's claim about inferred context is wrong.
    hash_class: bool,
    /// Traffic per (edge, class), decayed. Drives directed forgetting.
    usage: Vec<f64>,
    /// Deferred decay factors. Directed forgetting multiplied every weight of
    /// every slice on every observation -- O(slices * vocab * d) per step, and
    /// the reason a single forgetting cell cost about three hours. The stored
    /// weights are now the true weights divided by these, and a slice is
    /// brought back to true units only when it is next touched. Exact, not an
    /// approximation: decay is a scalar multiply and commutes with the folding.
    readout_scale: Vec<f64>,
    edge_scale: Vec<f64>,
    /// Timescale of the usage EMA, as a fraction per observation.
    ///
    /// This has to span a full rotation, not a structural quantity. Set from
    /// the slice count (1/512) it was forty times too fast: one domain visit
    /// is thousands of observations, so every domain not currently being
    /// visited fell below its share and got decayed. The rule then could not
    /// tell "retired" from "between visits", and forgetting destroyed live
    /// knowledge along with dead -- active accuracy fell 47.1% -> 8.8% as the
    /// rate rose, which is not reclamation, it is demolition.
    usage_decay: f64,
    /// Scales each write by addressing confidence: eta * margin / (margin +
    /// gate). Zero disables it, leaving eta untouched.
    ///
    /// This does not try to route better. It breaks the coupling between "I do
    /// not recognise this" and "therefore write hard" -- a misrouted write
    /// meets a slice that does not know the fact, so its error and hence its
    /// step are near-maximal, and the wrongest writes do the most damage.
    gate: f64,
    /// Hard ceiling on total classes when expansion is enabled. 0 keeps the
    /// old behaviour, in which `classes` is a budget growth may fill but
    /// never exceed.
    ///
    /// With it set, capacity is allocated only when a genuinely novel regime
    /// arrives that no existing class explains, so the parameter count is a
    /// consequence of the stream rather than a constant chosen in advance.
    /// One edge memory for all classes; the readout stays per-class.
    share_edge: bool,
    /// A readout block common to every class, added to the private one.
    ///
    /// The encoder fix removed "a new class inherits zeros" from the smaller
    /// half only: at 12 classes the edge memory is 65K parameters and the
    /// readout is 1.57M, so most of what a new regime has to learn from
    /// nothing still lives here. The source also has a shared hub tier, so
    /// every domain currently relearns the same hub facts privately.
    ///
    /// It cannot be shared outright the way the encoder can -- Mode B is
    /// defined by the same (entity, relation) mapping to different targets
    /// per domain, so a fully shared readout could not tell domains apart.
    /// Hence base plus private correction, with the private part able to
    /// override. Kept full-rank on purpose: a domain has hundreds of
    /// unrelated targets, so a rank-r correction could not address them. This
    /// version buys transfer, not compactness.
    readout_shared: Vec<f64>,
    expand_cap: usize,
    /// Consecutive novel observations required before a class is allocated.
    ///
    /// Novelty means "similarity to the best prototype fell grow_k sd below
    /// its running mean", and a domain switch drives exactly that dip -- the
    /// same transition that makes 44% of write magnitude land as intrusion.
    /// Undebounced, growth fires on boundaries rather than on regimes,
    /// allocating a slice for every crossing. A genuinely new regime stays
    /// novel; a transition does not.
    grow_hold: usize,
    novel_run: usize,
    /// Growth events bucketed by observations since the domain last changed.
    /// Spread out means growth tracks regimes; piled into the first bucket
    /// means it tracks boundaries.
    grow_at: [usize; 4],
    /// Addressing-blind control: caps the per-write update norm. Bounds the
    /// same damage without knowing anything about addressing, so it separates
    /// "magnitude control works" from "knowing where you are works".
    clip: f64,
    margin_now: f64,
    forget: f64,

    // forward trace, kept for the backward pass along the path
    path_edge: Vec<usize>,
    path_in: Vec<Vec<f64>>,
    path_pre: Vec<Vec<f64>>,
    path_norm: Vec<f64>,
    payload: Vec<f64>,
    /// The untouched content payload, used for every routing decision.
    ///
    /// Keeping this separate is the whole of the consistency guarantee. The
    /// first version routed on the running payload, which the edge memories
    /// rewrite as it travels, so hops two and three depended on learned
    /// parameters and the path drifted as training went on -- measured
    /// consistency 0.527, not the 1.000 the construction was supposed to
    /// give. Route on content, process on the running payload: the path is
    /// then a function of the fact alone and cannot move.
    route_payload: Vec<f64>,
    logits: Vec<f64>,
    probs: Vec<f64>,
    grad_p: Vec<f64>,
    grad_a: Vec<f64>,
    class_now: usize,
    /// The class the last forward actually used, which differs from
    /// `class_now` in hash mode.
    class_used: usize,
    // diagnostics
    train_home: std::collections::HashMap<(usize, usize), usize>,
    consistency_hits: f64,
    consistency_n: f64,
    /// Accumulated magnitude of writes since the last read, readout and edge
    /// memory separately.
    ///
    /// This is what separates a savings effect from plain convergence. As
    /// predictions improve the residual (p - e_y) shrinks, so every write
    /// shrinks, so interference shrinks -- forgetting would fall for a
    /// reason that has nothing to do with allocation. If the two readout
    /// conditions have the same write trajectory while their forgetting
    /// curves go opposite ways, convergence cannot be the explanation.
    write_norm_readout: f64,
    write_norm_edge: f64,
    class_switches: f64,
    class_steps: f64,
    last_class: usize,
    edge_visits: Vec<f64>,
    /// Which classes each domain's facts actually landed in.
    ///
    /// Class-switch rate cannot see the failure that matters here: two
    /// domains can each sit rock-steady in one class and still be sitting in
    /// the *same* class, which puts their conflicting targets in one slice.
    /// The prototypes are fixed random draws, so whether four domains land in
    /// four different cells is luck -- and that is a per-seed lottery the
    /// mechanism-level diagnostics all pass regardless of the outcome.
    domain_class: std::collections::HashMap<(usize, usize), f64>,
    cur_domain: usize,
    /// Instrumentation only -- never read by the mechanism.
    ///
    /// Forgetting here is intrusion: a write landing in a class that is not
    /// the live domain's home. Because a misrouted write meets a slice that
    /// does not know the fact, its error is near-maximal and the delta rule
    /// makes its step near-maximal, so the wrongest writes do the most damage.
    /// Counting intrusions therefore understates them; the magnitude share is
    /// the honest figure, and both are recorded.
    ///
    /// Bucketed by observations since the domain last changed, to distinguish
    /// intrusion concentrated at transitions (which an addressing-confidence
    /// gate would catch) from intrusion spread through the visit (which it
    /// would not).
    dc_bucket: std::collections::HashMap<(usize, usize, usize), f64>,
    dc_norm: std::collections::HashMap<(usize, usize), f64>,
    /// Write magnitude attributed to each domain, for the marginal-cost curve:
    /// what the k-th regime costs to acquire. Savings in this architecture is
    /// not a fading trace that makes relearning cheap -- there is no decay for
    /// a trace to fade through -- it is the k-th regime costing less than the
    /// first because shared structure has already been paid for.
    write_by_domain: std::collections::HashMap<usize, f64>,
    visit_step: f64,
}

impl EdgeMemory {
    /// Advances the Benna-Fusi diffusion on one class slice.
    ///
    /// Activity-gated: only the class just written advances, so a domain that
    /// is away neither leaks toward its neighbours nor drifts. Explicit Euler
    /// from pre-update values throughout, matching `annp_core::ladder`.
    ///
    /// Time is counted in activations of this slice, not in stream steps. The
    /// two differ by the slice's duty cycle, and confusing them is the single
    /// mistake this project has made most often (DESIGN-EDGE.md, s12).
    fn relax_slice(&mut self, c: usize) {
        if self.readout_rungs.is_empty() {
            return;
        }
        let m = self.readout_rungs.len() + 1;
        let n = self.vocab * self.d;
        let base = c * n;
        for i in 0..n {
            let idx = base + i;
            let mut prev_flux = 0.0;
            for k in 0..m {
                let cur = if k == 0 {
                    self.readout[idx]
                } else {
                    self.readout_rungs[k - 1][idx]
                };
                let flux = if k + 1 < m {
                    self.lad_cond[k] * (self.readout_rungs[k][idx] - cur)
                } else {
                    0.0
                };
                let delta = (flux - prev_flux) / self.lad_cap[k];
                if k == 0 {
                    self.readout[idx] = cur + delta;
                } else {
                    self.readout_rungs[k - 1][idx] = cur + delta;
                }
                prev_flux = flux;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes: usize,
        shortcuts: usize,
        hops: usize,
        classes: usize,
        d: usize,
        vocab: usize,
        ctx_rate: f64,
        forget: f64,
        hash_class: bool,
        class_readout: bool,
        init_classes: usize,
        grow_k: f64,
        cycle_observes: f64,
        rungs: usize,
        lad_r: f64,
        lad_g1: f64,
        gate: f64,
        clip: f64,
        share_edge: bool,
        share_readout: bool,
        expand_cap: usize,
        grow_hold: usize,
        rng: &mut Rng,
    ) -> Self {
        // Explicit Euler on the ladder is stable only while the fastest
        // conductance is well under the unit capacity. Past that the surface
        // rung oscillates and the readout is destroyed, which prints as
        // accuracy 0.0 at every delay and therefore as forgetting +0.000 --
        // a total collapse wearing the costume of perfect retention.
        assert!(
            rungs < 2 || lad_g1 < 0.5,
            "ladder conductance g1={lad_g1} is too large for explicit Euler \
             (need < 0.5); the readout would diverge and report itself as \
             zero forgetting. Raise --edge-ladder-visits."
        );
        let ring = Ring::new(nodes, shortcuts, rng);
        let mut keys = Vec::with_capacity(nodes);
        for n in 0..nodes {
            let deg = ring.edges[n].len();
            let mut k = Vec::with_capacity(deg);
            for _ in 0..deg {
                let mut v = vec![0.0; d];
                rng.fill_unit_vector(&mut v);
                k.push(v);
            }
            keys.push(k);
        }
        let mut emb = vec![0.0; vocab * d];
        for v in 0..vocab {
            rng.fill_unit_vector(&mut emb[v * d..(v + 1) * d]);
        }
        let mut proto = vec![0.0; classes * d];
        for c in 0..classes {
            rng.fill_unit_vector(&mut proto[c * d..(c + 1) * d]);
        }
        let n_edges = ring.n_edges();
        Self {
            vocab,
            d,
            hops,
            classes,
            keys,
            emb,
            // Zero, not random: an untrained edge should add nothing to a
            // payload passing through, so an unvisited path is the identity
            // rather than a random rotation.
            edge_mem: vec![0.0; n_edges * classes * d * d],
            readout: vec![0.0; if class_readout { classes * vocab * d } else { vocab * d }],
            class_readout,
            readout_rungs: if rungs > 1 && class_readout {
                (1..rungs).map(|_| vec![0.0; classes * vocab * d]).collect()
            } else {
                Vec::new()
            },
            gate,
            clip,
            expand_cap,
            share_edge,
            readout_shared: if share_readout && class_readout {
                vec![0.0; vocab * d]
            } else {
                Vec::new()
            },
            grow_hold,
            novel_run: 0,
            grow_at: [0; 4],
            margin_now: 1.0,
            lad_cap: (0..rungs).map(|k| lad_r.powi(k as i32)).collect(),
            lad_cond: (0..rungs).map(|k| lad_g1 * lad_r.powi(-(k as i32))).collect(),
            ctx: vec![0.0; d],
            ctx_rate,
            proto,
            n_active: init_classes.clamp(1, classes),
            sim_mean: 0.0,
            sim_var: 1.0,
            sim_n: 0.0,
            grow_k,
            ctx_history: std::collections::VecDeque::with_capacity(64),
            domain_ctx_sample: std::collections::HashMap::new(),
            growth_events: 0.0,
            growth_steals: 0.0,
            growth_checked: 0.0,
            hash_class,
            usage: vec![0.0; n_edges * classes],
            readout_scale: vec![1.0; classes.max(1)],
            edge_scale: vec![1.0; n_edges * classes],
            usage_decay: 1.0 / cycle_observes.max(1.0),
            forget,
            path_edge: Vec::with_capacity(hops),
            path_in: (0..hops).map(|_| vec![0.0; d]).collect(),
            path_pre: (0..hops).map(|_| vec![0.0; d]).collect(),
            path_norm: vec![0.0; hops],
            payload: vec![0.0; d],
            route_payload: vec![0.0; d],
            logits: vec![0.0; vocab],
            probs: vec![0.0; vocab],
            grad_p: vec![0.0; d],
            grad_a: vec![0.0; d],
            class_now: 0,
            class_used: 0,
            train_home: std::collections::HashMap::new(),
            consistency_hits: 0.0,
            consistency_n: 0.0,
            write_norm_readout: 0.0,
            write_norm_edge: 0.0,
            class_switches: 0.0,
            class_steps: 0.0,
            last_class: usize::MAX,
            edge_visits: vec![0.0; n_edges],
            domain_class: std::collections::HashMap::new(),
            cur_domain: 0,
            dc_bucket: std::collections::HashMap::new(),
            dc_norm: std::collections::HashMap::new(),
            write_by_domain: std::collections::HashMap::new(),
            visit_step: 0.0,
            ring,
        }
    }

    /// Feeds a stream token to the slow context. This is where the domain
    /// signal enters: in Mode B the targets are the only tokens that differ
    /// between domains, so absorbing them is what lets the class tell them
    /// apart at all.
    pub fn absorb_token(&mut self, token: usize) {
        let r = self.ctx_rate;
        let e = &self.emb[token * self.d..(token + 1) * self.d];
        for (c, x) in self.ctx.iter_mut().zip(e) {
            *c += r * (x - *c);
        }
        let (c, sim, margin) = self.class_sim_margin();
        self.margin_now = margin;
        self.sim_n += 1.0;
        let delta = sim - self.sim_mean;
        self.sim_mean += delta / self.sim_n.min(4096.0);
        self.sim_var += (delta * (sim - self.sim_mean) - self.sim_var) / self.sim_n.min(4096.0);
        let mut v = self.ctx.clone();
        normalize(&mut v);
        if self.ctx_history.len() >= 64 {
            self.ctx_history.pop_front();
        }
        self.ctx_history.push_back(v.clone());
        self.domain_ctx_sample.insert(self.cur_domain, v);
        self.maybe_grow(sim);
        let c = if self.n_active > c { self.class_of() } else { c };
        self.class_steps += 1.0;
        if self.last_class != usize::MAX && self.last_class != c {
            self.class_switches += 1.0;
        }
        self.last_class = c;
        self.class_now = c;
    }

    /// Appends one class slice at run time, leaving every existing weight
    /// byte-identical.
    ///
    /// This is the difference between growing into a pre-allocated ceiling --
    /// which is all `maybe_grow` ever did -- and actually expanding. Both the
    /// readout and, since the layout became class-major, the edge memory are
    /// class-major, so a new class is a pure append: no existing offset moves
    /// and nothing already learned is touched or relearned.
    ///
    /// The monolithic control cannot do this at all. Its capacity is the SDR
    /// width, and widening it changes the projection every code was written
    /// through, so expansion there means retraining from scratch.
    ///
    /// Non-destructive is a claim about weights, not about behaviour: a new
    /// prototype can still capture contexts an existing class was serving.
    /// That is what the growth-steal statistic measures, and it is why the
    /// prototype is placed on the residual orthogonal to the existing ones.
    pub fn expand_classes(&mut self, extra: usize) {
        if extra == 0 {
            return;
        }
        let ne = self.ring.n_edges();
        let new_total = self.classes + extra;
        self.edge_mem.resize(new_total * ne * self.d * self.d, 0.0);
        self.edge_scale.resize(new_total * ne, 1.0);
        self.usage.resize(new_total * ne, 0.0);
        if self.class_readout {
            self.readout.resize(new_total * self.vocab * self.d, 0.0);
            for r in self.readout_rungs.iter_mut() {
                r.resize(new_total * self.vocab * self.d, 0.0);
            }
            self.readout_scale.resize(new_total, 1.0);
        }
        self.proto.resize(new_total * self.d, 0.0);
        self.classes = new_total;
    }

    /// Index of the (edge, class) slice.
    ///
    /// Class-major on purpose. Edge-major -- `eid * classes + c` -- puts each
    /// edge's classes together, so raising the class count shifts the offset
    /// of every existing entry and capacity expansion becomes a whole-array
    /// re-layout. Class-major makes a new class a pure append, which is the
    /// property that lets this architecture grow without disturbing, or
    /// retraining, anything already stored.
    #[inline]
    fn slot(&self, eid: usize, c: usize) -> usize {
        if self.share_edge {
            // One encoder for every class. A fact's payload is built from its
            // entity and relation, and in Mode B those are byte-identical
            // across domains -- only the target differs. So the transform
            // belongs to everyone and only the readout is domain-specific.
            // Privatising it as well gave each new class a block of zeros at
            // every edge, which is why nothing amortised and why the k-th
            // regime never got cheaper.
            eid
        } else {
            c * self.ring.n_edges() + eid
        }
    }

    /// Folds a readout slice's deferred decay back into its weights.
    ///
    /// Must run before anything reads or writes the slice, so that every
    /// access outside these two helpers sees true weights and needs no
    /// knowledge of the scaling.
    fn materialize_readout(&mut self, c: usize) {
        let k = self.readout_scale[c];
        if k == 1.0 {
            return;
        }
        let rb = c * self.vocab * self.d;
        for w in &mut self.readout[rb..rb + self.vocab * self.d] {
            *w *= k;
        }
        self.readout_scale[c] = 1.0;
    }

    fn materialize_edge(&mut self, s: usize) {
        let k = self.edge_scale[s];
        if k == 1.0 {
            return;
        }
        let b = s * self.d * self.d;
        for w in &mut self.edge_mem[b..b + self.d * self.d] {
            *w *= k;
        }
        self.edge_scale[s] = 1.0;
    }

    /// Offset of the live readout slice.
    #[inline]
    fn rbase(&self) -> usize {
        if self.class_readout {
            self.class_used * self.vocab * self.d
        } else {
            0
        }
    }

    fn class_of(&self) -> usize {
        self.class_and_sim().0
    }

    fn class_and_sim(&self) -> (usize, f64) {
        let (c, m, _) = self.class_sim_margin();
        (c, m)
    }

    /// Best class, its similarity, and its margin over the runner-up.
    ///
    /// The margin is the addressing confidence. Intrusion is concentrated
    /// where it is small: 98.9% of writes in the first 100 observations after
    /// a domain switch land outside the live domain's class, falling to 0% by
    /// 2000. A write worth 44% of the total magnitude is being aimed by a
    /// signal that has not settled yet.
    fn class_sim_margin(&self) -> (usize, f64, f64) {
        let mut best = 0usize;
        let (mut m1, mut m2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for c in 0..self.n_active {
            let p = &self.proto[c * self.d..(c + 1) * self.d];
            let m: f64 = p.iter().zip(&self.ctx).map(|(a, b)| a * b).sum();
            if m > m1 {
                m2 = m1;
                m1 = m;
                best = c;
            } else if m > m2 {
                m2 = m;
            }
        }
        let margin = if m2.is_finite() { m1 - m2 } else { 1.0 };
        (best, m1, margin)
    }

    /// Adds a prototype at the current context when it is far enough below
    /// the running distribution of best matches to count as a new regime.
    fn maybe_grow(&mut self, sim: f64) {
        if self.sim_n < 64.0 {
            return;
        }
        if self.n_active >= self.classes {
            // Out of pre-allocated slots. Expanding here is what makes the
            // capacity on-demand rather than a budget fixed up front, and by
            // `expand_classes` it costs nothing already learned.
            if self.expand_cap > self.classes {
                self.expand_classes(1);
            } else {
                return;
            }
        }
        let sd = self.sim_var.max(1e-12).sqrt();
        if self.sim_mean - sim <= self.grow_k * sd {
            self.novel_run = 0;
            return;
        }
        self.novel_run += 1;
        if self.novel_run < self.grow_hold {
            return;
        }
        self.novel_run = 0;
        self.grow_at[match self.visit_step as u64 {
            0..=99 => 0,
            100..=499 => 1,
            500..=1999 => 2,
            _ => 3,
        }] += 1;
        // Where each *other* domain's context sits before the prototype is
        // added. The domain that triggered the growth is excluded: it is
        // meant to move.
        let others: Vec<Vec<f64>> = self
            .domain_ctx_sample
            .iter()
            .filter(|(d, _)| **d != self.cur_domain)
            .map(|(_, v)| v.clone())
            .collect();
        let before: Vec<usize> = others
            .iter()
            .map(|h| {
                let mut b = 0usize;
                let mut bm = f64::NEG_INFINITY;
                for c in 0..self.n_active {
                    let p = &self.proto[c * self.d..(c + 1) * self.d];
                    let m: f64 = p.iter().zip(h).map(|(a, x)| a * x).sum();
                    if m > bm {
                        bm = m;
                        b = c;
                    }
                }
                b
            })
            .collect();

        // The new prototype takes only what the existing ones do not already
        // explain. Placing it at the raw context direction instead makes it
        // beat every old prototype at once: in high dimension a random unit
        // vector is nearly orthogonal to data, so a data-placed prototype has
        // a far higher dot product with *everything* and wins every arg max.
        // Measured that way, growth-steal came out at 1.000 -- every
        // remembered context changed class and its knowledge was stranded.
        // The residual keeps the new direction novel without making it
        // dominant.
        let n = self.n_active;
        let mut v = self.ctx.clone();
        normalize(&mut v);
        for c in 0..n {
            let p: Vec<f64> = self.proto[c * self.d..(c + 1) * self.d].to_vec();
            let dot: f64 = p.iter().zip(&v).map(|(a, b)| a * b).sum();
            for (x, pc) in v.iter_mut().zip(&p) {
                *x -= dot * pc;
            }
        }
        normalize(&mut v);
        self.proto[n * self.d..(n + 1) * self.d].copy_from_slice(&v);
        self.n_active += 1;
        self.growth_events += 1.0;

        self.growth_checked += others.len() as f64;
        for (h, &was) in others.iter().zip(&before) {
            let mut b = 0usize;
            let mut bm = f64::NEG_INFINITY;
            for c in 0..self.n_active {
                let p = &self.proto[c * self.d..(c + 1) * self.d];
                let m: f64 = p.iter().zip(h).map(|(a, x)| a * x).sum();
                if m > bm {
                    bm = m;
                    b = c;
                }
            }
            if b != was {
                self.growth_steals += 1.0;
            }
        }
    }

    /// Classes in use, and the fraction of remembered contexts that changed
    /// class when a prototype was added. The second number must stay at zero:
    /// a context that moves leaves its knowledge stranded at the old class.
    pub fn growth_stats(&self) -> (f64, f64) {
        (
            self.n_active as f64,
            if self.growth_checked > 0.0 {
                self.growth_steals / self.growth_checked
            } else {
                0.0
            },
        )
    }

    /// Walks the fixed path, recording what the backward pass needs.
    fn forward(&mut self, entity: usize, relation: usize) {
        self.payload.iter_mut().for_each(|v| *v = 0.0);
        for t in [entity, relation] {
            for (p, e) in self
                .payload
                .iter_mut()
                .zip(&self.emb[t * self.d..(t + 1) * self.d])
            {
                *p += e;
            }
        }
        normalize(&mut self.payload);
        self.route_payload.copy_from_slice(&self.payload);

        self.path_edge.clear();
        let mut node = 0usize;
        // Ingress by the same fixed-key rule, over every node.
        let mut best = f64::NEG_INFINITY;
        for (n, ks) in self.keys.iter().enumerate() {
            let m: f64 = ks[0]
                .iter()
                .zip(&self.route_payload)
                .map(|(a, b)| a * b)
                .sum();
            if m > best {
                best = m;
                node = n;
            }
        }

        let c = if self.hash_class {
            (entity.wrapping_mul(31).wrapping_add(relation.wrapping_mul(131))) % self.classes
        } else {
            self.class_now
        };
        self.class_used = c;
        for h in 0..self.hops {
            // Fixed router: pick the out-edge whose key best matches.
            let mut slot = 0usize;
            let mut best_m = f64::NEG_INFINITY;
            for (s, k) in self.keys[node].iter().enumerate() {
                let m: f64 = k.iter().zip(&self.route_payload).map(|(a, b)| a * b).sum();
                if m > best_m {
                    best_m = m;
                    slot = s;
                }
            }
            let eid = self.ring.edge_id[node][slot];
            self.path_edge.push(eid);
            self.edge_visits[eid] += 1.0;

            self.path_in[h].copy_from_slice(&self.payload);
            let sl = self.slot(eid, c);
            self.materialize_edge(sl);
            let base = sl * self.d * self.d;
            for r in 0..self.d {
                let row = &self.edge_mem[base + r * self.d..base + (r + 1) * self.d];
                let dot: f64 = row.iter().zip(&self.payload).map(|(a, b)| a * b).sum();
                self.path_pre[h][r] = dot;
            }
            for (p, a) in self.payload.iter_mut().zip(&self.path_pre[h]) {
                *p += a.tanh();
            }
            self.path_norm[h] = self.payload.iter().map(|v| v * v).sum::<f64>().sqrt();
            normalize(&mut self.payload);

            node = self.ring.edges[node][slot];
        }

        if self.class_readout {
            let c = self.class_used;
            self.materialize_readout(c);
        }
        let rb = self.rbase();
        let sh = !self.readout_shared.is_empty();
        for v in 0..self.vocab {
            let row = &self.readout[rb + v * self.d..rb + (v + 1) * self.d];
            let mut z: f64 = row.iter().zip(&self.payload).map(|(a, b)| a * b).sum();
            if sh {
                let base = &self.readout_shared[v * self.d..(v + 1) * self.d];
                z += base
                    .iter()
                    .zip(&self.payload)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
            self.logits[v] = z;
        }
    }

    pub fn predict_fact(&mut self, entity: usize, relation: usize, target: usize) -> (f64, bool) {
        self.forward(entity, relation);
        score(&self.logits, target, &mut self.probs)
    }

    pub fn observe_fact(&mut self, entity: usize, relation: usize, target: usize, eta: f64) {
        self.forward(entity, relation);
        let _ = score(&self.logits, target, &mut self.probs);

        // Write hard only where the address is known. Both knobs act on the
        // same quantity -- how much this observation is allowed to change --
        // so a gain from the gate that the clip also produces is magnitude
        // control, not addressing awareness.
        let mut eta = eta;
        if self.gate > 0.0 {
            eta *= self.margin_now.max(0.0) / (self.margin_now.max(0.0) + self.gate);
        }
        if self.clip > 0.0 {
            let err: f64 = self
                .probs
                .iter()
                .enumerate()
                .map(|(v, p)| if v == target { 1.0 - p } else { *p })
                .map(|g| g * g)
                .sum::<f64>()
                .sqrt();
            let n = eta * err;
            if n > self.clip {
                eta *= self.clip / n;
            }
        }

        // Readout, delta rule on the final payload.
        let rb = self.rbase();
        let sh_w = !self.readout_shared.is_empty();
        let mut wn_r = 0.0f64;
        for v in 0..self.vocab {
            let g = if v == target {
                1.0 - self.probs[v]
            } else {
                -self.probs[v]
            };
            let step = eta * g;
            let row = &mut self.readout[rb + v * self.d..rb + (v + 1) * self.d];
            for (w, p) in row.iter_mut().zip(&self.payload) {
                *w += step * p;
            }
            if sh_w {
                let base = &mut self.readout_shared[v * self.d..(v + 1) * self.d];
                for (w, p) in base.iter_mut().zip(&self.payload) {
                    *w += step * p;
                }
            }
            wn_r += step * step;
        }

        // dL/dp_H, then backward along the path that was actually walked.
        self.grad_p.iter_mut().for_each(|g| *g = 0.0);
        for v in 0..self.vocab {
            let g = self.probs[v] - if v == target { 1.0 } else { 0.0 };
            if g.abs() < 1e-12 {
                continue;
            }
            let row = &self.readout[rb + v * self.d..rb + (v + 1) * self.d];
            for (gp, w) in self.grad_p.iter_mut().zip(row) {
                *gp += g * w;
            }
            if sh_w {
                let base = &self.readout_shared[v * self.d..(v + 1) * self.d];
                for (gp, w) in self.grad_p.iter_mut().zip(base) {
                    *gp += g * w;
                }
            }
        }

        let mut wn_e = 0.0f64;
        let c = self.class_used;
        for h in (0..self.hops).rev() {
            let eid = self.path_edge[h];
            let nrm = self.path_norm[h].max(1e-12);
            // Through the unit-norm step: (I - p p^T)/||u||, with p the
            // normalised output of this hop.
            let mut out = vec![0.0; self.d];
            let recon: Vec<f64> = self.path_in[h]
                .iter()
                .zip(&self.path_pre[h])
                .map(|(x, a)| (x + a.tanh()) / nrm)
                .collect();
            let dot: f64 = recon.iter().zip(&self.grad_p).map(|(a, b)| a * b).sum();
            for ((o, g), r) in out.iter_mut().zip(&self.grad_p).zip(&recon) {
                *o = (g - r * dot) / nrm;
            }
            for ((ga, o), pre) in self
                .grad_a
                .iter_mut()
                .zip(&out)
                .zip(&self.path_pre[h])
            {
                let t = pre.tanh();
                *ga = o * (1.0 - t * t);
            }

            let base = self.slot(eid, c) * self.d * self.d;
            for r in 0..self.d {
                let ga = self.grad_a[r];
                if ga.abs() < 1e-15 {
                    continue;
                }
                let row = &mut self.edge_mem[base + r * self.d..base + (r + 1) * self.d];
                for (w, x) in row.iter_mut().zip(&self.path_in[h]) {
                    *w -= eta * ga * x;
                }
                wn_e += (eta * ga) * (eta * ga);
            }

            // dL/dp_{h-1} = residual path + through the memory.
            for (i, (gp, o)) in self.grad_p.iter_mut().zip(&out).enumerate() {
                let mut acc = *o;
                for (r, ga) in self.grad_a.iter().enumerate() {
                    acc += ga * self.edge_mem[base + r * self.d + i];
                }
                *gp = acc;
            }

            let slot = self.slot(eid, c);
            for u in self.usage.iter_mut() {
                *u *= 1.0 - self.usage_decay;
            }
            self.usage[slot] += self.usage_decay;
        }

        *self
            .domain_class
            .entry((self.cur_domain, self.class_used))
            .or_insert(0.0) += 1.0;
        let bucket = match self.visit_step as u64 {
            0..=99 => 0,
            100..=499 => 1,
            500..=1999 => 2,
            _ => 3,
        };
        *self
            .dc_bucket
            .entry((self.cur_domain, self.class_used, bucket))
            .or_insert(0.0) += 1.0;
        *self
            .dc_norm
            .entry((self.cur_domain, self.class_used))
            .or_insert(0.0) += wn_r.sqrt();
        self.visit_step += 1.0;
        if !self.readout_rungs.is_empty() {
            let c = self.class_used;
            self.relax_slice(c);
        }
        *self
            .write_by_domain
            .entry(self.cur_domain)
            .or_insert(0.0) += wn_r.sqrt() + wn_e.sqrt();
        self.write_norm_readout += wn_r.sqrt();
        self.write_norm_edge += wn_e.sqrt();
        self.train_home
            .insert((entity, relation), *self.path_edge.last().unwrap_or(&0));

        // Directed forgetting: a slice that has stopped earning traffic gives
        // its capacity back. What is lost is what stopped being used, not
        // whatever the newest write collided with.
        if self.forget > 0.0 {
            let share = 1.0 / self.usage.len() as f64;
            for s in 0..self.usage.len() {
                let deficit = (share - self.usage[s]).max(0.0) / share;
                if deficit <= 0.0 {
                    continue;
                }
                let keep = 1.0 - self.forget * deficit;
                self.edge_scale[s] *= keep;
                // The readout is where the knowledge actually is, so decaying
                // only the edge transform does not forget anything: driving
                // E toward zero leaves the payload passing through unchanged
                // and the readout still maps it. Measured that way, retiring
                // a domain and then decaying its slices made it *better*
                // (40.0% -> 59.6%), because the decay was reverting a drifted
                // transform toward the identity rather than erasing a fact.
                // Same sign at every capacity tested, so it was never a
                // capacity effect.
                if self.class_readout {
                    self.readout_scale[s / self.ring.n_edges()] *= keep;
                }
            }
        }
    }

    /// Write magnitude accumulated since the last call, then reset.
    /// Tells the memory which domain is being trained, for the class-purity
    /// diagnostic only. Nothing in the mechanism reads it.
    pub fn set_domain(&mut self, d: usize) {
        if d != self.cur_domain {
            self.visit_step = 0.0;
        }
        self.cur_domain = d;
    }

    /// Index of the highest logit from the last forward.
    pub fn last_argmax(&self) -> usize {
        let mut best = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &l) in self.logits.iter().enumerate() {
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        best
    }

    /// How many classes the domains share, as a fraction of the domains that
    /// have a class at all. 0 means every domain has its own class.
    pub fn class_collision(&self) -> f64 {
        let mut by_domain: std::collections::HashMap<usize, (usize, f64)> =
            std::collections::HashMap::new();
        for (&(d, c), &n) in &self.domain_class {
            let e = by_domain.entry(d).or_insert((c, 0.0));
            if n > e.1 {
                *e = (c, n);
            }
        }
        let mut seen: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (c, _) in by_domain.values() {
            *seen.entry(*c).or_insert(0) += 1;
        }
        let n_dom = by_domain.len().max(1) as f64;
        let collided: f64 = seen.values().filter(|v| **v > 1).map(|v| *v as f64).sum();
        collided / n_dom
    }

    pub fn take_write_norms(&mut self) -> (f64, f64) {
        let out = (self.write_norm_readout, self.write_norm_edge);
        self.write_norm_readout = 0.0;
        self.write_norm_edge = 0.0;
        out
    }

    pub fn note_consistency(&mut self, entity: usize, relation: usize) {
        if let Some(&home) = self.train_home.get(&(entity, relation)) {
            self.consistency_n += 1.0;
            if home == *self.path_edge.last().unwrap_or(&usize::MAX) {
                self.consistency_hits += 1.0;
            }
        }
    }

    pub fn routing_consistency(&self) -> f64 {
        if self.consistency_n > 0.0 {
            self.consistency_hits / self.consistency_n
        } else {
            f64::NAN
        }
    }

    /// Class switches per absorbed token. Should be ~0 inside a domain visit
    /// and non-zero across them; a class that never switches, or switches all
    /// the time, is not carrying the regime and the class axis is decoration.
    /// (intrusion rate by count, intrusion share of write magnitude,
    /// rate within each since-transition bucket).
    ///
    /// A domain's home class is the one it wrote to most; anything else is an
    /// intrusion. Post-hoc, so the mechanism is never told which class is whose.
    /// (live classes, distinct home classes, share of writes landing in a
    /// class that is home to more than one domain).
    ///
    /// The intrusion metric defines a domain's home as its own modal class, so
    /// when two domains share a home every one of their mutually destructive
    /// writes scores as legitimate. That is the failure this measures instead:
    /// collision, where the allocator never separated the domains at all.
    /// Rows = domains, columns = classes, entries = share of that domain's
    /// writes. The modal-class summary collapses this and would report one
    /// shared home even when domains split their mass across different
    /// classes, so the matrix is what should be read.
    pub fn domain_class_matrix(&self, domains: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; self.classes]; domains];
        for (&(d, c), &n) in &self.domain_class {
            if d < domains && c < self.classes {
                m[d][c] += n;
            }
        }
        for row in m.iter_mut() {
            let t: f64 = row.iter().sum();
            if t > 0.0 {
                for x in row.iter_mut() {
                    *x /= t;
                }
            }
        }
        m
    }

    pub fn collision_stats(&self) -> (usize, usize, f64) {
        let mut home: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut best: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        for (&(d, c), &n) in &self.domain_class {
            if n > *best.get(&d).unwrap_or(&-1.0) {
                best.insert(d, n);
                home.insert(d, c);
            }
        }
        let mut owners: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for c in home.values() {
            *owners.entry(*c).or_insert(0) += 1;
        }
        let distinct = owners.len();
        let (mut tot, mut shared) = (0.0, 0.0);
        for (&(_, c), &n) in &self.domain_class {
            tot += n;
            if owners.get(&c).copied().unwrap_or(0) > 1 {
                shared += n;
            }
        }
        (
            self.n_active,
            distinct,
            if tot > 0.0 { shared / tot } else { 0.0 },
        )
    }

    /// Growth events by observations since the domain last changed.
    pub fn growth_timing(&self) -> [usize; 4] {
        self.grow_at
    }

    pub fn write_for_domain(&self, d: usize) -> f64 {
        self.write_by_domain.get(&d).copied().unwrap_or(0.0)
    }

    pub fn intrusion_stats(&self) -> (f64, f64, [f64; 4]) {
        let mut home: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut best: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();
        for (&(d, c), &n) in &self.domain_class {
            if n > *best.get(&d).unwrap_or(&-1.0) {
                best.insert(d, n);
                home.insert(d, c);
            }
        }
        let (mut tot, mut bad) = (0.0, 0.0);
        let (mut wtot, mut wbad) = (0.0, 0.0);
        let mut bt = [0.0f64; 4];
        let mut bb = [0.0f64; 4];
        for (&(d, c), &n) in &self.domain_class {
            tot += n;
            if home.get(&d) != Some(&c) {
                bad += n;
            }
        }
        for (&(d, c), &w) in &self.dc_norm {
            wtot += w;
            if home.get(&d) != Some(&c) {
                wbad += w;
            }
        }
        for (&(d, c, b), &n) in &self.dc_bucket {
            bt[b] += n;
            if home.get(&d) != Some(&c) {
                bb[b] += n;
            }
        }
        let rate = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
        (
            rate(bad, tot),
            rate(wbad, wtot),
            [
                rate(bb[0], bt[0]),
                rate(bb[1], bt[1]),
                rate(bb[2], bt[2]),
                rate(bb[3], bt[3]),
            ],
        )
    }

    pub fn class_switch_rate(&self) -> f64 {
        if self.class_steps > 0.0 {
            self.class_switches / self.class_steps
        } else {
            f64::NAN
        }
    }

    pub fn edge_entropy(&self) -> (f64, f64) {
        let total: f64 = self.edge_visits.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let h = -self
            .edge_visits
            .iter()
            .map(|v| v / total)
            .filter(|p| *p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f64>();
        (h, (self.ring.n_edges() as f64).log2())
    }

    pub fn snapshot(&self) -> Vec<f64> {
        let mut v = self.ctx.clone();
        v.push(self.class_now as f64);
        v
    }

    pub fn restore(&mut self, snap: &[f64]) {
        let n = self.ctx.len();
        self.ctx.copy_from_slice(&snap[..n]);
        self.class_now = snap[n] as usize;
    }
}

impl Clone for EdgeMemory {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            vocab: self.vocab,
            d: self.d,
            hops: self.hops,
            classes: self.classes,
            keys: self.keys.clone(),
            emb: self.emb.clone(),
            edge_mem: self.edge_mem.clone(),
            readout: self.readout.clone(),
            class_readout: self.class_readout,
            readout_rungs: self.readout_rungs.clone(),
            lad_cap: self.lad_cap.clone(),
            lad_cond: self.lad_cond.clone(),
            ctx: self.ctx.clone(),
            ctx_rate: self.ctx_rate,
            proto: self.proto.clone(),
            n_active: self.n_active,
            sim_mean: self.sim_mean,
            sim_var: self.sim_var,
            sim_n: self.sim_n,
            grow_k: self.grow_k,
            ctx_history: self.ctx_history.clone(),
            domain_ctx_sample: self.domain_ctx_sample.clone(),
            growth_events: self.growth_events,
            growth_steals: self.growth_steals,
            growth_checked: self.growth_checked,
            hash_class: self.hash_class,
            usage: self.usage.clone(),
            gate: self.gate,
            clip: self.clip,
            expand_cap: self.expand_cap,
            share_edge: self.share_edge,
            readout_shared: self.readout_shared.clone(),
            grow_hold: self.grow_hold,
            novel_run: self.novel_run,
            grow_at: self.grow_at,
            margin_now: self.margin_now,
            readout_scale: self.readout_scale.clone(),
            edge_scale: self.edge_scale.clone(),
            usage_decay: self.usage_decay,
            forget: self.forget,
            path_edge: self.path_edge.clone(),
            path_in: self.path_in.clone(),
            path_pre: self.path_pre.clone(),
            path_norm: self.path_norm.clone(),
            payload: self.payload.clone(),
            route_payload: self.route_payload.clone(),
            logits: self.logits.clone(),
            probs: self.probs.clone(),
            grad_p: self.grad_p.clone(),
            grad_a: self.grad_a.clone(),
            class_now: self.class_now,
            class_used: self.class_used,
            train_home: self.train_home.clone(),
            consistency_hits: self.consistency_hits,
            consistency_n: self.consistency_n,
            write_norm_readout: self.write_norm_readout,
            write_norm_edge: self.write_norm_edge,
            class_switches: self.class_switches,
            class_steps: self.class_steps,
            last_class: self.last_class,
            edge_visits: self.edge_visits.clone(),
            domain_class: self.domain_class.clone(),
            cur_domain: self.cur_domain,
            dc_bucket: self.dc_bucket.clone(),
            dc_norm: self.dc_norm.clone(),
            write_by_domain: self.write_by_domain.clone(),
            visit_step: self.visit_step,
        }
    }
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn score(logits: &[f64], target: usize, out: &mut [f64]) -> (f64, bool) {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (v, (&l, p)) in logits.iter().zip(out.iter_mut()).enumerate() {
        let e = (l - peak).exp();
        *p = e;
        sum += e;
        if l > best_v {
            best_v = l;
            best = v;
        }
    }
    let inv = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv;
    }
    (-out[target].max(f64::MIN_POSITIVE).ln(), best == target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_fact_always_walks_the_same_path() {
        let mut rng = Rng::new(3);
        let mut m = EdgeMemory::new(16, 2, 3, 8, 32, 128, 0.01, 0.0, false, false, 8, 3.0, 20000.0, 1, 2.0, 0.1, 0.0, 0.0, false, false, 0, 1, &mut rng);
        m.forward(5, 9);
        let a = m.path_edge.clone();
        for t in 0..50 {
            m.absorb_token(t);
            m.observe_fact(t % 20, 21, t % 30, 0.3);
        }
        m.forward(5, 9);
        assert_eq!(
            a, m.path_edge,
            "routing must not move: it is fixed at construction"
        );
    }

    #[test]
    fn different_facts_take_different_paths() {
        let mut rng = Rng::new(5);
        let mut m = EdgeMemory::new(32, 3, 3, 8, 32, 256, 0.01, 0.0, false, false, 8, 3.0, 20000.0, 1, 2.0, 0.1, 0.0, 0.0, false, false, 0, 1, &mut rng);
        let mut seen = std::collections::HashSet::new();
        for t in 0..60 {
            m.forward(t, t + 1);
            seen.insert(m.path_edge.clone());
        }
        assert!(seen.len() > 1, "every fact took the same path");
    }

    #[test]
    fn hash_class_ignores_the_stream_entirely() {
        let mut rng = Rng::new(23);
        let mut m = EdgeMemory::new(16, 2, 3, 8, 32, 256, 0.2, 0.0, true, false, 8, 3.0, 20000.0, 1, 2.0, 0.1, 0.0, 0.0, false, false, 0, 1, &mut rng);
        m.forward(4, 7);
        let a = m.class_used;
        for _ in 0..300 {
            m.absorb_token(200);
        }
        m.forward(4, 7);
        assert_eq!(a, m.class_used, "a hashed class must not track context");
    }

    #[test]
    fn the_class_follows_the_stream() {
        let mut rng = Rng::new(7);
        let mut m = EdgeMemory::new(16, 2, 3, 8, 32, 256, 0.2, 0.0, false, false, 8, 3.0, 20000.0, 1, 2.0, 0.1, 0.0, 0.0, false, false, 0, 1, &mut rng);
        for _ in 0..200 {
            m.absorb_token(11);
        }
        let a = m.class_now;
        for _ in 0..200 {
            m.absorb_token(190);
        }
        assert_ne!(a, m.class_now, "the class must track what is passing by");
    }

    #[test]
    fn a_write_only_touches_the_edges_that_were_walked() {
        let mut rng = Rng::new(11);
        let mut m = EdgeMemory::new(16, 2, 3, 4, 16, 64, 0.01, 0.0, false, false, 8, 3.0, 20000.0, 1, 2.0, 0.1, 0.0, 0.0, false, false, 0, 1, &mut rng);
        // Warm the readout first. From a zero readout the first write leaves
        // it rank one along the payload, so dL/dp_H comes out exactly
        // parallel to p_H and the unit-norm projection (I - p p^T) cancels it
        // completely -- no gradient reaches any edge. It is a first-step
        // artefact that clears itself once the readout spans more than one
        // direction, but the isolation property worth asserting is the
        // steady-state one.
        for t in 0..8 {
            m.observe_fact(t, t + 1, t + 2, 0.3);
        }
        m.forward(2, 3);
        let walked: std::collections::HashSet<usize> = m.path_edge.iter().copied().collect();
        let before = m.edge_mem.clone();
        m.observe_fact(2, 3, 5, 0.5);
        // Class-major: an edge's classes are strided by n_edges, not adjacent.
        // The property asserted is unchanged -- only where the bytes live is.
        let blk = m.d * m.d;
        let ne = m.ring.n_edges();
        for e in 0..ne {
            let changed = (0..m.classes).any(|c| {
                let b = (c * ne + e) * blk;
                before[b..b + blk]
                    .iter()
                    .zip(&m.edge_mem[b..b + blk])
                    .any(|(x, y)| (x - y).abs() > 1e-12)
            });
            assert_eq!(changed, walked.contains(&e), "edge {e} changed={changed}");
        }
    }

    #[test]
    fn expanding_capacity_leaves_every_existing_weight_untouched() {
        let mut rng = Rng::new(9);
        let mut m = EdgeMemory::new(
            16, 2, 3, 4, 16, 64, 0.01, 0.0, false, true, 4, 3.0, 20000.0, 1, 2.0, 0.1, 0.0,
            0.0, false, false, 0, 1, &mut rng,
        );
        for t in 0..40 {
            m.observe_fact(t % 12, (t + 1) % 12, (t + 2) % 12, 0.3);
        }
        let logits_before = {
            m.forward(3, 4);
            m.logits.clone()
        };
        let ne = m.ring.n_edges();
        let blk = m.d * m.d;
        let old_classes = m.classes;
        let edge_before = m.edge_mem.clone();
        let read_before = m.readout.clone();

        m.expand_classes(4);

        // Class-major means slice (c, e) keeps its offset when classes grow.
        // Edge-major would have shifted every one of them.
        for c in 0..old_classes {
            for e in 0..ne {
                let b = (c * ne + e) * blk;
                assert_eq!(
                    &edge_before[b..b + blk],
                    &m.edge_mem[b..b + blk],
                    "edge slice (class {c}, edge {e}) moved or changed"
                );
            }
        }
        assert_eq!(
            &read_before[..],
            &m.readout[..old_classes * m.vocab * m.d],
            "an existing readout slice changed under expansion"
        );

        // And the behaviour it supports is identical, not merely similar.
        m.forward(3, 4);
        assert_eq!(
            logits_before, m.logits,
            "expansion perturbed the prediction it was supposed to leave alone"
        );
    }
}
