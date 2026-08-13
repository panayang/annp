//! The processing unit, and the routing decision it makes.
//!
//! A node is one ladder-backed matrix `W` plus the last payload it saw. `W`
//! learns the transition from that payload to the next one, so the node is an
//! online predictor of its own input stream, and causality is whatever arrival
//! order happens to be — a local partial order, not a global mask.
//!
//! Four things had to be settled to make the forward pass runnable at all.
//! None of them were in DESIGN.md §1.4/§1.5; all four are flagged there now.
//!
//! 1. **`f64` everywhere, not `f32` payloads.** The ladder provably needs
//!    `f64` (`linalg`), and a mixed-precision boundary in the hottest loop is
//!    exactly where a silent bug would live. Narrowing payloads later is a
//!    measured optimisation, not a default.
//!
//! 2. **The output is residual: `u = q + phi(W q)`.** Without it a fresh
//!    network annihilates everything — `W` starts at zero, so `phi(Wq) = 0`,
//!    every payload becomes the zero vector, and nothing can ever be learned.
//!    With it, a node that knows nothing passes the particle through unchanged
//!    and the network starts as a pure diffusion.
//!
//! 3. **Routing has no parameters at all.** DESIGN §1.5 route (b) gave each
//!    edge a key `K_e` trained by a surprise report sent back from the
//!    neighbour — which needs a return path and an extra field on the particle.
//!    It is unnecessary. Node `j` already computes `W_j q_j`, its prediction of
//!    the input it expects next. Score an edge by how well the payload matches
//!    that. Particles then flow to whoever is expecting them, which is the
//!    variational story route (b) was reaching for, with **zero routing
//!    parameters, no return path, and no new particle field**.
//!
//! 4. **Absorption is relative, not a fixed reference.** A particle stops where
//!    it fits best: it moves on only if some neighbour expects this content
//!    *better than the node it is already at*.
//!
//!    The original rule pinned absorption to a constant logit of zero, and that
//!    was wrong in a way three separate measurements found. An unvisited node
//!    publishes a zero expectation, so its edge scored zero on content — the
//!    same as a node that has learned and simply does not match. Ignorance and
//!    mismatch were indistinguishable, and homeostasis, which pushes losing
//!    edges up without bound, then tipped every such edge above the absorb
//!    reference. Symptoms: hop count climbing with `d_head` (7.4 to 10.4 as
//!    content scores shrink like `1/sqrt(d)` and bias dominates), absorption
//!    swinging from 0% to 25% under a zero-sum bias shift, and `top_k = 1`
//!    never halting at all — with no splitting, mass never decays, so the mass
//!    floor is inert and halting rests entirely on an absorb option that
//!    homeostasis had suppressed.
//!
//!    The absorbed share is computed from content alone. With the homeostatic
//!    bias gone there is nothing else that could have moved it, so the
//!    decoupling §10.1 ④ claimed holds trivially now rather than by argument.
//!
//! Routing reads **rung 1**, the same state the transform reads. Publishing the
//! deepest rung instead was tried on the theory that an expectation rebuilt
//! from a single recent input gives every particle a landscape that jitters
//! underneath it; it measured 0.3% *worse* (DESIGN.md §16). The ladder earns
//! its keep by restraining rung 1, not by offering a second readout — which
//! also means §1.4's reading of the deep rungs as a separately usable
//! "consolidated key-value memory" is not supported.
//!
//! Two mechanisms have been **removed** rather than kept and defaulted off,
//! because measurement said they earned nothing (DESIGN.md §15.2, §16).
//!
//! *Top-k truncation* was meant to bound compute and did the opposite: it
//! squeezed the absorb option out of the retained set, so mass never drained
//! and paths ran long. Branching is bounded by the mass floor alone. Under the
//! relative rules it changed the result by 0.7%, and the best configuration was
//! the one that truncated nothing.
//!
//! *Homeostatic edge bias* cost 0.58% and bought 0.7 points of load balance,
//! with neither setting anywhere near collapse — while being implicated in
//! every defect found: suppressed absorption, hop count drifting with `d_head`,
//! and `top_k = 1` never halting. Nothing now competes with the content score.

use crate::graph::Topology;
use crate::ladder::{AssocMemory, Schedule};
use crate::linalg::{dot, norm};

/// Everything that shapes a node. Grouped so the engine can be handed one
/// value and the manifest can record it verbatim.
#[derive(Clone, Copy, Debug)]
pub struct NodeParams {
    pub d_head: usize,
    pub absorb: AbsorbRule,
    /// Delta-rule step. At 1.0 with unit-norm keys a write is exact in one
    /// shot, which is the parameter-free choice and what E0 measured.
    pub eta: f64,
    pub schedule: Schedule,
    pub rungs: usize,
    /// Timescales in a node's context key. `1` means the key is the last input
    /// alone, which is exactly the original behaviour.
    pub context_scales: usize,
    /// Which ladder rung the forward pass reads. Zero is the fastest and is
    /// what the design has always used; see `AssocMemory::read_at`.
    pub read_rung: usize,
    /// Past keys each node remembers, for the key-echo diagnostic. Zero is off.
    ///
    /// A node's memory is addressed by its context key, so retrieval only pays
    /// when a key comes back. This measures whether it does: the cosine between
    /// the key a visit reads with and the closest key that node has read with
    /// before. Diagnostic only — nothing in the dynamics reads it.
    pub key_echo: usize,
}

/// How a node decides whether to forward a particle at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorbRule {
    /// Stop unless some neighbour expects this content better than the current
    /// node does. Homeostasis cannot influence the decision.
    Relative,
    /// `Relative`, with the node's own surprise subtracted from the reference:
    /// stop where you are understood, keep moving where you are not.
    ///
    /// A node that predicted this arrival well has nothing left to learn from
    /// it and the particle is home; a node that was surprised does not
    /// understand it, and the particle should keep looking. Surprise is already
    /// computed, is already the design's one credit signal, and shares units
    /// with the inner products (both come from unit vectors), so this costs no
    /// new constant.
    ///
    /// It also anneals by itself. An untrained node has surprise ~1, which
    /// drops the reference far enough that every neighbour is a candidate; as
    /// the network learns, surprise falls, the reference rises, and paths
    /// shorten. The exploration schedule is driven by how well the network
    /// understands its input rather than by a hand-set temperature.
    RelativeSurprise,
    /// Absorption competes at a constant logit of zero. Kept to reproduce the
    /// measurements that condemned it; do not use for new results.
    FixedReference,
}

/// What a node did with one arriving particle.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Emitted payload, unit norm.
    pub emitted: Vec<f64>,
    /// `||q - W q_prev||` before the write. Zero on a node's first ever visit,
    /// where there is no previous input to have predicted from.
    pub surprise: f64,
    /// Routing weights over `out_edges ++ [absorb]`, summing to 1.
    pub weights: Vec<f64>,
    /// What this node contributed, `tanh(W k)`, before it was added to the
    /// payload. The emitted payload is the sum of the two, and the sum is
    /// dominated by the payload — a unit-norm arrival against a contribution of
    /// magnitude about 0.4 — so a readout that averages emitted payloads is
    /// mostly averaging its own input. Kept separately so the two can be
    /// deposited apart and weighted by whatever is reading them.
    pub answer: Vec<f64>,
    /// `max(0, 1 - surprise)`: the share of the arriving payload this node's
    /// memory actually explained.
    ///
    /// Surprise is `||v - W k||` against a unit target, so zero means the memory
    /// predicted exactly, one means it predicted nothing, and above one means it
    /// pointed the wrong way. The clamp therefore silences a node that knows
    /// nothing rather than letting it vote negatively, which makes weighting by
    /// this quantity a soft version of picking the experts that know something.
    pub confidence: f64,
}

/// Everything a node reads during a tick and never writes.
///
/// Grouped because the alternative is threading five borrows through every
/// call, and because it makes the read-only half of a tick explicit: all of
/// this is a frozen snapshot, which is what lets nodes run in parallel.
pub struct Context<'a> {
    pub params: &'a NodeParams,
    pub out_edges: &'a [u32],
    /// Published expectations for the whole bank.
    pub expects: &'a [f64],
    /// This node's own slice of `expects`.
    pub self_expect: &'a [f64],
}

/// Reusable buffers for one `Node::step`. Held per worker so a tick can be
/// processed in parallel without any allocation in the hot loop.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    pred: Vec<f64>,
    key_before: Vec<f64>,
    logits: Vec<f64>,
    order: Vec<usize>,
}

/// One processing unit. Owns everything it touches during a tick, which is
/// what lets the engine hand out disjoint `&mut Node` across threads while
/// sharing the published expectations immutably.
#[derive(Clone, Debug)]
pub struct Node {
    memory: AssocMemory,
    /// Leaky integrators of the payloads this node has seen, with geometrically
    /// spaced time constants `tau_j = r^j`. Their normalised sum is the delta
    /// rule's key.
    ///
    /// This is what lets `W` map *recent history* to the next input rather than
    /// the last input alone, and it is the difference between representing
    /// `E[next | current]` and `E[next | history]`. Everything else the node
    /// does — routing, absorption, surprise — is unchanged.
    ///
    /// Not a ladder, deliberately. A ladder conserves its capacity-weighted
    /// total, so injecting a unit payload on every visit would accumulate
    /// without bound and rung 1 would converge to the historical mean of
    /// everything this node has ever seen — the very quantity §16.1 measured to
    /// be useless as a key. Each integrator here is a convex combination of
    /// unit vectors and so stays bounded, while the geometric spacing keeps the
    /// point of a ladder: no single characteristic timescale.
    ///
    /// `tau_0 = 1` reduces to `c = q`, so `context_scales == 1` is exactly the
    /// original last-input key rather than an approximation of it.
    context: Vec<Vec<f64>>,
    /// Normalised sum of `context`, cached because every visit needs it twice.
    key: Vec<f64>,
    /// Ring of past read keys, and where the next one goes.
    echo_past: Vec<Vec<f64>>,
    echo_at: usize,
    /// Sum and count of "closest previous key" cosines, and how many landed
    /// above a half — a key that similar has genuinely been seen before.
    echo_sum: f64,
    echo_count: u64,
    echo_hits: u64,
    /// What the node adds to a payload, and how wrong it was.
    ///
    /// A visit does `u = q + tanh(Wq)`: the memory's guess at what follows this
    /// context is added to the payload unconditionally, with nothing deciding
    /// whether the guess is worth trusting. `pred_sum` is the magnitude of that
    /// addition against a unit-norm payload, so it says how much is being mixed
    /// in; `surprise_sum` is the delta rule's own report of how far the guess
    /// was from what actually arrived. Large and wrong is the combination that
    /// would explain §34.5.
    ///
    /// Both are consumed by the run report. The accumulator this replaces was
    /// removed in §32.2 precisely because nothing read it.
    surprise_sum: f64,
    pred_sum: f64,
    visit_count: u64,
    has_fired: bool,
    /// Mass forwarded through each plastic out-edge since the last rewiring.
    /// Structural bookkeeping only — it plays no part in routing, which is a
    /// plain content softmax.
    plastic_usage: Vec<f64>,
    /// Visits since the last rewiring, and the surprise accumulated over them.
    turnover_visits: u64,
}

impl Node {
    /// Runs one arriving payload through this node.
    ///
    /// Order is learn-then-predict: first score how well the previous input
    /// predicted this one and write that correction, then emit from the
    /// updated matrix. `expects` is the tick's frozen snapshot and is never
    /// written here, so a tick's routing cannot depend on processing order.
    pub fn step(&mut self, ctx: &Context<'_>, q: &[f64], scratch: &mut Scratch) -> Outcome {
        let params = ctx.params;
        let degree = ctx.out_edges.len();
        let d = params.d_head;
        assert_eq!(q.len(), d, "payload width must match d_head");
        debug_assert!(
            (norm(q) - 1.0).abs() < 1e-9,
            "payloads are unit norm by construction"
        );
        scratch.pred.resize(d, 0.0);

        // Learn. On the very first visit there is no previous input, so there
        // is nothing that could have been predicted and nothing to correct.
        // The key *before* this payload is folded in: what the node had to
        // predict from. Reading back later uses the key *including* it.
        scratch.key_before.clear();
        scratch.key_before.extend_from_slice(&self.key);
        let surprise = if self.has_fired {
            self.memory.write(&scratch.key_before, q, params.eta)
        } else {
            self.has_fired = true;
            0.0
        };
        self.absorb_into_context(q);
        self.memory.relax();

        // Key echo, before the current key joins the ring so a key is never
        // compared against itself.
        if params.key_echo > 0 {
            let mut best = -1.0f64;
            for old in &self.echo_past {
                let dot: f64 = old.iter().zip(&self.key).map(|(a, b)| a * b).sum();
                if dot > best {
                    best = dot;
                }
            }
            if !self.echo_past.is_empty() {
                self.echo_sum += best;
                self.echo_count += 1;
                if best > 0.5 {
                    self.echo_hits += 1;
                }
            }
            if self.echo_past.len() < params.key_echo {
                self.echo_past.push(self.key.clone());
            } else {
                self.echo_past[self.echo_at].copy_from_slice(&self.key);
                self.echo_at = (self.echo_at + 1) % params.key_echo;
            }
        }

        // Predict, from the context including what just arrived. This is also
        // what neighbours route against once published.
        self.memory
            .read_at(params.read_rung)
            .mul_vec(&self.key, &mut scratch.pred);

        // Emit. Residual, so an untrained node is transparent rather than
        // annihilating. Renormalised because unit-norm payloads are the
        // invariant keeping every d_head-sized product in range.
        let mut emitted = vec![0.0; d];
        let mut added = 0.0;
        for (e, (&qi, &p)) in emitted.iter_mut().zip(q.iter().zip(&scratch.pred)) {
            let g = p.tanh();
            added += g * g;
            *e = qi + g;
        }
        self.surprise_sum += surprise;
        self.pred_sum += added.sqrt();
        self.visit_count += 1;
        let len = norm(&emitted);
        if len > 1e-12 {
            for e in emitted.iter_mut() {
                *e /= len;
            }
        } else {
            // Unreachable while the residual term is present, but a zero
            // payload would silently poison every downstream inner product.
            emitted.copy_from_slice(q);
        }

        let answer: Vec<f64> = scratch.pred.iter().map(|p| p.tanh()).collect();
        let confidence = (1.0 - surprise).max(0.0);
        let weights = self.route(ctx, &emitted, surprise, scratch);

        self.turnover_visits += 1;
        let lattice = degree - self.plastic_usage.len();
        for (used, w) in self.plastic_usage.iter_mut().zip(&weights[lattice..degree]) {
            *used += w;
        }

        Outcome {
            emitted,
            answer,
            confidence,
            surprise,
            weights,
        }
    }

    /// Routing weights over `out_edges ++ [absorb]`, summing to one.
    fn route(
        &mut self,
        ctx: &Context<'_>,
        emitted: &[f64],
        surprise: f64,
        scratch: &mut Scratch,
    ) -> Vec<f64> {
        let Context {
            params,
            out_edges,
            expects,
            self_expect,
        } = *ctx;
        let d = params.d_head;
        let absorb_slot = out_edges.len();
        let Scratch { logits, order, .. } = scratch;

        logits.clear();
        for &target in out_edges {
            let lo = target as usize * d;
            logits.push(dot(emitted, &expects[lo..lo + d]));
        }

        let mut weights = vec![0.0; absorb_slot + 1];
        let reference = match params.absorb {
            AbsorbRule::Relative => dot(emitted, self_expect),
            AbsorbRule::RelativeSurprise => dot(emitted, self_expect) - surprise,
            AbsorbRule::FixedReference => 0.0,
        };

        // Candidates are the neighbours that beat the reference. Under the
        // relative rule an empty candidate set means the particle is already
        // where it fits best, and it stops.
        order.clear();
        order.extend((0..absorb_slot).filter(|&j| match params.absorb {
            AbsorbRule::Relative | AbsorbRule::RelativeSurprise => logits[j] > reference,
            AbsorbRule::FixedReference => true,
        }));
        if order.is_empty() {
            weights[absorb_slot] = 1.0;
            return weights;
        }

        // Stage one: how much stays here, from content only. The homeostatic
        // bias is deliberately absent, which is what makes it structurally
        // unable to move the absorption rate.
        let peak = order.iter().map(|&j| logits[j]).fold(reference, f64::max);
        let absorbed = (reference - peak).exp();
        let mut total = absorbed;
        for &j in order.iter() {
            total += (logits[j] - peak).exp();
        }
        let p_absorb = absorbed / total;

        // Stage two: divide the forwarded remainder among the candidates.
        let peak = order
            .iter()
            .map(|&j| logits[j])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for &j in order.iter() {
            let w = (logits[j] - peak).exp();
            weights[j] = w;
            total += w;
        }
        let forwarded = 1.0 - p_absorb;
        for &j in order.iter() {
            weights[j] *= forwarded / total;
        }
        weights[absorb_slot] = p_absorb;
        weights
    }

    /// Folds a payload into every timescale and refreshes the cached key.
    fn absorb_into_context(&mut self, q: &[f64]) {
        let scales = self.context.len();
        for (state, tau) in self.context.iter_mut().zip(Self::timescales(scales)) {
            let blend = 1.0 / tau;
            for (c, &x) in state.iter_mut().zip(q) {
                *c += blend * (x - *c);
            }
        }
        self.key.fill(0.0);
        for state in &self.context {
            for (k, &c) in self.key.iter_mut().zip(state) {
                *k += c;
            }
        }
        let len = norm(&self.key);
        if len > 1e-12 {
            for k in self.key.iter_mut() {
                *k /= len;
            }
        } else {
            // Cancelled to nothing, which needs a usable direction rather than
            // a zero key; the newest payload is the best available.
            self.key.copy_from_slice(q);
        }
    }

    /// `tau_j = 2^j`. The ratio is fixed rather than taken from the ladder's
    /// `r`, because these are bounded integrators and not a diffusion — sharing
    /// a symbol would suggest a coupling that does not exist.
    fn timescales(count: usize) -> impl Iterator<Item = f64> {
        (0..count).map(|j| (1u64 << j) as f64)
    }

    /// This node's normalised prediction of the input it expects next, or
    /// all-zero if it has never fired.
    fn expectation_into(&self, params: &NodeParams, out: &mut [f64], scratch: &mut Scratch) {
        if !self.has_fired {
            out.fill(0.0);
            return;
        }
        scratch.pred.resize(params.d_head, 0.0);
        self.memory
            .read_at(params.read_rung)
            .mul_vec(&self.key, &mut scratch.pred);
        let len = norm(&scratch.pred);
        if len > 1e-12 {
            for (o, &p) in out.iter_mut().zip(&scratch.pred) {
                *o = p / len;
            }
        } else {
            // No usable prediction: stay neutral rather than pointing anywhere.
            out.fill(0.0);
        }
    }
}

/// All nodes, plus the tick-consistent snapshot they route against.
#[derive(Clone, Debug)]
pub struct NodeBank {
    params: NodeParams,
    nodes: Vec<Node>,
    /// `expects[i]` is node `i`'s normalised prediction of the input it will
    /// see next, or all-zero if it has never fired. Written once per tick,
    /// read-only while a tick is being processed, so every node routes against
    /// the same snapshot and the result cannot depend on processing order.
    expects: Vec<f64>,
    scratch: Scratch,
}

impl NodeBank {
    pub fn new(topology: &Topology, params: NodeParams) -> Self {
        assert!(params.d_head > 0, "d_head must be positive");
        assert!(params.eta > 0.0, "eta must be positive");
        assert!(
            params.context_scales >= 1,
            "a node needs at least one timescale"
        );
        let n = topology.ring().len();
        let d = params.d_head;
        let nodes = (0..n)
            .map(|i| Node {
                // One rung means no consolidation at all, matching what
                // `embed_rungs` already accepts. Without this the node ladder
                // cannot be ablated, which is how it went unmeasured.
                memory: if params.rungs <= 1 {
                    AssocMemory::single(d, 1.0)
                } else {
                    AssocMemory::ladder(d, params.schedule, params.rungs)
                },
                context: vec![vec![0.0; d]; params.context_scales.max(1)],
                key: vec![0.0; d],
                echo_past: Vec::new(),
                echo_at: 0,
                echo_sum: 0.0,
                echo_count: 0,
                echo_hits: 0,
                surprise_sum: 0.0,
                pred_sum: 0.0,
                visit_count: 0,
                has_fired: false,
                plastic_usage: vec![0.0; topology.plastic_slots(i as u32).len()],
                turnover_visits: 0,
            })
            .collect();
        Self {
            params,
            nodes,
            expects: vec![0.0; n * d],
            scratch: Scratch::default(),
        }
    }

    #[inline]
    pub fn params(&self) -> &NodeParams {
        &self.params
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The published expectation of node `i`, unit norm or all-zero.
    #[inline]
    pub fn expectation(&self, node: u32) -> &[f64] {
        let d = self.params.d_head;
        let lo = node as usize * d;
        &self.expects[lo..lo + d]
    }

    #[inline]
    pub fn has_fired(&self, node: u32) -> bool {
        self.nodes[node as usize].has_fired
    }

    /// Disjoint access to the nodes and the frozen expectation snapshot, so a
    /// tick can be processed across threads.
    #[inline]
    pub fn parts_mut(&mut self) -> (&mut [Node], &[f64], NodeParams) {
        (&mut self.nodes, &self.expects, self.params)
    }

    /// Single-threaded convenience wrapper over `Node::step`.
    pub fn process(&mut self, topology: &Topology, node: u32, q: &[f64]) -> Outcome {
        let params = self.params;
        let edges = topology.out_edges(node);
        let mut scratch = std::mem::take(&mut self.scratch);
        let d = params.d_head;
        let lo = node as usize * d;
        let self_expect = self.expects[lo..lo + d].to_vec();
        let ctx = Context {
            params: &params,
            out_edges: edges,
            expects: &self.expects,
            self_expect: &self_expect,
        };
        let out = self.nodes[node as usize].step(&ctx, q, &mut scratch);
        self.scratch = scratch;
        out
    }

    /// Mean surprise and mean magnitude of what a visit adds to the payload,
    /// against a payload of norm one. `None` before anything has fired.
    pub fn visit_stats(&self) -> Option<(f64, f64)> {
        let n: u64 = self.nodes.iter().map(|x| x.visit_count).sum();
        if n == 0 {
            return None;
        }
        let s: f64 = self.nodes.iter().map(|x| x.surprise_sum).sum();
        let p: f64 = self.nodes.iter().map(|x| x.pred_sum).sum();
        Some((s / n as f64, p / n as f64))
    }

    /// Mean "closest previous key" cosine across every visit that had a past to
    /// compare against, and the share of those above a half. `None` when the
    /// diagnostic is off.
    ///
    /// A node's memory is addressed by this key, so a low echo means retrieval
    /// has nothing to retrieve: the write went to a key that never comes back.
    pub fn key_echo(&self) -> Option<(f64, f64)> {
        let total: u64 = self.nodes.iter().map(|n| n.echo_count).sum();
        if total == 0 {
            return None;
        }
        let sum: f64 = self.nodes.iter().map(|n| n.echo_sum).sum();
        let hits: u64 = self.nodes.iter().map(|n| n.echo_hits).sum();
        Some((sum / total as f64, hits as f64 / total as f64))
    }

    /// Visits a node has taken since its last rewiring. `None` before it has
    /// fired at all.
    ///
    /// Cadence, not surprise: §18.1 settled that a node rewires every `d_head`
    /// visits rather than on a surprise threshold. A mean-surprise accumulator
    /// used to be returned alongside this and was discarded at the only call
    /// site, so it was costing an add per visit to compute a number nothing
    /// read.
    pub fn turnover_visits(&self, node: u32) -> Option<u64> {
        let n = &self.nodes[node as usize];
        (n.turnover_visits > 0).then_some(n.turnover_visits)
    }

    /// Plastic slot that carried the least mass since the last rewiring, as an
    /// index into the node's out-edges. `None` if the node has no plastic edges.
    pub fn coldest_plastic_slot(&self, node: u32, lattice_degree: usize) -> Option<usize> {
        let usage = &self.nodes[node as usize].plastic_usage;
        let (offset, _) = usage
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))?;
        Some(lattice_degree + offset)
    }

    /// Clears the turnover counters after a rewiring.
    pub fn reset_turnover(&mut self, node: u32) {
        let n = &mut self.nodes[node as usize];
        n.plastic_usage.fill(0.0);
        n.turnover_visits = 0;
    }

    /// Publishes node `i`'s current expectation for the next tick to route
    /// against. Called by the engine between ticks, never during one.
    pub fn publish(&mut self, node: u32) {
        let d = self.params.d_head;
        let lo = node as usize * d;
        let params = self.params;
        let mut scratch = std::mem::take(&mut self.scratch);
        let (nodes, expects) = (&self.nodes, &mut self.expects);
        nodes[node as usize].expectation_into(&params, &mut expects[lo..lo + d], &mut scratch);
        self.scratch = scratch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Ring, SmallWorld};
    use crate::rng::Rng;

    /// `nodes` is a ring length, not a torus side: a ring of `n` has `n`
    /// nodes where the old grid had `n^2`, so a fixture that used to be roomy
    /// at 5 has to say so explicitly now.
    fn fixture(nodes: usize) -> (Topology, NodeBank) {
        let mut rng = Rng::new(1);
        let t = Topology::small_world(Ring::new(nodes), SmallWorld::default(), &mut rng);
        let p = NodeParams {
            absorb: AbsorbRule::Relative,
            d_head: 16,
            eta: 1.0,
            schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
            rungs: 4,
            context_scales: 1,
            key_echo: 0,
            read_rung: 0,
        };
        let bank = NodeBank::new(&t, p);
        (t, bank)
    }

    fn unit(rng: &mut Rng, d: usize) -> Vec<f64> {
        let mut v = vec![0.0; d];
        rng.fill_unit_vector(&mut v);
        v
    }

    #[test]
    fn routing_weights_are_a_probability_distribution() {
        let (t, mut bank) = fixture(64);
        let mut rng = Rng::new(2);
        for node in 0..t.ring().len() as u32 {
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q);
            assert_eq!(out.weights.len(), t.degree(node) + 1);
            let total: f64 = out.weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "weights sum to {total}");
            assert!(out.weights.iter().all(|w| *w >= 0.0));
        }
    }

    #[test]
    fn surprise_reopens_an_untrained_network() {
        // The counterpart to `an_untrained_network_absorbs_immediately`: under
        // the plain relative rule nothing beats where the particle already is,
        // so it never moves. Subtracting an untrained node's surprise (~1)
        // drops the reference far enough that every neighbour qualifies, and
        // the annealing back to a short path is driven by learning rather than
        // by a schedule.
        let mut rng = Rng::new(31);
        let t = Topology::small_world(Ring::new(32), SmallWorld::default(), &mut rng);
        let mut bank = NodeBank::new(
            &t,
            NodeParams {
                absorb: AbsorbRule::RelativeSurprise,
                d_head: 16,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                context_scales: 1,
                key_echo: 0,
                read_rung: 0,
            },
        );
        // The very first arrival at a node has no predecessor, so surprise is
        // zero by definition and it behaves like the plain relative rule.
        let q = unit(&mut rng, 16);
        assert_eq!(bank.process(&t, 0, &q).weights[t.degree(0)], 1.0);
        // The second arrival is genuinely unpredicted, and now it travels.
        let q2 = unit(&mut rng, 16);
        let out = bank.process(&t, 0, &q2);
        assert!(
            out.weights[t.degree(0)] < 1.0,
            "surprise did not reopen any route"
        );
    }

    #[test]
    fn an_untrained_network_absorbs_immediately() {
        // Under the relative rule, nowhere expects anything at the start, so no
        // neighbour can beat where the particle already is. The network begins
        // fully transparent and grows its paths only as nodes learn — depth is
        // earned rather than spent at initialisation.
        let (t, mut bank) = fixture(64);
        let mut rng = Rng::new(21);
        for node in 0..t.ring().len() as u32 {
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q);
            assert_eq!(
                out.weights[t.degree(node)],
                1.0,
                "node {node} forwarded something"
            );
        }
    }

    #[test]
    fn a_particle_moves_on_once_a_neighbour_expects_it_better() {
        let (t, mut bank) = fixture(36);
        let mut rng = Rng::new(22);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));
        let student = t.out_edges(0)[2];
        for _ in 0..60 {
            bank.process(&t, student, &a);
            bank.process(&t, student, &b);
        }
        bank.process(&t, student, &a);
        bank.publish(student);

        let out = bank.process(&t, 0, &b);
        assert!(
            out.weights[2] > 0.0,
            "the edge to the node expecting this got nothing"
        );
        assert!(
            out.weights[t.degree(0)] < 1.0,
            "everything was absorbed anyway"
        );
    }

    #[test]
    fn one_timescale_is_exactly_the_last_input_key() {
        // tau_0 = 1 gives c <- q, so a single scale is the original behaviour
        // rather than an approximation of it. Anything else would mean the
        // ablation is not comparing like with like.
        let mut rng = Rng::new(61);
        let d = 16;
        let mut bank = {
            let t = Topology::small_world(Ring::new(32), SmallWorld::default(), &mut Rng::new(1));
            let p = NodeParams {
                absorb: AbsorbRule::RelativeSurprise,
                d_head: d,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                context_scales: 1,
                key_echo: 0,
                read_rung: 0,
            };
            (NodeBank::new(&t, p), t)
        };
        let (bank, t) = (&mut bank.0, bank.1);
        let inputs: Vec<Vec<f64>> = (0..8).map(|_| unit(&mut rng, d)).collect();
        for q in &inputs {
            bank.process(&t, 0, q);
        }
        // With one scale the key is the most recent payload. Exact in exact
        // arithmetic; the renormalisation divides by a norm that is 1 only to
        // within an ulp, so compare to that tolerance rather than bitwise.
        let last = inputs.last().unwrap();
        for (k, x) in bank.nodes[0].key.iter().zip(last) {
            assert!(
                (k - x).abs() < 1e-14,
                "key drifted from the last input: {k} vs {x}"
            );
        }
        bank.publish(0);
        assert!(bank.expectation(0).iter().any(|x| *x != 0.0));
    }

    #[test]
    fn more_timescales_make_the_key_depend_on_older_inputs() {
        // The whole point: with several scales the key after seeing (a, b)
        // differs from the key after seeing (c, b), so W can distinguish
        // histories that end the same way.
        let build = |scales: usize| {
            let t = Topology::small_world(Ring::new(32), SmallWorld::default(), &mut Rng::new(1));
            let p = NodeParams {
                absorb: AbsorbRule::RelativeSurprise,
                d_head: 16,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                context_scales: scales,
                key_echo: 0,
                read_rung: 0,
            };
            (NodeBank::new(&t, p), t)
        };
        let mut rng = Rng::new(62);
        let (a, b, c) = (unit(&mut rng, 16), unit(&mut rng, 16), unit(&mut rng, 16));

        for (scales, should_differ) in [(1usize, false), (4, true)] {
            let (mut bank1, t) = build(scales);
            bank1.process(&t, 0, &a);
            bank1.process(&t, 0, &b);
            let key_ab = bank1.nodes[0].key.clone();

            let (mut bank2, t2) = build(scales);
            bank2.process(&t2, 0, &c);
            bank2.process(&t2, 0, &b);
            let key_cb = bank2.nodes[0].key.clone();

            let gap: f64 = key_ab
                .iter()
                .zip(&key_cb)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
            if should_differ {
                assert!(gap > 0.05, "{scales} scales: keys differ by only {gap}");
            } else {
                assert!(gap < 1e-12, "one scale must ignore what came before: {gap}");
            }
        }
    }

    #[test]
    fn the_context_key_stays_bounded() {
        // The reason this is not a ladder. Each integrator is a convex
        // combination of unit vectors, so nothing accumulates; a conserving
        // ladder fed a unit payload every visit would grow without bound and
        // its key would converge to the historical mean.
        let t = Topology::small_world(Ring::new(32), SmallWorld::default(), &mut Rng::new(1));
        let p = NodeParams {
            absorb: AbsorbRule::RelativeSurprise,
            d_head: 16,
            eta: 1.0,
            schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
            rungs: 4,
            context_scales: 6,
            key_echo: 0,
            read_rung: 0,
        };
        let mut bank = NodeBank::new(&t, p);
        let mut rng = Rng::new(63);
        for _ in 0..20_000 {
            let q = unit(&mut rng, 16);
            bank.process(&t, 0, &q);
            for state in &bank.nodes[0].context {
                assert!(norm(state) <= 1.0 + 1e-9, "integrator grew past unit norm");
            }
            assert!(
                (norm(&bank.nodes[0].key) - 1.0).abs() < 1e-9,
                "key is not unit norm"
            );
        }
    }

    #[test]
    fn a_fresh_node_is_transparent() {
        // W starts at zero, so the residual must carry the payload through
        // unchanged. Without it the network annihilates its own input and can
        // never bootstrap.
        let (t, mut bank) = fixture(25);
        let mut rng = Rng::new(3);
        let q = unit(&mut rng, 16);
        let out = bank.process(&t, 0, &q);
        for (e, x) in out.emitted.iter().zip(&q) {
            assert!((e - x).abs() < 1e-12, "fresh node altered the payload");
        }
        assert_eq!(
            out.surprise, 0.0,
            "nothing was predicted, so nothing was missed"
        );
    }

    #[test]
    fn emitted_payloads_are_always_unit_norm() {
        let (t, mut bank) = fixture(36);
        let mut rng = Rng::new(4);
        for _ in 0..500 {
            let node = rng.next_below(t.ring().len() as u64) as u32;
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q);
            assert!((norm(&out.emitted) - 1.0).abs() < 1e-9, "norm drifted");
        }
    }

    #[test]
    fn a_node_learns_to_predict_a_repeated_transition() {
        // Feed a -> b over and over. Surprise must fall, and the published
        // expectation after seeing `a` must point at `b`.
        let (t, mut bank) = fixture(25);
        let mut rng = Rng::new(5);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));

        let mut first = None;
        let mut last = 0.0;
        for _ in 0..40 {
            bank.process(&t, 0, &a);
            let out = bank.process(&t, 0, &b);
            first.get_or_insert(out.surprise);
            last = out.surprise;
        }
        assert!(last < 0.5 * first.unwrap(), "{first:?} -> {last}");

        bank.process(&t, 0, &a);
        bank.publish(0);
        let alignment = dot(bank.expectation(0), &b);
        assert!(
            alignment > 0.8,
            "expectation points at {alignment} of the way to b"
        );
    }

    #[test]
    fn particles_route_towards_whoever_expects_them() {
        // Teach one neighbour to expect a payload, and routing mass must
        // concentrate on it without any edge parameter having been trained.
        let (t, mut bank) = fixture(36);
        let mut rng = Rng::new(6);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));

        let source = 0u32;
        let student = t.out_edges(source)[2];
        for _ in 0..60 {
            bank.process(&t, student, &a);
            bank.process(&t, student, &b);
        }
        bank.process(&t, student, &a);
        bank.publish(student);

        // Send the student's expected input through the source node. A fresh
        // source is transparent, so the emitted payload is still `b`.
        let out = bank.process(&t, source, &b);
        let best = out
            .weights
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            best, 2,
            "mass should go to the edge whose target expects this"
        );
    }

    #[test]
    fn unfired_nodes_publish_nothing_and_stay_neutral() {
        let (_, bank) = fixture(25);
        for node in 0..bank.len() as u32 {
            assert!(!bank.has_fired(node));
            assert!(bank.expectation(node).iter().all(|x| *x == 0.0));
        }
    }
}
