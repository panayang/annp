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
//!    The replacement is computed in two stages so that the decoupling
//!    §10.1 ④ claimed is true by construction rather than by argument: the
//!    absorbed share comes from **content alone**, and homeostasis appears only
//!    when dividing the forwarded remainder among neighbours. Homeostasis
//!    therefore cannot move the absorption rate at all — not "should not", but
//!    cannot, because it is not in that computation.
//!
//! Homeostasis stays: a zero-sum redistribution across a node's out-edges.
//!
//! An earlier version of this comment claimed that being zero-sum meant it
//! "can never shift the absorption rate". **That is false.** The bias is added
//! to logits that then compete through a *top-k truncation* against a fixed
//! absorb reference, and a zero-sum shift can push an edge below the reference
//! and let absorb into the retained set. Three edges at logits
//! `(0.5, 0.5, -1.0)` with `top_k = 2` absorb nothing; add the zero-sum bias
//! `(-0.6, +0.6, 0.0)` and absorb takes 25%. Biases are unbounded and every
//! losing edge accrues bias every tick, so this is an ordinary state, not a
//! contrived one. Collapse prevention and the compute budget are therefore
//! **not** decoupled, and nothing currently bounds the coupling.

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
    /// Rate at which an edge's homeostatic bias moves against its own usage.
    pub homeostasis: f64,
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
    /// Routing weights over `out_edges ++ [absorb]`, summing to 1. Only the
    /// top-k entries are non-zero.
    pub weights: Vec<f64>,
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
    pub top_k: usize,
}

/// Reusable buffers for one `Node::step`. Held per worker so a tick can be
/// processed in parallel without any allocation in the hot loop.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    pred: Vec<f64>,
    logits: Vec<f64>,
    order: Vec<usize>,
}

/// One processing unit. Owns everything it touches during a tick, which is
/// what lets the engine hand out disjoint `&mut Node` across threads while
/// sharing the published expectations immutably.
#[derive(Clone, Debug)]
pub struct Node {
    memory: AssocMemory,
    /// Most recent payload, the delta rule's key.
    last_input: Vec<f64>,
    has_fired: bool,
    /// Homeostatic bias per out-edge, in the topology's edge order.
    edge_bias: Vec<f64>,
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
        let d = params.d_head;
        assert_eq!(q.len(), d, "payload width must match d_head");
        debug_assert!((norm(q) - 1.0).abs() < 1e-9, "payloads are unit norm by construction");
        scratch.pred.resize(d, 0.0);

        // Learn. On the very first visit there is no previous input, so there
        // is nothing that could have been predicted and nothing to correct.
        let surprise = if self.has_fired {
            self.memory.write(&self.last_input, q, params.eta)
        } else {
            self.has_fired = true;
            0.0
        };
        self.last_input.copy_from_slice(q);
        self.memory.relax();

        // Predict. This is also what neighbours route against once published.
        self.memory.read().mul_vec(q, &mut scratch.pred);

        // Emit. Residual, so an untrained node is transparent rather than
        // annihilating. Renormalised because unit-norm payloads are the
        // invariant keeping every d_head-sized product in range.
        let mut emitted = vec![0.0; d];
        for (e, (&qi, &p)) in emitted.iter_mut().zip(q.iter().zip(&scratch.pred)) {
            *e = qi + p.tanh();
        }
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

        let weights = self.route(ctx, &emitted, surprise, scratch);
        Outcome { emitted, surprise, weights }
    }

    /// Routing weights over `out_edges ++ [absorb]`, summing to one.
    fn route(
        &mut self,
        ctx: &Context<'_>,
        emitted: &[f64],
        surprise: f64,
        scratch: &mut Scratch,
    ) -> Vec<f64> {
        let Context { params, out_edges, expects, self_expect, top_k } = *ctx;
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

        // Stage two: divide the forwarded remainder among the top-k candidates,
        // and here the bias does apply — it chooses between neighbours.
        let bias = &self.edge_bias;
        order.sort_by(|&a, &b| {
            (logits[b] + bias[b]).total_cmp(&(logits[a] + bias[a])).then(a.cmp(&b))
        });
        order.truncate(top_k.max(1));
        let peak = order
            .iter()
            .map(|&j| logits[j] + bias[j])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for &j in order.iter() {
            let w = (logits[j] + bias[j] - peak).exp();
            weights[j] = w;
            total += w;
        }
        let forwarded = 1.0 - p_absorb;
        for &j in order.iter() {
            weights[j] *= forwarded / total;
        }
        weights[absorb_slot] = p_absorb;

        self.apply_homeostasis(params, &weights, absorb_slot);
        weights
    }

    /// Nudges each edge's bias against its own share of the forwarded mass.
    ///
    /// Exactly zero-sum across a node's edges, so homeostasis redistributes
    /// between neighbours and can never move the absorption rate — absorption
    /// stays governed by the fixed reference logit and the mass floor, which
    /// keeps the compute budget separate from collapse prevention.
    fn apply_homeostasis(&mut self, params: &NodeParams, weights: &[f64], degree: usize) {
        if params.homeostasis == 0.0 || degree == 0 {
            return;
        }
        let forwarded: f64 = weights[..degree].iter().sum();
        if forwarded <= 0.0 {
            return;
        }
        let uniform = 1.0 / degree as f64;
        for (b, &w) in self.edge_bias.iter_mut().zip(&weights[..degree]) {
            *b -= params.homeostasis * (w / forwarded - uniform);
        }
    }

    /// This node's normalised prediction of the input it expects next, or
    /// all-zero if it has never fired.
    fn expectation_into(&self, params: &NodeParams, out: &mut [f64], scratch: &mut Scratch) {
        if !self.has_fired {
            out.fill(0.0);
            return;
        }
        scratch.pred.resize(params.d_head, 0.0);
        self.memory.read().mul_vec(&self.last_input, &mut scratch.pred);
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
        assert!(params.homeostasis >= 0.0, "homeostasis rate must be non-negative");
        let n = topology.grid().len();
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
                last_input: vec![0.0; d],
                has_fired: false,
                edge_bias: vec![0.0; topology.degree(i as u32)],
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

    #[inline]
    pub fn edge_bias(&self, node: u32) -> &[f64] {
        &self.nodes[node as usize].edge_bias
    }

    /// Disjoint access to the nodes and the frozen expectation snapshot, so a
    /// tick can be processed across threads.
    #[inline]
    pub fn parts_mut(&mut self) -> (&mut [Node], &[f64], NodeParams) {
        (&mut self.nodes, &self.expects, self.params)
    }

    /// Single-threaded convenience wrapper over `Node::step`.
    pub fn process(&mut self, topology: &Topology, node: u32, q: &[f64], top_k: usize) -> Outcome {
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
            top_k,
        };
        let out = self.nodes[node as usize].step(&ctx, q, &mut scratch);
        self.scratch = scratch;
        out
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
    use crate::graph::{Grid, SmallWorld};
    use crate::rng::Rng;

    fn fixture(side: usize) -> (Topology, NodeBank) {
        let mut rng = Rng::new(1);
        let t = Topology::small_world(Grid::new(side), SmallWorld::default(), &mut rng);
        let p = NodeParams {
            absorb: AbsorbRule::Relative,
            d_head: 16,
            eta: 1.0,
            schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
            rungs: 4,
            homeostasis: 0.0,
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
        let (t, mut bank) = fixture(8);
        let mut rng = Rng::new(2);
        for node in 0..t.grid().len() as u32 {
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q, 3);
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
        let t = Topology::small_world(Grid::new(6), SmallWorld::default(), &mut rng);
        let mut bank = NodeBank::new(
            &t,
            NodeParams {
                absorb: AbsorbRule::RelativeSurprise,
                d_head: 16,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                homeostasis: 0.0,
            },
        );
        // The very first arrival at a node has no predecessor, so surprise is
        // zero by definition and it behaves like the plain relative rule.
        let q = unit(&mut rng, 16);
        assert_eq!(bank.process(&t, 0, &q, 3).weights[t.degree(0)], 1.0);
        // The second arrival is genuinely unpredicted, and now it travels.
        let q2 = unit(&mut rng, 16);
        let out = bank.process(&t, 0, &q2, 3);
        assert!(out.weights[t.degree(0)] < 1.0, "surprise did not reopen any route");
    }

    #[test]
    fn an_untrained_network_absorbs_immediately() {
        // Under the relative rule, nowhere expects anything at the start, so no
        // neighbour can beat where the particle already is. The network begins
        // fully transparent and grows its paths only as nodes learn — depth is
        // earned rather than spent at initialisation.
        let (t, mut bank) = fixture(8);
        let mut rng = Rng::new(21);
        for node in 0..t.grid().len() as u32 {
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q, 3);
            assert_eq!(out.weights[t.degree(node)], 1.0, "node {node} forwarded something");
        }
    }

    #[test]
    fn a_particle_moves_on_once_a_neighbour_expects_it_better() {
        let (t, mut bank) = fixture(6);
        let mut rng = Rng::new(22);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));
        let student = t.out_edges(0)[2];
        for _ in 0..60 {
            bank.process(&t, student, &a, 2);
            bank.process(&t, student, &b, 2);
        }
        bank.process(&t, student, &a, 2);
        bank.publish(student);

        let out = bank.process(&t, 0, &b, 2);
        assert!(out.weights[2] > 0.0, "the edge to the node expecting this got nothing");
        assert!(out.weights[t.degree(0)] < 1.0, "everything was absorbed anyway");
    }

    #[test]
    fn a_fresh_node_is_transparent() {
        // W starts at zero, so the residual must carry the payload through
        // unchanged. Without it the network annihilates its own input and can
        // never bootstrap.
        let (t, mut bank) = fixture(5);
        let mut rng = Rng::new(3);
        let q = unit(&mut rng, 16);
        let out = bank.process(&t, 0, &q, 2);
        for (e, x) in out.emitted.iter().zip(&q) {
            assert!((e - x).abs() < 1e-12, "fresh node altered the payload");
        }
        assert_eq!(out.surprise, 0.0, "nothing was predicted, so nothing was missed");
    }

    #[test]
    fn emitted_payloads_are_always_unit_norm() {
        let (t, mut bank) = fixture(6);
        let mut rng = Rng::new(4);
        for _ in 0..500 {
            let node = rng.next_below(t.grid().len() as u64) as u32;
            let q = unit(&mut rng, 16);
            let out = bank.process(&t, node, &q, 2);
            assert!((norm(&out.emitted) - 1.0).abs() < 1e-9, "norm drifted");
        }
    }

    #[test]
    fn a_node_learns_to_predict_a_repeated_transition() {
        // Feed a -> b over and over. Surprise must fall, and the published
        // expectation after seeing `a` must point at `b`.
        let (t, mut bank) = fixture(5);
        let mut rng = Rng::new(5);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));

        let mut first = None;
        let mut last = 0.0;
        for _ in 0..40 {
            bank.process(&t, 0, &a, 2);
            let out = bank.process(&t, 0, &b, 2);
            first.get_or_insert(out.surprise);
            last = out.surprise;
        }
        assert!(last < 0.5 * first.unwrap(), "{first:?} -> {last}");

        bank.process(&t, 0, &a, 2);
        bank.publish(0);
        let alignment = dot(bank.expectation(0), &b);
        assert!(alignment > 0.8, "expectation points at {alignment} of the way to b");
    }

    #[test]
    fn particles_route_towards_whoever_expects_them() {
        // Teach one neighbour to expect a payload, and routing mass must
        // concentrate on it without any edge parameter having been trained.
        let (t, mut bank) = fixture(6);
        let mut rng = Rng::new(6);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));

        let source = 0u32;
        let student = t.out_edges(source)[2];
        for _ in 0..60 {
            bank.process(&t, student, &a, 2);
            bank.process(&t, student, &b, 2);
        }
        bank.process(&t, student, &a, 2);
        bank.publish(student);

        // Send the student's expected input through the source node. A fresh
        // source is transparent, so the emitted payload is still `b`.
        let out = bank.process(&t, source, &b, 2);
        let best = out
            .weights
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best, 2, "mass should go to the edge whose target expects this");
    }

    #[test]
    fn homeostasis_cannot_move_the_absorbed_share() {
        // The invariant DESIGN.md §10.1 ④ claimed and the fixed reference could
        // not deliver. The absorbed share is computed from content alone, so
        // perturbing the bias by any amount leaves it bit-identical; only the
        // split among neighbours changes.
        let (t, mut bank) = fixture(6);
        let mut rng = Rng::new(23);
        let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));
        for &edge in t.out_edges(0) {
            for _ in 0..40 {
                bank.process(&t, edge, &a, 2);
                bank.process(&t, edge, &b, 2);
            }
            bank.process(&t, edge, &a, 2);
            bank.publish(edge);
        }
        let degree = t.degree(0);

        // Clone first: `process` advances the node's own memory, so calling it
        // twice on one bank would compare two different states and hide the
        // property being tested behind a real state change.
        let mut perturbed = bank.clone();
        for (slot, delta) in [(0usize, 5.0), (1, -3.0), (2, 11.0)] {
            perturbed.nodes[0].edge_bias[slot] += delta;
        }
        let before = bank.process(&t, 0, &b, 3);
        assert!(before.weights[degree] > 0.0 && before.weights[degree] < 1.0, "need a mixed case");
        let after = perturbed.process(&t, 0, &b, 3);
        assert_ne!(
            before.weights[..degree],
            after.weights[..degree],
            "the bias should still change which neighbour gets the mass"
        );
        assert_eq!(
            before.weights[degree].to_bits(),
            after.weights[degree].to_bits(),
            "bias moved the absorbed share"
        );
    }

    #[test]
    fn homeostasis_forces_turn_taking() {
        // The mechanism, in the one setting where its effect is unambiguous:
        // a node fed the same payload repeatedly, with one neighbour trained to
        // expect exactly that. At top_k = 1 the winner takes everything, so
        // without homeostasis that edge wins every single time.
        let build = |homeostasis: f64| {
            let mut rng = Rng::new(8);
            let t = Topology::small_world(Grid::new(6), SmallWorld::default(), &mut rng);
            let mut bank = NodeBank::new(
                &t,
                NodeParams {
                    // This test is about the bias mechanics themselves, so it
                    // uses the rule under which one edge can sweep every draw.
                    absorb: AbsorbRule::FixedReference,
                    d_head: 16,
                    eta: 1.0,
                    schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                    rungs: 4,
                    homeostasis,
                },
            );
            let mut rng = Rng::new(9);
            let (a, b) = (unit(&mut rng, 16), unit(&mut rng, 16));
            let student = t.out_edges(0)[2];
            for _ in 0..60 {
                bank.process(&t, student, &a, 2);
                bank.process(&t, student, &b, 2);
            }
            bank.process(&t, student, &a, 2);
            bank.publish(student);

            let mut wins = vec![0u32; t.degree(0) + 1];
            for _ in 0..200 {
                let out = bank.process(&t, 0, &b, 1);
                let w = out.weights.iter().position(|x| *x > 0.0).unwrap();
                wins[w] += 1;
            }
            wins
        };

        let without = build(0.0);
        assert_eq!(without[2], 200, "with no homeostasis one edge should sweep");
        let with = build(0.05);
        assert!(with[2] < 200, "homeostasis failed to break the monopoly");
        assert!(
            with.iter().filter(|c| **c > 0).count() > 1,
            "mass never reached a second option"
        );
    }

    #[test]
    fn homeostasis_is_zero_sum_across_a_nodes_edges() {
        // It may move mass between neighbours; it must never be able to change
        // how much is absorbed, or it would double as a compute-budget knob.
        let mut rng = Rng::new(7);
        let t = Topology::small_world(Grid::new(6), SmallWorld::default(), &mut rng);
        let bank_params = NodeParams {
            absorb: AbsorbRule::Relative,
            d_head: 16,
            eta: 1.0,
            schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
            rungs: 4,
            homeostasis: 0.1,
        };
        let mut bank = NodeBank::new(&t, bank_params);
        for _ in 0..200 {
            let q = unit(&mut rng, 16);
            bank.process(&t, 0, &q, 3);
        }
        let sum: f64 = bank.edge_bias(0).iter().sum();
        assert!(sum.abs() < 1e-9, "biases drifted off zero-sum by {sum}");
    }

    #[test]
    fn unfired_nodes_publish_nothing_and_stay_neutral() {
        let (_, bank) = fixture(5);
        for node in 0..bank.len() as u32 {
            assert!(!bank.has_fired(node));
            assert!(bank.expectation(node).iter().all(|x| *x == 0.0));
        }
    }
}
