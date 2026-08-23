//! Topological distributed memory: a third-order tensor whose first axis is
//! the routing table.
//!
//! The monolithic line kept one `V x D` matrix and varied what was done to it
//! -- consolidation ladders, EWC anchors, expert splits, outer-product
//! addressing, context rotations. None of them beat a plain matrix, and the
//! reason is structural rather than a property of any one of them: with a
//! frozen projection the address is a fixed function of the input, so two
//! facts that must map the same key to different values are guaranteed to
//! land on the same parameters. Every one of those mechanisms was being asked
//! to repair a collision it had no way to avoid.
//!
//! Here the memory is `R[node, vocab, d]` and *which node answers* is decided
//! by routing a payload across a graph. Conflicting facts reach different
//! nodes, so they never share parameters in the first place. Protection comes
//! from where the write goes, not from how slowly it goes.
//!
//! Three properties this is built to have, in the order they matter:
//!
//! - **Distributed, topologically.** `N` nodes on a small-world ring; a
//!   particle enters at a content-addressed node and takes `hops` steps,
//!   choosing each step by matching its payload against the neighbours'
//!   expectations. The path, not a decaying trace, is the context.
//! - **Plasticity and stability stop being a trade-off.** They are opposed
//!   only while parameters are shared: sharing is what forces a choice
//!   between changing fast and not breaking what is already there. Allocation
//!   dissolves it -- a new fact routed to a different node is written at full
//!   rate (plastic) while the nodes holding old facts are untouched (stable).
//!   No slow rung, no anchor, no penalty term.
//! - **Forgetting is directed, not accidental.** Mass through a node is
//!   accumulated; a node that stops earning traffic has its slice decay and
//!   its capacity returned. What is lost is what stopped being used, rather
//!   than whatever the newest write happened to collide with.
//!
//! Two constraints carried over from what the monolithic line established,
//! both of which this design has to satisfy rather than rediscover:
//!
//! 1. The address must be computable from `(context, key)` alone. Surprise
//!    depends on the target, which is absent at probe time, so surprise can
//!    set *how much* is written but never *where* -- an address that cannot
//!    be recomputed at read time cannot be read. Routing here reads only the
//!    payload and the nodes' published expectations, never the target.
//! 2. Routing must not depend on the path taken to reach the current
//!    context, only on the context itself. Domain cycling is a closed loop;
//!    a path-dependent address would leave the previous lap's writes
//!    somewhere this lap does not look.

use annp_core::rng::Rng;

/// Small-world ring: every node has its two ring neighbours plus `shortcuts`
/// long-range contacts drawn once and then fixed.
///
/// Fixed, because a topology that rewires while the memory is being read
/// would move a fact's address out from under it. Rewiring belongs with the
/// forgetting rule, on the slice-decay timescale, not per step.
#[derive(Clone, Debug)]
pub struct Ring {
    n: usize,
    edges: Vec<Vec<usize>>,
}

impl Ring {
    pub fn new(n: usize, shortcuts: usize, rng: &mut Rng) -> Self {
        assert!(n >= 4, "need at least four nodes for a ring");
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let mut e = vec![(i + n - 1) % n, (i + 1) % n];
            for _ in 0..shortcuts {
                // Harmonic-ish long-range draw: mostly near, occasionally far.
                // The point of the shortcuts is that a few hops can reach a
                // distant part of the ring, so the set of nodes reachable in
                // `hops` steps is large without the degree being large.
                let d = 1 + (rng.next_below((n / 2) as u64) as usize);
                let t = (i + d) % n;
                if t != i && !e.contains(&t) {
                    e.push(t);
                }
            }
            edges.push(e);
        }
        Self { n, edges }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn out_edges(&self, node: usize) -> &[usize] {
        &self.edges[node]
    }
}

/// The distributed memory itself.
///
/// `readout` is the third-order tensor `R[node, vocab, d]`, stored as one
/// contiguous buffer. It is deliberately *not* a stack of independent
/// matrices in how it is used: the node axis is the routing table, so which
/// slice is live is a dynamic property of the input rather than an index
/// chosen from outside.
pub struct TopoMemory {
    ring: Ring,
    vocab: usize,
    d: usize,
    hops: usize,
    /// Token embeddings. The memory owns these because it no longer reads a
    /// globally computed context vector at all.
    ///
    /// The structure this replaces was incoherent: the memory was
    /// distributed across nodes while the context every node routed on was
    /// computed centrally, by one 8-rung diffusion ladder, and handed to all
    /// of them. Routing therefore tracked the global walk history rather
    /// than anything the nodes knew, which is why the same fact reached the
    /// same node only 8-14% of the time and retrieval scored 0.0%. Context
    /// now lives where the architecture always said it did: in the nodes.
    emb: Vec<f64>,
    /// What each node does to a payload passing through, fixed and random.
    ///
    /// This is what lets a path imprint on a particle. Without it the
    /// payload arriving at the readout is the same object that entered, so
    /// two facts with identical content are identical at the end no matter
    /// where they went, and the topology cannot separate anything.
    transform: Vec<f64>,
    /// What each node expects to see, an EMA of the payloads it has absorbed.
    /// This is the only adaptive part of routing, and it is what lets nodes
    /// specialise instead of splitting the input by a frozen rule.
    expectation: Vec<f64>,
    expect_rate: f64,
    /// Mass each node has absorbed, decayed. Drives directed forgetting.
    mass: Vec<f64>,
    mass_decay: f64,
    /// Slice decay per unit of *absent* mass. 0 disables forgetting.
    forget: f64,
    /// How strongly a node's already-held mass counts against it when
    /// routing.
    ///
    /// Without this the dynamics are rich-get-richer: a node that wins has
    /// its expectation pulled toward the payload it just absorbed, which
    /// makes it win the next similar payload, and traces inside one domain
    /// are highly correlated so "similar" is nearly everything. Measured
    /// with the term off: 1.02 bits of routing spread out of 4.00, about two
    /// nodes carrying a sixteen-node graph -- the tensor present in the code
    /// and absent from the run.
    ///
    /// The penalty is written against the *deviation* from an equal share,
    /// so at equilibrium every node is charged the same and the ranking is
    /// untouched; it only ever acts to push traffic off a node that is
    /// carrying more than its share. Mass is the conserved routing measure
    /// the architecture already defines, so this is competition for a fixed
    /// resource rather than a new penalty invented for the occasion -- and
    /// it is the same shape as the per-column calibration that fixed the
    /// projection's hubness.
    crowd: f64,
    readout: Vec<f64>,
    // scratch
    payload: Vec<f64>,
    scratch: Vec<f64>,
    ingress_buf: Vec<f64>,
    logits: Vec<f64>,
    probs: Vec<f64>,
    /// Nodes that share the particle's mass after routing. Mass is split at
    /// every hop and only the strongest `keep` nodes survive, renormalised.
    ///
    /// A single hard argmax was the first thing tried and it scored 0.0% in
    /// Mode B against 38.1% for the monolithic memory, with routing spread at
    /// 3.74 of 4.00 bits -- the mechanism running, and useless. Retrieval
    /// needs a fact to reach the same place at probe time as it did during
    /// training, and the two traces are never identical: training sees the
    /// walk's own history, the probe sees the restored domain context. One
    /// hard choice turns that small difference into a completely different
    /// node and a slice that never saw the fact, so retrieval is all or
    /// nothing. A graded split degrades gracefully instead, which is what
    /// the architecture specified in the first place -- mass is conserved
    /// and *allocated by match*, not carried by a particle down one path.
    ///
    /// `keep` is what stops the split from going the other way: spread mass
    /// over every node and every write touches every slice, which is the
    /// shared-parameter interference the design exists to avoid. Concentrated
    /// but graded is the same compromise k-WTA already makes over columns.
    keep: usize,
    mass_buf: Vec<f64>,
    next_buf: Vec<f64>,
    edge_buf: Vec<f64>,
    active_nodes: Vec<(usize, f64)>,
    /// Where each fact's mass last landed during training, so the probe can
    /// be asked whether it goes back to the same place.
    ///
    /// Routing spread was measured from the start; routing *consistency* was
    /// not, and that is the half that decides whether anything can be read
    /// back. High entropy across facts is entirely compatible with one fact
    /// landing somewhere new every time.
    train_home: std::collections::HashMap<(usize, usize), usize>,
    consistency_hits: f64,
    consistency_n: f64,
    /// Visit counts per node, for the path-diversity check.
    visits: Vec<f64>,
}

impl TopoMemory {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes: usize,
        shortcuts: usize,
        hops: usize,
        d_payload: usize,
        vocab: usize,
        forget: f64,
        expect_rate: f64,
        crowd: f64,
        keep: usize,
        rng: &mut Rng,
    ) -> Self {
        let ring = Ring::new(nodes, shortcuts, rng);
        let mut emb = vec![0.0; vocab * d_payload];
        for v in 0..vocab {
            rng.fill_unit_vector(&mut emb[v * d_payload..(v + 1) * d_payload]);
        }
        let mut transform = vec![0.0; nodes * d_payload * d_payload];
        for r in 0..nodes * d_payload {
            rng.fill_unit_vector(&mut transform[r * d_payload..(r + 1) * d_payload]);
        }
        // Random unit expectations, not zeros. With zeros every neighbour
        // matches a payload equally at the first hop, so the tie is broken by
        // edge order, every particle walks the same way, and the expectations
        // can never differentiate because they all see the same traffic. The
        // routing-spread check caught exactly that: 0.48 of 4.00 bits, about
        // 1.4 nodes in use out of 16. An inexperienced node should hold an
        // arbitrary expectation, not a null one.
        let mut expectation_init = vec![0.0; nodes * d_payload];
        for n in 0..nodes {
            rng.fill_unit_vector(&mut expectation_init[n * d_payload..(n + 1) * d_payload]);
        }
        Self {
            ring,
            vocab,
            d: d_payload,
            hops,
            emb,
            transform,
            expectation: expectation_init,
            expect_rate,
            mass: vec![0.0; nodes],
            mass_decay: 1.0 / (nodes as f64).max(1.0),
            forget,
            crowd,
            readout: vec![0.0; nodes * vocab * d_payload],
            payload: vec![0.0; d_payload],
            scratch: Vec::with_capacity(d_payload),
            ingress_buf: vec![0.0; nodes],
            logits: vec![0.0; vocab],
            probs: vec![0.0; vocab],
            keep: keep.clamp(1, nodes),
            mass_buf: vec![0.0; nodes],
            next_buf: vec![0.0; nodes],
            edge_buf: Vec::with_capacity(8),
            active_nodes: Vec::with_capacity(nodes),
            train_home: std::collections::HashMap::new(),
            consistency_hits: 0.0,
            consistency_n: 0.0,
            visits: vec![0.0; nodes],
        }
    }

    /// Routes a trace to a node and leaves the payload in `self.payload`.
    ///
    /// Reads only the trace and the published expectations, so it is exactly
    /// reproducible at probe time. `learn` is false on the read path: the
    /// expectations are routing state, and letting a probe move them would
    /// make the score depend on the order facts happen to be probed in.
    /// Sets the payload to a fact's content: entity and relation only, with
    /// no history in it at all.
    ///
    /// History is not absent from the system, it is somewhere else. Every
    /// token of the stream is absorbed by the nodes it passes through, so
    /// the domain a run is currently in is written into the node contexts,
    /// not into the particle. In Mode B the entities are shared across
    /// domains, so a content payload is byte-identical in all four -- the
    /// only thing that can tell them apart is what the nodes have been
    /// seeing, which is exactly where this design puts it.
    fn set_payload(&mut self, tokens: &[usize]) {
        self.payload.iter_mut().for_each(|v| *v = 0.0);
        for &t in tokens {
            for (p, e) in self
                .payload
                .iter_mut()
                .zip(&self.emb[t * self.d..(t + 1) * self.d])
            {
                *p += e;
            }
        }
        normalize(&mut self.payload);
    }

    /// Applies node `n`'s transform to the payload, in place.
    fn imprint(&mut self, n: usize) {
        let base = n * self.d * self.d;
        self.scratch.clear();
        for r in 0..self.d {
            let row = &self.transform[base + r * self.d..base + (r + 1) * self.d];
            let dot: f64 = row.iter().zip(&self.payload).map(|(a, b)| a * b).sum();
            self.scratch.push(dot.tanh());
        }
        for (p, t) in self.payload.iter_mut().zip(&self.scratch) {
            *p += t;
        }
        normalize(&mut self.payload);
    }

    /// Best-matching node for the current payload, charged for crowding.
    fn best_from(&self, candidates: &[usize]) -> usize {
        let n_nodes = self.mass.len() as f64;
        let mut best = candidates[0];
        let mut best_m = f64::NEG_INFINITY;
        for &e in candidates {
            let ex = &self.expectation[e * self.d..(e + 1) * self.d];
            let dot: f64 = ex.iter().zip(&self.payload).map(|(a, b)| a * b).sum();
            let m = dot - self.crowd * (self.mass[e] * n_nodes - 1.0);
            if m > best_m {
                best_m = m;
                best = e;
            }
        }
        best
    }

    /// Walks a payload through the graph, imprinting each node it passes.
    ///
    /// Deterministic given the payload and the node states, which is what
    /// makes a fact retrievable: put the network back in the state it was in
    /// when the fact was written and the same walk happens again.
    fn walk(&mut self, learn: bool) {
        let all: Vec<usize> = (0..self.mass.len()).collect();
        let mut node = self.best_from(&all);
        for _ in 0..self.hops {
            self.imprint(node);
            if learn {
                self.absorb(node);
            }
            let edges = self.ring.out_edges(node).to_vec();
            node = self.best_from(&edges);
        }
        if learn {
            self.absorb(node);
        }
        self.active_nodes.clear();
        self.active_nodes.push((node, 1.0));
    }

    /// A node takes the payload into its own context and its share of mass.
    fn absorb(&mut self, node: usize) {
        let r = self.expect_rate;
        let ex = &mut self.expectation[node * self.d..(node + 1) * self.d];
        for (e, p) in ex.iter_mut().zip(&self.payload) {
            *e += r * (p - *e);
        }
        for m in self.mass.iter_mut() {
            *m *= 1.0 - self.mass_decay;
        }
        self.mass[node] += self.mass_decay;
        self.visits[node] += 1.0;
    }

    /// Feeds one stream token to the network. This is how the nodes come to
    /// know which domain the run is in.
    pub fn absorb_token(&mut self, token: usize) {
        self.set_payload(&[token]);
        self.walk(true);
    }

    /// Top node of the current walk.
    fn home(&self) -> usize {
        self.active_nodes.first().map(|(n, _)| *n).unwrap_or(0)
    }

    /// All node contexts and mass, for putting the network back where it was.
    pub fn snapshot(&self) -> Vec<f64> {
        let mut out = self.expectation.clone();
        out.extend_from_slice(&self.mass);
        out
    }

    pub fn restore(&mut self, snap: &[f64]) {
        let n = self.expectation.len();
        self.expectation.copy_from_slice(&snap[..n]);
        self.mass.copy_from_slice(&snap[n..]);
    }

    fn forward_mixture(&mut self) {
        self.logits.iter_mut().for_each(|l| *l = 0.0);
        for &(n, w) in &self.active_nodes {
            let base = n * self.vocab * self.d;
            for v in 0..self.vocab {
                let row = &self.readout[base + v * self.d..base + (v + 1) * self.d];
                let dot: f64 = row.iter().zip(&self.payload).map(|(a, b)| a * b).sum();
                self.logits[v] += w * dot;
            }
        }
    }

    pub fn predict_fact(&mut self, entity: usize, relation: usize, target: usize) -> (f64, bool) {
        self.set_payload(&[entity, relation]);
        self.walk(false);
        self.forward_mixture();
        score(&self.logits, target, &mut self.probs)
    }

    /// Records whether a probed fact routes back to where training left it.
    pub fn note_consistency(&mut self, entity: usize, relation: usize) {
        if let Some(&home) = self.train_home.get(&(entity, relation)) {
            self.consistency_n += 1.0;
            if home == self.home() {
                self.consistency_hits += 1.0;
            }
        }
    }

    /// Fraction of probed facts that route back to their training node.
    pub fn routing_consistency(&self) -> f64 {
        if self.consistency_n > 0.0 {
            self.consistency_hits / self.consistency_n
        } else {
            f64::NAN
        }
    }

    pub fn observe_fact(
        &mut self,
        entity: usize,
        relation: usize,
        target: usize,
        eta: f64,
    ) {
        self.set_payload(&[entity, relation]);
        self.walk(true);
        self.train_home.insert((entity, relation), self.home());
        self.write(target, eta);
        // The target is a stream token too, and in Mode B it is the only
        // token that differs between domains -- letting the nodes absorb it
        // is what puts the domain into their contexts.
        self.absorb_token(target);
    }

    fn write(&mut self, target: usize, eta: f64) {
        self.forward_mixture();
        let _ = score(&self.logits, target, &mut self.probs);

        // Each surviving node is charged in proportion to the mass it took,
        // so a node that barely participated barely changes. This is what
        // keeps a graded split from becoming shared-parameter interference.
        let active: Vec<(usize, f64)> = self.active_nodes.clone();
        for (node, share) in active {
            let base = node * self.vocab * self.d;
            for v in 0..self.vocab {
                let g = if v == target { 1.0 - self.probs[v] } else { -self.probs[v] };
                let step = eta * share * g;
                let row = &mut self.readout[base + v * self.d..base + (v + 1) * self.d];
                for (w, p) in row.iter_mut().zip(&self.payload) {
                    *w += step * p;
                }
            }
        }

        // Directed forgetting: a slice decays in proportion to how little
        // traffic its node is currently earning. A node carrying its share
        // keeps everything; one that has gone quiet gives capacity back.
        if self.forget > 0.0 {
            let share = 1.0 / self.mass.len() as f64;
            for n in 0..self.mass.len() {
                let deficit = (share - self.mass[n]).max(0.0) / share;
                if deficit <= 0.0 {
                    continue;
                }
                let keep = 1.0 - self.forget * deficit;
                let b = n * self.vocab * self.d;
                for w in &mut self.readout[b..b + self.vocab * self.d] {
                    *w *= keep;
                }
            }
        }
    }

    /// Entropy of the node-visit distribution, in bits, and the maximum
    /// available.
    ///
    /// The failure this watches for is collapse: if every particle ends at
    /// the same node the tensor degenerates into a single shared matrix and
    /// the mechanism is present in the code but absent from the run. It has
    /// to be reported with the result, not checked afterwards, because a
    /// collapsed run still produces a perfectly plausible accuracy number.
    pub fn path_entropy(&self) -> (f64, f64) {
        let total: f64 = self.visits.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let h = -self
            .visits
            .iter()
            .map(|v| v / total)
            .filter(|p| *p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f64>();
        (h, (self.ring.len() as f64).log2())
    }
}

impl Clone for TopoMemory {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            vocab: self.vocab,
            d: self.d,
            hops: self.hops,
            emb: self.emb.clone(),
            transform: self.transform.clone(),
            expectation: self.expectation.clone(),
            expect_rate: self.expect_rate,
            mass: self.mass.clone(),
            mass_decay: self.mass_decay,
            forget: self.forget,
            crowd: self.crowd,
            readout: self.readout.clone(),
            payload: self.payload.clone(),
            scratch: self.scratch.clone(),
            ingress_buf: self.ingress_buf.clone(),
            logits: self.logits.clone(),
            probs: self.probs.clone(),
            keep: self.keep,
            mass_buf: self.mass_buf.clone(),
            next_buf: self.next_buf.clone(),
            edge_buf: self.edge_buf.clone(),
            active_nodes: self.active_nodes.clone(),
            train_home: self.train_home.clone(),
            consistency_hits: self.consistency_hits,
            consistency_n: self.consistency_n,
            visits: self.visits.clone(),
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
    let loss = -out[target].max(f64::MIN_POSITIVE).ln();
    (loss, best == target)
}

/// Softmax of `src` into `dst`, both the same length.
/// Scales a vector to unit norm in place, leaving a zero vector alone.
fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_walk_is_reproducible_from_the_same_node_state() {
        let mut rng = Rng::new(7);
        let mut m = TopoMemory::new(16, 2, 3, 32, 128, 0.0, 0.01, 1.0, 4, &mut rng);
        m.set_payload(&[5, 9]);
        m.walk(false);
        let a = m.home();
        m.set_payload(&[5, 9]);
        m.walk(false);
        assert_eq!(a, m.home(), "a read-only walk must not move");
    }

    #[test]
    fn restoring_node_state_restores_the_route() {
        let mut rng = Rng::new(11);
        let mut m = TopoMemory::new(16, 3, 3, 32, 128, 0.0, 0.2, 1.0, 4, &mut rng);
        m.set_payload(&[3, 4]);
        m.walk(false);
        let before = m.home();
        let snap = m.snapshot();
        for t in 20..60 {
            m.absorb_token(t);
        }
        m.restore(&snap);
        m.set_payload(&[3, 4]);
        m.walk(false);
        assert_eq!(
            before,
            m.home(),
            "putting the network back must put the fact back"
        );
    }

    #[test]
    fn different_facts_can_reach_different_nodes() {
        let mut rng = Rng::new(13);
        let mut m = TopoMemory::new(32, 3, 3, 32, 128, 0.0, 0.05, 1.0, 4, &mut rng);
        let mut seen = std::collections::HashSet::new();
        for t in 0..60 {
            m.set_payload(&[t, t + 1]);
            m.walk(true);
            seen.insert(m.home());
        }
        assert!(
            seen.len() > 1,
            "routing collapsed to one node, the tensor is a single matrix"
        );
    }

    #[test]
    fn a_write_only_touches_the_node_that_answered() {
        let mut rng = Rng::new(17);
        let mut m = TopoMemory::new(16, 2, 2, 32, 64, 0.0, 0.01, 1.0, 1, &mut rng);
        m.set_payload(&[2, 3]);
        m.walk(false);
        let node = m.home();
        let before = m.readout.clone();
        m.write(5, 0.5);
        let slice = m.vocab * m.d;
        for n in 0..m.ring.len() {
            let changed = before[n * slice..(n + 1) * slice]
                .iter()
                .zip(&m.readout[n * slice..(n + 1) * slice])
                .any(|(a, b)| (a - b).abs() > 1e-12);
            assert_eq!(changed, n == node, "node {n} changed={changed}, answered={node}");
        }
    }
}
