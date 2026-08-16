//! A growing tree of nodes, where a node is a nonlinear local write on a
//! consolidation ladder.
//!
//! `DESIGN-TREE.md` is the specification. The parts that matter here:
//!
//! **Routing is causal.** Surprise against the target would need the target,
//! and the protocol is predict, charge, then write, so routing on it leaks the
//! answer. Every node carries a prototype — the running mean of what it has
//! absorbed — and routing is the mismatch between the input and that prototype.
//! Only the write uses the target, and by then it is legitimately available.
//!
//! **Rolling is the ladder, not a window.** Each node's typical surprise sits
//! on its own one-element ladder, so "recent versus long-run" is the ladder's
//! own timescales rather than a smoothing constant nobody can justify.
//!
//! **Growth never stops, the shape is bounded instead.** A stopping rule would
//! carve out a period in which the architecture is still being decided, and
//! that period is a training phase. Instead every path ends at `depth_max` and
//! every node holds at most `n` children, so the node count is pinned at
//! `(n^(depth_max+1) - 1) / (n - 1)` with no second parameter, and compute per
//! token is exactly `depth_max` node evaluations however large the tree grows.
//!
//! **Every level predicts and the path mixes.** Leaf-only would put the tail of
//! a Zipf source where the fewest samples are; mixing along the path is what
//! backoff exists for. Weights come from each node's rolling prediction cost,
//! so they cost no parameter either.

use crate::ladder::{AssocMemory, Ladder, Schedule};

/// Ladder-weighted running mean: two chains fed in lockstep, read as a ratio.
/// A fresh node has seen nothing, and a threshold of zero means everything
/// looks novel to it, which is the right default for something with no history.
fn mean(value: &Ladder, count: &Ladder) -> f64 {
    let n = count.rung(0).as_slice()[0];
    if n <= f64::MIN_POSITIVE {
        return 0.0;
    }
    value.rung(0).as_slice()[0] / n
}

struct Node {
    /// `vocab x d`, the predictor. Rung 1 is live.
    mem: AssocMemory,
    /// `d x 1`, running mean of absorbed inputs. Routing compares against this.
    proto: Ladder,
    /// `1 x 1` pairs, value over count. The ladder is a *closed* chain: it
    /// conserves the capacity-weighted total, so injecting a number every step
    /// integrates it and never averages it. Reading value/count instead gives a
    /// genuine running mean while keeping the ladder's timescales, and still
    /// introduces no smoothing constant.
    ///
    /// Measured with the integrating version: the threshold outgrew the bounded
    /// surprise within a few hundred tokens, novelty never fired again, and the
    /// tree came out as a chain of depth 6 and width 1.
    route_s: Ladder,
    route_n: Ladder,
    /// Rolling prediction cost in nats — the mixing weight.
    pred_s: Ladder,
    pred_n: Ladder,
    children: Vec<usize>,
    seen: u64,
}

pub struct Tree {
    nodes: Vec<Node>,
    vocab: usize,
    d: usize,
    fanout: usize,
    depth_max: usize,
    rungs: usize,
    schedule: Schedule,
    eta: f64,
    // scratch, reused so a step allocates nothing
    path: Vec<usize>,
    logits: Vec<f64>,
    resid: Vec<f64>,
    mix: Vec<f64>,
    weights: Vec<f64>,
    per_level: Vec<f64>,
    per_level_n: Vec<f64>,
    grown: usize,
}

/// Everything the shape and the ladders need. Grouped so the two budgets that
/// `depth_max` pins down -- compute per token and node count -- sit next to
/// each other rather than being spread across an argument list.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub vocab: usize,
    pub d: usize,
    pub fanout: usize,
    pub depth_max: usize,
    pub rungs: usize,
    pub r: f64,
    pub g1: f64,
    pub eta: f64,
}

impl Tree {
    pub fn new(spec: Spec) -> Self {
        let Spec {
            vocab,
            d,
            fanout,
            depth_max,
            rungs,
            r,
            g1,
            eta,
        } = spec;
        assert!(vocab >= 2 && d >= 1 && fanout >= 1 && depth_max >= 1);
        let schedule = Schedule::Geometric { r, g1 };
        let mut t = Self {
            nodes: Vec::new(),
            vocab,
            d,
            fanout,
            depth_max,
            rungs,
            schedule,
            eta,
            path: Vec::with_capacity(depth_max + 1),
            logits: vec![0.0; vocab],
            resid: vec![0.0; vocab],
            mix: vec![0.0; vocab],
            weights: Vec::with_capacity(depth_max + 1),
            per_level: vec![0.0; depth_max + 1],
            per_level_n: vec![0.0; depth_max + 1],
            grown: 0,
        };
        t.spawn();
        t
    }

    fn spawn(&mut self) -> usize {
        let mem = if self.rungs <= 1 {
            AssocMemory::single_rect(self.vocab, self.d, 1.0)
        } else {
            AssocMemory::ladder_rect(self.vocab, self.d, self.schedule, self.rungs)
        };
        // Two rungs is the floor `Ladder::new` allows, and it is all a scalar
        // needs: rung 1 tracks recent, rung 2 pulls it toward the long run.
        let scalar = || Ladder::new(self.schedule, self.rungs.max(2), 1, 1);
        self.nodes.push(Node {
            mem,
            proto: Ladder::new(self.schedule, self.rungs.max(2), self.d, 1),
            route_s: scalar(),
            route_n: scalar(),
            pred_s: scalar(),
            pred_n: scalar(),
            children: Vec::new(),
            seen: 0,
        });
        self.grown += 1;
        self.nodes.len() - 1
    }

    /// The largest the tree can become, from the shape alone.
    pub fn capacity(&self) -> usize {
        let mut total = 0usize;
        let mut level = 1usize;
        for _ in 0..=self.depth_max {
            total += level;
            level = level.saturating_mul(self.fanout);
        }
        total
    }

    pub fn live_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Multiply-accumulates per token: read, write and one diffusion step per
    /// rung, at every level of the path.
    pub fn cost(&self) -> usize {
        let diffusion = if self.rungs <= 1 { 0 } else { self.rungs };
        (self.depth_max + 1) * (2 + diffusion) * self.vocab * self.d
    }

    pub fn live_parameters(&self) -> usize {
        self.nodes.len() * self.vocab * self.d
    }

    pub fn state_held(&self) -> usize {
        self.nodes.len() * self.vocab * self.d * self.rungs.max(1)
    }

    /// Mean nats charged at each depth, over everything seen so far. The
    /// division-of-labour check: if level 1 does not improve on level 0, the
    /// tree is not splitting the problem and the topology is dead.
    pub fn per_level_bits(&self) -> Vec<f64> {
        self.per_level
            .iter()
            .zip(&self.per_level_n)
            .map(|(s, n)| {
                if *n > 0.0 {
                    s / n * std::f64::consts::LOG2_E
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Cosine mismatch between `x` and a node's prototype, in `[0, 2]`. A node
    /// that has absorbed nothing yet has a zero prototype and is maximally
    /// surprised, so it will not steal traffic from a node that has learned
    /// something.
    fn mismatch(&self, node: usize, x: &[f64], xn: f64) -> f64 {
        let p = self.nodes[node].proto.rung(0).as_slice();
        let pn = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        if pn <= f64::MIN_POSITIVE || xn <= f64::MIN_POSITIVE {
            return 1.0;
        }
        let dot: f64 = p.iter().zip(x).map(|(a, b)| a * b).sum();
        1.0 - dot / (pn * xn)
    }

    /// Walks from the root to `depth_max`, instantiating a child when the input
    /// belongs to none of the existing ones. Fills `self.path`.
    fn route(&mut self, x: &[f64]) {
        self.path.clear();
        let xn = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut here = 0usize;
        self.path.push(here);
        for _ in 0..self.depth_max {
            let kids = self.nodes[here].children.clone();
            let mut fresh = false;
            let next = if kids.is_empty() {
                let c = self.spawn();
                self.nodes[here].children.push(c);
                fresh = true;
                c
            } else {
                let mut best = (f64::INFINITY, kids[0]);
                let mut all_novel = true;
                for &k in &kids {
                    let s = self.mismatch(k, x, xn);
                    // Its own rolling surprise is the threshold, so "novel"
                    // means novel *to that child*, not novel in absolute terms.
                    let thresh = mean(&self.nodes[k].route_s, &self.nodes[k].route_n);
                    if s <= thresh {
                        all_novel = false;
                    }
                    if s < best.0 {
                        best = (s, k);
                    }
                }
                if all_novel && kids.len() < self.fanout {
                    let c = self.spawn();
                    self.nodes[here].children.push(c);
                    fresh = true;
                    c
                } else {
                    best.1
                }
            };
            // Charge the routing surprise to whoever took it, before the
            // prototype moves, so the threshold reflects what it used to expect.
            //
            // Except on the step that created it. A node with no prototype
            // scores the maximum mismatch by definition, and letting that into
            // its own threshold sets the bar at the ceiling: nothing is ever
            // novel to it again and it accepts everything. Measured: orthogonal
            // inputs ended up sharing a leaf because the first child swallowed
            // both. A node's expectation should start at what it was created
            // for, which is exactly this input, so the honest charge is none.
            if !fresh {
                let s = self.mismatch(next, x, xn);
                self.nodes[next].route_s.inject(&[s], &[1.0], 1.0);
                self.nodes[next].route_s.relax();
                self.nodes[next].route_n.inject(&[1.0], &[1.0], 1.0);
                self.nodes[next].route_n.relax();
            }
            self.path.push(next);
            here = next;
        }
    }

    /// Predict `target` from `key`, pay, then write. Returns nats under the
    /// path mixture.
    pub fn observe(&mut self, key: &[f64], target: u32) -> f64 {
        assert_eq!(key.len(), self.d, "tree: key width mismatch");
        self.route(key);

        // Mixing weights from each node's rolling prediction cost. exp(-nats)
        // is that node's typical likelihood, so this is reliability weighting
        // and it introduces no parameter.
        self.weights.clear();
        let mut wsum = 0.0;
        for &node in &self.path {
            let r = mean(&self.nodes[node].pred_s, &self.nodes[node].pred_n);
            let w = (-r).exp();
            self.weights.push(w);
            wsum += w;
        }
        if wsum <= f64::MIN_POSITIVE {
            let uniform = 1.0 / self.weights.len() as f64;
            self.weights.iter_mut().for_each(|w| *w = uniform);
            wsum = 1.0;
        }
        for w in self.weights.iter_mut() {
            *w /= wsum;
        }

        self.mix.iter_mut().for_each(|v| *v = 0.0);
        // Softmax per node, accumulate the mixture, and charge each node its
        // own cost so the per-level curve is readable.
        for (li, &node) in self.path.clone().iter().enumerate() {
            self.nodes[node].mem.read().mul_vec(key, &mut self.logits);
            let peak = self
                .logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut z = 0.0;
            for l in self.logits.iter_mut() {
                *l = (*l - peak).exp();
                z += *l;
            }
            for l in self.logits.iter_mut() {
                *l /= z;
            }
            let own = -self.logits[target as usize].max(f64::MIN_POSITIVE).ln();
            self.per_level[li] += own;
            self.per_level_n[li] += 1.0;

            let w = self.weights[li];
            for (m, p) in self.mix.iter_mut().zip(&self.logits) {
                *m += w * p;
            }

            // Local write: this node's own residual, not the mixture's, so each
            // level learns the conditional it is actually responsible for.
            for (r, p) in self.resid.iter_mut().zip(&self.logits) {
                *r = -p;
            }
            self.resid[target as usize] += 1.0;
            self.nodes[node].mem.inject(&self.resid, key, self.eta);
            self.nodes[node].mem.relax();

            self.nodes[node].pred_s.inject(&[own], &[1.0], 1.0);
            self.nodes[node].pred_s.relax();
            self.nodes[node].pred_n.inject(&[1.0], &[1.0], 1.0);
            self.nodes[node].pred_n.relax();
            self.nodes[node].proto.inject(key, &[1.0], 1.0);
            self.nodes[node].proto.relax();
            self.nodes[node].seen += 1;
        }

        -self.mix[target as usize].max(f64::MIN_POSITIVE).ln()
    }

    /// How many nodes sit at each depth. A tree that never branches is a chain,
    /// and a chain is not a tree.
    pub fn width_by_depth(&self) -> Vec<usize> {
        let mut w = vec![0usize; self.depth_max + 1];
        let mut frontier = vec![0usize];
        let mut depth = 0;
        while !frontier.is_empty() && depth <= self.depth_max {
            w[depth] = frontier.len();
            let mut next = Vec::new();
            for &f in &frontier {
                next.extend_from_slice(&self.nodes[f].children);
            }
            frontier = next;
            depth += 1;
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn codes(vocab: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut c = vec![0.0; vocab * d];
        for t in 0..vocab {
            rng.fill_unit_vector(&mut c[t * d..(t + 1) * d]);
        }
        c
    }

    fn tree(vocab: usize, d: usize, depth: usize) -> Tree {
        Tree::new(Spec {
            vocab,
            d,
            fanout: 2,
            depth_max: depth,
            rungs: 4,
            r: 2.0,
            g1: 0.01,
            eta: 0.3,
        })
    }

    #[test]
    fn the_shape_alone_bounds_the_node_count() {
        let mut t = tree(8, 16, 4);
        let c = codes(8, 16, 1);
        let mut rng = Rng::new(2);
        for _ in 0..4000 {
            let tok = rng.next_below(8) as u32;
            let k = &c[tok as usize * 16..(tok as usize + 1) * 16];
            t.observe(k, rng.next_below(8) as u32);
        }
        // (2^5 - 1) / (2 - 1) = 31
        assert_eq!(t.capacity(), 31);
        assert!(
            t.live_nodes() <= t.capacity(),
            "grew past the shape: {} > {}",
            t.live_nodes(),
            t.capacity()
        );
        assert!(t.live_nodes() > 1, "never grew at all");
    }

    #[test]
    fn every_path_reaches_the_full_depth() {
        let mut t = tree(6, 12, 3);
        let c = codes(6, 12, 7);
        let mut rng = Rng::new(8);
        for _ in 0..500 {
            let tok = rng.next_below(6) as u32;
            t.observe(&c[tok as usize * 12..(tok as usize + 1) * 12], 0);
        }
        // depth_max + 1 levels are charged, every token, with no early stop.
        let n = t.per_level_bits();
        assert_eq!(n.len(), 4);
        assert!(
            n.iter().all(|v| v.is_finite()),
            "a level was never used: {n:?}"
        );
    }

    #[test]
    fn a_distinct_input_gets_its_own_branch() {
        // Two inputs that share nothing should not end up on the same leaf once
        // the tree has had a chance to split. If they do, routing is not
        // separating anything and the topology cannot divide labour.
        let d = 32;
        let mut t = tree(4, d, 2);
        let mut a = vec![0.0; d];
        let mut b = vec![0.0; d];
        a[0] = 1.0;
        b[d - 1] = 1.0;
        for _ in 0..200 {
            t.observe(&a, 0);
            t.observe(&b, 1);
        }
        t.route(&a);
        let leaf_a = *t.path.last().unwrap();
        t.route(&b);
        let leaf_b = *t.path.last().unwrap();
        assert_ne!(leaf_a, leaf_b, "orthogonal inputs share a leaf");
    }

    #[test]
    fn routing_never_looks_at_the_target() {
        // Same inputs, different targets: the path must be identical. If the
        // target could reach the router, this is where it would show.
        let d = 24;
        let mut x = tree(5, d, 3);
        let mut y = tree(5, d, 3);
        let c = codes(5, d, 11);
        let mut rng = Rng::new(12);
        let stream: Vec<u32> = (0..300).map(|_| rng.next_below(5) as u32).collect();
        for (i, &tok) in stream.iter().enumerate() {
            let k = &c[tok as usize * d..(tok as usize + 1) * d];
            x.observe(k, tok);
            y.observe(k, ((tok as usize + i) % 5) as u32);
        }
        let probe = &c[0..d];
        x.route(probe);
        y.route(probe);
        assert_eq!(x.path, y.path, "the path depends on the targets");
    }

    #[test]
    fn the_mixture_is_a_distribution() {
        let d = 16;
        let vocab = 7;
        let mut t = tree(vocab, d, 2);
        let c = codes(vocab, d, 3);
        let mut rng = Rng::new(4);
        for _ in 0..300 {
            let tok = rng.next_below(vocab as u64) as u32;
            t.observe(&c[tok as usize * d..(tok as usize + 1) * d], tok);
        }
        t.observe(&c[0..d], 0);
        let total: f64 = t.mix.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "mixture sums to {total}");
        assert!(t.mix.iter().all(|p| *p >= 0.0));
    }
}
