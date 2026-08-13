//! The node ring and its routing topology.
//!
//! Nodes live on a periodic 1-D lattice — a ring of `N` nodes. One dimension,
//! not two, and the reason is the only thing this file exists for.
//!
//! Under cursor ingress a token at stream position `t` enters at `t mod N`, so
//! **ring distance is stream lag**, exactly and without remainder. A 2-D torus
//! of side `G` splits the same lag into `(delta mod G, floor(delta/G) mod G)`,
//! which preserves two scales — one step in x is lag 1, one step in y is lag
//! `G` — and aliases every other lag onto those. DESIGN.md §36 measured what
//! that costs: the assembled vector stops naming a token past about three
//! positions back, on both a synthetic chain and real text, and no parameter
//! reaches it because the limit is the hop radius rather than any coefficient.
//!
//! Each node keeps its two ring neighbours plus a few long-range contacts drawn
//! with `P(v) ~ dist(u,v)^-alpha`. On a `d`-dimensional lattice `alpha = d` is
//! the unique exponent at which decentralised greedy routing is polylogarithmic
//! (Kleinberg 2000), so here it is 1. What that buys, now that distance is lag,
//! is a **delay structure**: one hop crosses a stream lag drawn from a
//! scale-free law, every scale present, and no period to choose. §37 rules out
//! getting the same thing from the consolidation ladder — averaging over a
//! timescale destroys the time index, so a low-pass is not a delay line.
//!
//! Two consequences worth stating before they surprise anyone.
//!
//! Reach is bounded by `N`: lag `k` and lag `k + N` land on the same node and
//! nothing downstream can separate them. Longer context costs nodes, and the
//! hops to use them grow only logarithmically.
//!
//! A trace has to survive as long as the reach it is meant to serve. A chord
//! lands on the node that held a token `k` positions back, and finds nothing if
//! that node has been overwritten since. Writes per node fall as visits per
//! token fall, which the same chords are what make possible, so the sparse
//! operating point is not a preference here but a requirement.
//!
//! Edges are directed. Particles flow one way, the ring edges are symmetric
//! anyway, and the long-range contacts stay as Kleinberg defines them.

use crate::rng::Rng;

/// Node positions on a periodic 1-D lattice.
///
/// There are no distance shells to precompute: on a ring exactly two nodes sit
/// at any distance `0 < d < N/2`, which is what makes "draw a node at distance
/// d" a closed form rather than a table.
#[derive(Clone, Debug)]
pub struct Ring {
    len: usize,
}

impl Ring {
    pub fn new(len: usize) -> Self {
        // Below 3 the two neighbours of a node collide.
        assert!(len >= 3, "ring length must be at least 3, got {len}");
        Self { len }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Largest distance any two nodes can be apart.
    #[inline]
    pub fn max_distance(&self) -> usize {
        self.len / 2
    }

    /// How many nodes sit at exactly this distance from any given node. Two,
    /// except at zero and — on an even ring — at the antipode.
    #[inline]
    pub fn shell_size(&self, distance: usize) -> usize {
        if distance == 0 || distance > self.max_distance() {
            0
        } else if 2 * distance == self.len {
            1
        } else {
            2
        }
    }

    /// Node reached by stepping `delta` forward, wrapping around.
    #[inline]
    pub fn shift(&self, node: u32, delta: usize) -> u32 {
        ((node as usize + delta) % self.len) as u32
    }

    /// Shorter of the two ways round.
    pub fn distance(&self, a: u32, b: u32) -> usize {
        let gap = (a as usize + self.len - b as usize) % self.len;
        gap.min(self.len - gap)
    }

    /// The two ring neighbours, in a fixed order so construction is
    /// reproducible.
    pub fn neighbours(&self, node: u32) -> [u32; 2] {
        [self.shift(node, 1), self.shift(node, self.len - 1)]
    }

    /// Uniformly random node at exactly `distance` from `node`.
    pub fn random_at_distance(&self, node: u32, distance: usize, rng: &mut Rng) -> u32 {
        debug_assert!(self.shell_size(distance) > 0, "empty shell at {distance}");
        if self.shell_size(distance) == 1 || rng.next_below(2) == 0 {
            self.shift(node, distance)
        } else {
            self.shift(node, self.len - distance)
        }
    }
}

/// How many long-range contacts each node gets, and how their lengths are
/// distributed.
#[derive(Clone, Copy, Debug)]
pub struct SmallWorld {
    pub long_range: usize,
    /// `P(v) ~ dist(u,v)^-exponent`. Leave at the lattice dimension, which is
    /// 1 on a ring, unless you are deliberately running the `topology` sweep.
    pub exponent: f64,
}

impl Default for SmallWorld {
    fn default() -> Self {
        Self {
            long_range: 4,
            exponent: 1.0,
        }
    }
}

/// Directed out-edges in compressed sparse row form.
///
/// CSR rather than a fixed stride because §1.9 will grow and prune edges, which
/// makes degree per-node. Nothing downstream may assume uniform degree.
#[derive(Clone, Debug)]
pub struct Topology {
    ring: Ring,
    start: Vec<u32>,
    target: Vec<u32>,
    /// Out-edges `0..lattice_degree` of every node are its ring neighbours and
    /// are **permanent**. §9 established that greedy routing cannot stall
    /// precisely because those always offer a strictly closer step; rewiring
    /// them away would destroy both that guarantee and connectivity. Everything
    /// past them is a long-range contact and is plastic. Derived from the ring
    /// rather than written down, so a change of lattice cannot leave a stale
    /// constant behind.
    lattice_degree: usize,
    /// Distance CDF used to draw long-range contacts, kept so rewiring samples
    /// from the same law the graph was built with.
    distance_cdf: Vec<f64>,
}

impl Topology {
    /// Both ring neighbours plus `spec.long_range` contacts drawn from the
    /// distance-decaying law.
    ///
    /// With ring distance equal to stream lag, that law is what makes a single
    /// hop a delay of scale-free length: every lag scale is represented,
    /// logarithmically, with no period chosen anywhere.
    pub fn small_world(ring: Ring, spec: SmallWorld, rng: &mut Rng) -> Self {
        assert!(spec.exponent >= 0.0, "exponent must be non-negative");
        let n = ring.len();

        // P(distance = d) ~ (nodes at distance d) * d^-exponent. Sampling the
        // distance first and then a node within that shell is exact and O(1),
        // where weighting all N-1 candidates directly would be O(N) per draw.
        let mut cdf = Vec::with_capacity(ring.max_distance());
        let mut acc = 0.0;
        for d in 1..=ring.max_distance() {
            acc += ring.shell_size(d) as f64 * (d as f64).powf(-spec.exponent);
            cdf.push(acc);
        }
        assert!(acc > 0.0, "no reachable shells: ring is degenerate");
        for c in cdf.iter_mut() {
            *c /= acc;
        }

        let lattice_degree = ring.neighbours(0).len();
        let mut start = Vec::with_capacity(n + 1);
        let mut target = Vec::with_capacity(n * (lattice_degree + spec.long_range));
        let mut row: Vec<u32> = Vec::with_capacity(lattice_degree + spec.long_range);
        for node in 0..n as u32 {
            start.push(target.len() as u32);
            row.clear();
            row.extend_from_slice(&ring.neighbours(node));

            for _ in 0..spec.long_range {
                // Reject self-loops and repeats: a duplicated edge would
                // silently double that contact's routing weight.
                let mut tries = 0;
                loop {
                    let u = rng.next_f64();
                    let d = cdf.partition_point(|&c| c < u).min(cdf.len() - 1) + 1;
                    let v = ring.random_at_distance(node, d, rng);
                    if v != node && !row.contains(&v) {
                        row.push(v);
                        break;
                    }
                    tries += 1;
                    assert!(
                        tries < 10_000,
                        "could not place a long-range contact for node {node}; \
                         is long_range too close to the node count?"
                    );
                }
            }
            target.extend_from_slice(&row);
        }
        start.push(target.len() as u32);
        Self {
            ring,
            start,
            target,
            lattice_degree,
            distance_cdf: cdf,
        }
    }

    #[inline]
    pub fn lattice_degree(&self) -> usize {
        self.lattice_degree
    }

    /// Slots of `node` that may be rewired: everything past the lattice.
    #[inline]
    pub fn plastic_slots(&self, node: u32) -> std::ops::Range<usize> {
        self.lattice_degree..self.degree(node)
    }

    /// Draws a candidate target for `node` from the same distance-decaying law
    /// the graph was built with, rejecting itself and anything it already
    /// points at.
    pub fn sample_contact(&self, node: u32, rng: &mut Rng) -> Option<u32> {
        for _ in 0..64 {
            let u = rng.next_f64();
            let d = self
                .distance_cdf
                .partition_point(|&c| c < u)
                .min(self.distance_cdf.len() - 1)
                + 1;
            let v = self.ring.random_at_distance(node, d, rng);
            if v != node && !self.out_edges(node).contains(&v) {
                return Some(v);
            }
        }
        None
    }

    /// Points one plastic slot somewhere else. Degree is unchanged, so the CSR
    /// needs no reallocation and per-hop compute is exactly constant.
    pub fn rewire(&mut self, node: u32, slot: usize, new_target: u32) {
        assert!(
            self.plastic_slots(node).contains(&slot),
            "slot {slot} of node {node} is a lattice edge and is not plastic"
        );
        assert_ne!(new_target, node, "rewiring would create a self-loop");
        let base = self.start[node as usize] as usize;
        self.target[base + slot] = new_target;
    }

    #[inline]
    pub fn ring(&self) -> &Ring {
        &self.ring
    }

    #[inline]
    pub fn out_edges(&self, node: u32) -> &[u32] {
        let lo = self.start[node as usize] as usize;
        let hi = self.start[node as usize + 1] as usize;
        &self.target[lo..hi]
    }

    #[inline]
    pub fn degree(&self, node: u32) -> usize {
        self.out_edges(node).len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.target.len()
    }

    /// Hops taken by decentralised greedy routing: at each step move to the
    /// out-neighbour closest to `to`, knowing only the current node's edges.
    ///
    /// This always terminates. The two ring neighbours guarantee that one of
    /// them is strictly closer whenever `from != to`, so greedy can never stall
    /// in a local minimum, and it takes at most `distance(from, to)` hops even
    /// if every long-range contact is useless.
    pub fn greedy_hops(&self, from: u32, to: u32) -> usize {
        let mut at = from;
        let mut hops = 0;
        let limit = self.ring.len();
        while at != to {
            let best = *self
                .out_edges(at)
                .iter()
                .min_by_key(|&&v| self.ring.distance(v, to))
                .expect("every node has out-edges");
            debug_assert!(
                self.ring.distance(best, to) < self.ring.distance(at, to),
                "greedy stalled at {at}: no neighbour is closer to {to}"
            );
            at = best;
            hops += 1;
            assert!(hops <= limit, "greedy routing failed to terminate");
        }
        hops
    }

    /// Nodes reachable from `origin`. Used to prove the graph is not secretly
    /// partitioned.
    pub fn reachable_count(&self, origin: u32) -> usize {
        let mut seen = vec![false; self.ring.len()];
        let mut stack = vec![origin];
        seen[origin as usize] = true;
        let mut count = 1;
        while let Some(node) = stack.pop() {
            for &v in self.out_edges(node) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    count += 1;
                    stack.push(v);
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topo(len: usize, seed: u64) -> Topology {
        let mut rng = Rng::new(seed);
        Topology::small_world(Ring::new(len), SmallWorld::default(), &mut rng)
    }

    #[test]
    fn distance_wraps_the_short_way() {
        let r = Ring::new(8);
        // Opposite ends are adjacent, not seven apart.
        assert_eq!(r.distance(0, 7), 1);
        assert_eq!(r.distance(7, 0), 1);
        assert_eq!(r.distance(0, 4), 4, "antipode of an even ring");
        assert_eq!(r.distance(3, 3), 0);
        for a in 0..r.len() as u32 {
            for b in 0..r.len() as u32 {
                assert_eq!(r.distance(a, b), r.distance(b, a));
            }
        }
    }

    #[test]
    fn shells_partition_everything_except_the_node_itself() {
        // Every other node belongs to exactly one shell, so the sizes must add
        // up to N - 1. If they did not, distance sampling would silently
        // exclude part of the ring — the partitioned-graph bug in another guise.
        // Distance zero is empty because nothing is ever drawn there.
        for len in [8usize, 9] {
            let r = Ring::new(len);
            let drawable: usize = (1..=r.max_distance()).map(|d| r.shell_size(d)).sum();
            assert_eq!(drawable, r.len() - 1, "ring of {len}");
            assert_eq!(r.shell_size(0), 0);
            assert_eq!(r.shell_size(1), 2);
            assert_eq!(r.shell_size(r.max_distance() + 1), 0, "past the antipode");
        }
        // The antipode of an even ring is a single node, not a pair; counting
        // it twice would bias every long-range draw toward the far side.
        assert_eq!(Ring::new(8).shell_size(4), 1);
        assert_eq!(Ring::new(9).shell_size(4), 2);
    }

    #[test]
    fn neighbours_are_two_distinct_nodes_at_distance_one() {
        let r = Ring::new(5);
        for node in 0..r.len() as u32 {
            let ns = r.neighbours(node);
            assert_ne!(ns[0], ns[1], "duplicate neighbour of {node}");
            for &a in &ns {
                assert_eq!(r.distance(node, a), 1);
                assert_ne!(a, node);
            }
        }
    }

    #[test]
    fn random_at_distance_lands_at_that_distance_and_reaches_both_sides() {
        let r = Ring::new(11);
        let mut rng = Rng::new(5);
        let (mut ahead, mut behind) = (0, 0);
        for _ in 0..400 {
            let v = r.random_at_distance(3, 3, &mut rng);
            assert_eq!(r.distance(3, v), 3);
            if v == r.shift(3, 3) {
                ahead += 1;
            } else {
                behind += 1;
            }
        }
        // Both directions must be reachable: drawing only one way would turn a
        // symmetric law into a drift.
        assert!(
            ahead > 100 && behind > 100,
            "{ahead} ahead, {behind} behind"
        );
    }

    #[test]
    fn every_node_has_the_full_degree_with_no_self_loops_or_repeats() {
        let t = topo(64, 7);
        let want = 2 + SmallWorld::default().long_range;
        for node in 0..t.ring().len() as u32 {
            let edges = t.out_edges(node);
            assert_eq!(edges.len(), want, "node {node} degree");
            assert!(!edges.contains(&node), "node {node} has a self-loop");
            let mut sorted = edges.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                edges.len(),
                "node {node} has a duplicate edge"
            );
        }
        assert_eq!(t.edge_count(), t.ring().len() * want);
    }

    #[test]
    fn the_lattice_degree_follows_the_ring_rather_than_a_constant() {
        // It used to be written down as four. A ring has two, and a stale
        // constant would have left the first long-range contacts inside the
        // permanent range, where turnover can never touch them.
        let t = topo(32, 1);
        assert_eq!(t.lattice_degree(), 2);
        for node in 0..t.ring().len() as u32 {
            for &v in &t.out_edges(node)[..t.lattice_degree()] {
                assert_eq!(t.ring().distance(node, v), 1, "permanent slot of {node}");
            }
        }
    }

    #[test]
    fn the_graph_is_one_piece() {
        // The bug that cost the previous generation: an all-to-all structure
        // quietly built block-diagonal. Reachability from an arbitrary node
        // must cover everything.
        let t = topo(256, 3);
        assert_eq!(t.reachable_count(0), t.ring().len());
        assert_eq!(t.reachable_count(129), t.ring().len());
    }

    #[test]
    fn greedy_never_stalls_and_stays_within_the_ring_distance() {
        let t = topo(128, 9);
        for to in [0u32, 1, 37, 64, 127] {
            for from in 0..t.ring().len() as u32 {
                let hops = t.greedy_hops(from, to);
                assert!(
                    hops <= t.ring().distance(from, to),
                    "{from} -> {to} took {hops} hops"
                );
            }
        }
    }

    #[test]
    fn long_range_lengths_follow_the_requested_exponent() {
        // Recover alpha from the sample: observed counts at distance d divided
        // by the shell size should scale as d^-alpha.
        let ring = Ring::new(1024);
        let mut rng = Rng::new(11);
        let spec = SmallWorld {
            long_range: 24,
            exponent: 1.0,
        };
        let t = Topology::small_world(ring, spec, &mut rng);

        let mut counts = vec![0.0; t.ring().max_distance() + 1];
        let lattice = t.lattice_degree();
        for node in 0..t.ring().len() as u32 {
            for &v in &t.out_edges(node)[lattice..] {
                counts[t.ring().distance(node, v)] += 1.0;
            }
        }
        let (x, y): (Vec<f64>, Vec<f64>) = (2..=t.ring().max_distance() / 2)
            .filter(|&d| counts[d] > 0.0)
            .map(|d| {
                (
                    (d as f64).ln(),
                    (counts[d] / t.ring().shell_size(d) as f64).ln(),
                )
            })
            .unzip();
        let slope = crate::linalg::linear_fit(&x, &y).0;
        assert!(
            (slope + spec.exponent).abs() < 0.2,
            "recovered exponent {}",
            -slope
        );
    }
}
