//! The node grid and its routing topology.
//!
//! Nodes live on a periodic square lattice — a 2-D torus of side `G`, so
//! `N = G^2`. The torus is not decoration: it is the metric that makes
//! content-addressed ingress (DESIGN.md §1.3) mean anything, and it is what
//! lets a particle's next hop be chosen from local information alone.
//!
//! Each node keeps its four axis neighbours plus a few long-range contacts,
//! drawn with `P(v) ~ dist(u,v)^-alpha`. On a `d`-dimensional lattice `alpha =
//! d` is the unique exponent at which decentralised greedy routing — each hop
//! chosen from local knowledge only — is polylogarithmic (Kleinberg 2000),
//! which is exactly our setting: route (b) descends a local surprise gradient
//! with no global view. Hence the default of 2.
//!
//! What the `topology` bench actually shows, though, is narrower than that
//! (DESIGN.md §8.6). Out to `N = 262144` the polylogarithmic regime is not
//! reached, and anything in `alpha` from 1 to 2 routes within 6% of the same
//! hop count. What *is* robust is the ceiling: past 2 the cost climbs sharply.
//! So 2 is the right default because it is asymptotically correct and costs
//! nothing at finite size — not because the measurement singles it out.
//!
//! Edges are directed. Particles flow one way, and the lattice edges are
//! symmetric anyway, so nothing is lost and the long-range contacts stay as
//! Kleinberg defines them.

use crate::rng::Rng;

/// Node positions on a periodic square lattice.
#[derive(Clone, Debug)]
pub struct Grid {
    side: usize,
    /// Relative offsets from any node, bucketed by their torus distance.
    /// `shells[d]` holds every `(dx, dy)` at distance exactly `d`, which makes
    /// "pick a uniformly random node at distance d" an O(1) draw.
    shells: Vec<Vec<(u32, u32)>>,
}

impl Grid {
    pub fn new(side: usize) -> Self {
        // Below 3 the two axis neighbours along a dimension collide.
        assert!(side >= 3, "grid side must be at least 3, got {side}");
        let mut shells: Vec<Vec<(u32, u32)>> = Vec::new();
        for dx in 0..side {
            for dy in 0..side {
                let d = Self::axis_gap(dx, side) + Self::axis_gap(dy, side);
                if shells.len() <= d {
                    shells.resize(d + 1, Vec::new());
                }
                shells[d].push((dx as u32, dy as u32));
            }
        }
        Self { side, shells }
    }

    /// Shorter of the two ways round the ring.
    #[inline]
    fn axis_gap(delta: usize, side: usize) -> usize {
        delta.min(side - delta)
    }

    #[inline]
    pub fn side(&self) -> usize {
        self.side
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.side * self.side
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Largest distance any two nodes can be apart.
    #[inline]
    pub fn max_distance(&self) -> usize {
        self.shells.len() - 1
    }

    /// How many nodes sit at exactly this distance from any given node.
    #[inline]
    pub fn shell_size(&self, distance: usize) -> usize {
        self.shells.get(distance).map_or(0, Vec::len)
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize) -> u32 {
        debug_assert!(x < self.side && y < self.side);
        (x * self.side + y) as u32
    }

    #[inline]
    pub fn coords(&self, node: u32) -> (usize, usize) {
        let n = node as usize;
        debug_assert!(n < self.len());
        (n / self.side, n % self.side)
    }

    /// Node reached by stepping `(dx, dy)` from `node`, wrapping around.
    #[inline]
    pub fn shift(&self, node: u32, dx: u32, dy: u32) -> u32 {
        let (x, y) = self.coords(node);
        self.index((x + dx as usize) % self.side, (y + dy as usize) % self.side)
    }

    /// Manhattan distance on the torus.
    pub fn distance(&self, a: u32, b: u32) -> usize {
        let (ax, ay) = self.coords(a);
        let (bx, by) = self.coords(b);
        let dx = (ax + self.side - bx) % self.side;
        let dy = (ay + self.side - by) % self.side;
        Self::axis_gap(dx, self.side) + Self::axis_gap(dy, self.side)
    }

    /// The four lattice neighbours, in a fixed order so construction is
    /// reproducible.
    pub fn axis_neighbours(&self, node: u32) -> [u32; 4] {
        let last = (self.side - 1) as u32;
        [
            self.shift(node, 1, 0),
            self.shift(node, last, 0),
            self.shift(node, 0, 1),
            self.shift(node, 0, last),
        ]
    }

    /// Uniformly random node at exactly `distance` from `node`.
    pub fn random_at_distance(&self, node: u32, distance: usize, rng: &mut Rng) -> u32 {
        let shell = &self.shells[distance];
        let (dx, dy) = shell[rng.next_below(shell.len() as u64) as usize];
        self.shift(node, dx, dy)
    }
}

/// How many long-range contacts each node gets, and how their lengths are
/// distributed.
#[derive(Clone, Copy, Debug)]
pub struct SmallWorld {
    pub long_range: usize,
    /// `P(v) ~ dist(u,v)^-exponent`. Leave at the lattice dimension (2) unless
    /// you are deliberately running the `topology` sweep.
    pub exponent: f64,
}

impl Default for SmallWorld {
    fn default() -> Self {
        Self { long_range: 4, exponent: 2.0 }
    }
}

/// Directed out-edges in compressed sparse row form.
///
/// CSR rather than a fixed stride because §1.9 will grow and prune edges, which
/// makes degree per-node. Nothing downstream may assume uniform degree.
#[derive(Clone, Debug)]
pub struct Topology {
    grid: Grid,
    start: Vec<u32>,
    target: Vec<u32>,
    /// Out-edges `0..lattice_degree` of every node are its axis neighbours and
    /// are **permanent**. §9 established that greedy routing cannot stall
    /// precisely because those four always offer a strictly closer step;
    /// rewiring them away would destroy both that guarantee and connectivity.
    /// Everything past them is a long-range contact and is plastic.
    lattice_degree: usize,
    /// Distance CDF used to draw long-range contacts, kept so rewiring samples
    /// from the same law the graph was built with.
    distance_cdf: Vec<f64>,
}

impl Topology {
    /// Four lattice neighbours plus `spec.long_range` contacts drawn from the
    /// distance-decaying law.
    pub fn small_world(grid: Grid, spec: SmallWorld, rng: &mut Rng) -> Self {
        assert!(spec.exponent >= 0.0, "exponent must be non-negative");
        let n = grid.len();

        // P(distance = d) ~ (nodes at distance d) * d^-exponent. Sampling the
        // distance first and then a node within that shell is exact and O(1),
        // where weighting all N-1 candidates directly would be O(N) per draw.
        let mut cdf = Vec::with_capacity(grid.max_distance());
        let mut acc = 0.0;
        for d in 1..=grid.max_distance() {
            acc += grid.shell_size(d) as f64 * (d as f64).powf(-spec.exponent);
            cdf.push(acc);
        }
        assert!(acc > 0.0, "no reachable shells: grid is degenerate");
        for c in cdf.iter_mut() {
            *c /= acc;
        }

        let lattice_degree = 4;
        let mut start = Vec::with_capacity(n + 1);
        let mut target = Vec::with_capacity(n * (4 + spec.long_range));
        let mut row: Vec<u32> = Vec::with_capacity(4 + spec.long_range);
        for node in 0..n as u32 {
            start.push(target.len() as u32);
            row.clear();
            row.extend_from_slice(&grid.axis_neighbours(node));

            for _ in 0..spec.long_range {
                // Reject self-loops and repeats: a duplicated edge would
                // silently double that contact's routing weight.
                let mut tries = 0;
                loop {
                    let u = rng.next_f64();
                    let d = cdf.partition_point(|&c| c < u).min(cdf.len() - 1) + 1;
                    let v = grid.random_at_distance(node, d, rng);
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
        Self { grid, start, target, lattice_degree, distance_cdf: cdf }
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
            let d = self.distance_cdf.partition_point(|&c| c < u).min(self.distance_cdf.len() - 1) + 1;
            let v = self.grid.random_at_distance(node, d, rng);
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
    pub fn grid(&self) -> &Grid {
        &self.grid
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
    /// This always terminates. The four lattice neighbours guarantee that some
    /// neighbour is strictly closer whenever `from != to`, so greedy can never
    /// stall in a local minimum, and it takes at most `distance(from, to)` hops
    /// even if every long-range contact is useless.
    pub fn greedy_hops(&self, from: u32, to: u32) -> usize {
        let mut at = from;
        let mut hops = 0;
        let limit = self.grid.len();
        while at != to {
            let best = *self
                .out_edges(at)
                .iter()
                .min_by_key(|&&v| self.grid.distance(v, to))
                .expect("every node has out-edges");
            debug_assert!(
                self.grid.distance(best, to) < self.grid.distance(at, to),
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
        let mut seen = vec![false; self.grid.len()];
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

    fn topo(side: usize, seed: u64) -> Topology {
        let mut rng = Rng::new(seed);
        Topology::small_world(Grid::new(side), SmallWorld::default(), &mut rng)
    }

    #[test]
    fn index_and_coords_round_trip() {
        let g = Grid::new(7);
        for node in 0..g.len() as u32 {
            let (x, y) = g.coords(node);
            assert_eq!(g.index(x, y), node);
        }
    }

    #[test]
    fn distance_wraps_around_both_axes() {
        let g = Grid::new(8);
        // Opposite edges of the torus are adjacent, not seven apart.
        assert_eq!(g.distance(g.index(0, 0), g.index(7, 0)), 1);
        assert_eq!(g.distance(g.index(0, 0), g.index(0, 7)), 1);
        assert_eq!(g.distance(g.index(0, 0), g.index(7, 7)), 2);
        // Antipode of an even torus.
        assert_eq!(g.distance(g.index(0, 0), g.index(4, 4)), 8);
        assert_eq!(g.distance(g.index(2, 5), g.index(2, 5)), 0);
        // Symmetry, on every pair.
        for a in 0..g.len() as u32 {
            for b in 0..g.len() as u32 {
                assert_eq!(g.distance(a, b), g.distance(b, a));
            }
        }
    }

    #[test]
    fn shells_partition_the_whole_grid() {
        // Every relative offset belongs to exactly one shell, so shell sizes
        // must sum to N. If they did not, distance sampling would silently
        // exclude part of the grid — the partitioned-graph bug in another guise.
        let g = Grid::new(9);
        let total: usize = (0..=g.max_distance()).map(|d| g.shell_size(d)).sum();
        assert_eq!(total, g.len());
        assert_eq!(g.shell_size(0), 1);
        assert_eq!(g.shell_size(1), 4, "four nodes at Manhattan distance 1");
    }

    #[test]
    fn axis_neighbours_are_four_distinct_nodes_at_distance_one() {
        let g = Grid::new(5);
        for node in 0..g.len() as u32 {
            let ns = g.axis_neighbours(node);
            for (i, &a) in ns.iter().enumerate() {
                assert_eq!(g.distance(node, a), 1, "neighbour {i} of {node}");
                assert_ne!(a, node);
                for &b in &ns[i + 1..] {
                    assert_ne!(a, b, "duplicate neighbour of {node}");
                }
            }
        }
    }

    #[test]
    fn every_node_has_the_full_degree_with_no_self_loops_or_repeats() {
        let t = topo(12, 7);
        let want = 4 + SmallWorld::default().long_range;
        for node in 0..t.grid().len() as u32 {
            let edges = t.out_edges(node);
            assert_eq!(edges.len(), want, "node {node} degree");
            assert!(!edges.contains(&node), "node {node} has a self-loop");
            let mut sorted = edges.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), edges.len(), "node {node} has a duplicate edge");
        }
        assert_eq!(t.edge_count(), t.grid().len() * want);
    }

    #[test]
    fn the_graph_is_one_piece() {
        // The bug that cost the previous generation: an all-to-all structure
        // that was quietly built block-diagonal. Reachability from an arbitrary
        // node must cover everything.
        let t = topo(16, 3);
        assert_eq!(t.reachable_count(0), t.grid().len());
        assert_eq!(t.reachable_count(129), t.grid().len());
    }

    #[test]
    fn long_range_lengths_follow_the_requested_exponent() {
        // Recover alpha from the sample: observed counts at distance d divided
        // by the shell size should scale as d^-alpha.
        let side = 33;
        let grid = Grid::new(side);
        let mut rng = Rng::new(11);
        let spec = SmallWorld { long_range: 24, exponent: 2.0 };
        let t = Topology::small_world(grid, spec, &mut rng);

        let mut counts = vec![0.0; t.grid().max_distance() + 1];
        for node in 0..t.grid().len() as u32 {
            // Long-range contacts are the entries past the four lattice ones.
            for &v in &t.out_edges(node)[4..] {
                counts[t.grid().distance(node, v)] += 1.0;
            }
        }
        let (x, y): (Vec<f64>, Vec<f64>) = (2..=t.grid().max_distance() / 2)
            .filter(|&d| counts[d] > 0.0)
            .map(|d| ((d as f64).ln(), (counts[d] / t.grid().shell_size(d) as f64).ln()))
            .unzip();
        let slope = crate::linalg::linear_fit(&x, &y).0;
        assert!((slope + spec.exponent).abs() < 0.2, "recovered exponent {}", -slope);
    }

    #[test]
    fn greedy_routing_always_arrives() {
        let t = topo(24, 5);
        let g = t.grid();
        let mut rng = Rng::new(99);
        for _ in 0..2_000 {
            let a = rng.next_below(g.len() as u64) as u32;
            let b = rng.next_below(g.len() as u64) as u32;
            let hops = t.greedy_hops(a, b);
            // Long-range contacts may help but the lattice bounds the worst case.
            assert!(hops <= g.distance(a, b), "{a}->{b}: {hops} hops");
        }
    }

    #[test]
    fn construction_is_reproducible_and_seed_sensitive() {
        assert_eq!(topo(10, 1).target, topo(10, 1).target);
        assert_ne!(topo(10, 1).target, topo(10, 2).target);
    }
}
