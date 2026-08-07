//! Deterministic event-driven scheduler.
//!
//! DESIGN.md §3 makes bit-level reproducibility non-negotiable, which rules out
//! genuinely asynchronous message passing. Instead time is bucketed into ticks
//! and one tick is processed like this:
//!
//! 1. sort the live particles by `(node, serial)`;
//! 2. process each node's particles **in serial order**, because the delta-rule
//!    writes into a node's matrix do not commute;
//! 3. run different nodes **in parallel** — they touch disjoint state, and the
//!    expectations they route against are a snapshot frozen before the tick;
//! 4. merge the children back in node order, assigning fresh serials.
//!
//! Every step of that is order-independent across threads, so the result is
//! identical for any thread count. `identical_under_any_thread_count` pins it.
//!
//! Mass is the conserved measure. A particle's mass is split across the chosen
//! edges and the absorb option, which sum to one, so `absorbed + in flight` is
//! invariant per token up to floating point. Children below `mass_floor` are
//! absorbed where they stand rather than forwarded, which bounds the live
//! particle count by `1 / mass_floor` without any hop limit.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::graph::Topology;
use crate::node::{Context, NodeBank, Scratch};

/// Live particles, struct-of-arrays. The engine sorts indices, never payloads.
#[derive(Clone, Debug)]
pub struct Swarm {
    d_head: usize,
    payload: Vec<f64>,
    mass: Vec<f64>,
    token: Vec<u32>,
    slot: Vec<u16>,
    node: Vec<u32>,
    /// Creation rank within the previous tick's deterministic merge. The
    /// tie-break that fixes write order when several particles land on one node.
    serial: Vec<u64>,
}

impl Swarm {
    fn new(d_head: usize) -> Self {
        Self {
            d_head,
            payload: Vec::new(),
            mass: Vec::new(),
            token: Vec::new(),
            slot: Vec::new(),
            node: Vec::new(),
            serial: Vec::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mass.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mass.is_empty()
    }

    #[inline]
    pub fn payload(&self, i: usize) -> &[f64] {
        &self.payload[i * self.d_head..(i + 1) * self.d_head]
    }

    #[inline]
    pub fn mass(&self, i: usize) -> f64 {
        self.mass[i]
    }

    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }

    fn clear(&mut self) {
        self.payload.clear();
        self.mass.clear();
        self.token.clear();
        self.slot.clear();
        self.node.clear();
        self.serial.clear();
    }

    fn push(&mut self, node: u32, token: u32, slot: u16, mass: f64, payload: &[f64], serial: u64) {
        debug_assert_eq!(payload.len(), self.d_head);
        self.payload.extend_from_slice(payload);
        self.mass.push(mass);
        self.token.push(token);
        self.slot.push(slot);
        self.node.push(node);
        self.serial.push(serial);
    }
}

/// One token's reassembled output.
#[derive(Clone, Debug)]
pub struct TokenOutput {
    /// `sum_i mass_i * payload_i`, laid out slot-major.
    pub accumulated: Vec<f64>,
    pub absorbed_mass: f64,
    /// Mass-weighted **visit** count. Note the units: a fragment absorbed at
    /// its ingress node without crossing a single edge counts as 1, so this is
    /// edge traversals **plus one**, and is not directly comparable with
    /// `Topology::greedy_hops`, which counts edges and returns 0 for a
    /// zero-length route.
    pub hop_mass: f64,
    /// Node visits charged to this token. This is the compute it actually
    /// cost, and the quantity DESIGN.md §1.6 claims is independent of how far
    /// into the sequence the token sits.
    pub visits: u64,
    /// Mass-weighted **circular** sums of the grid coordinates where this
    /// token's mass came to rest, `[x, y]`.
    ///
    /// Circular because the grid is a torus. Averaging raw coordinates would
    /// put the mean of `{0, side-1}` in the middle of the grid instead of at
    /// the seam where it belongs — the classic wraparound bug, and the reason
    /// `resting_place` exists rather than a plain division.
    anchor_cos: [f64; 2],
    anchor_sin: [f64; 2],
    injected_tick: u64,
}

impl TokenOutput {
    /// Mass-weighted mean number of node visits before absorption. Subtract
    /// one to compare against an edge-counting figure.
    pub fn mean_hops(&self) -> f64 {
        if self.absorbed_mass > 0.0 { self.hop_mass / self.absorbed_mass } else { 0.0 }
    }

    /// Circular mean of where this token's mass was absorbed.
    ///
    /// `None` when the mass is spread evenly enough around either ring that the
    /// resultant vanishes and no direction is meaningful — a genuine
    /// degeneracy, not a tuned threshold.
    pub fn resting_place(&self, side: usize) -> Option<(usize, usize)> {
        let axis = |cos: f64, sin: f64| -> Option<usize> {
            if (cos * cos + sin * sin).sqrt() < 1e-12 {
                return None;
            }
            let theta = sin.atan2(cos);
            let unit = (theta.rem_euclid(std::f64::consts::TAU)) / std::f64::consts::TAU;
            Some(((unit * side as f64).round() as usize) % side)
        };
        Some((
            axis(self.anchor_cos[0], self.anchor_sin[0])?,
            axis(self.anchor_cos[1], self.anchor_sin[1])?,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineParams {
    /// Children below this mass are absorbed instead of forwarded, which caps
    /// the live particle count at `1 / mass_floor` per unit of injected mass.
    pub mass_floor: f64,
    /// Payload slots per token, needed to lay out the reassembly buffer.
    pub slots: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TickStats {
    pub tick: u64,
    pub processed: usize,
    pub spawned: usize,
    pub active_nodes: usize,
    pub absorbed_mass: f64,
    pub forwarded_mass: f64,
    pub mean_surprise: f64,
}

/// What one node produced during a tick.
struct GroupOutput {
    node: u32,
    /// `(target, token, slot, mass, payload)`
    children: Vec<(u32, u32, u16, f64, Vec<f64>)>,
    /// `(token, slot, mass, payload)`
    absorbed: Vec<(u32, u16, f64, Vec<f64>)>,
    surprise_sum: f64,
    visits: u64,
}

pub struct Engine {
    params: EngineParams,
    d_head: usize,
    current: Swarm,
    next: Swarm,
    tick: u64,
    next_serial: u64,
    /// Particle indices sorted by `(node, serial)`.
    order: Vec<u32>,
    /// For each node, its half-open range in `order`, or `None`.
    range: Vec<Option<(u32, u32)>>,
    outputs: BTreeMap<u32, TokenOutput>,
    visits: Vec<u64>,
}

impl Engine {
    pub fn new(bank: &NodeBank, params: EngineParams) -> Self {
        assert!(params.mass_floor > 0.0, "mass floor must be positive or particles never die");
        assert!(params.slots >= 1, "a token needs at least one slot");
        let d_head = bank.params().d_head;
        Self {
            params,
            d_head,
            current: Swarm::new(d_head),
            next: Swarm::new(d_head),
            tick: 0,
            next_serial: 0,
            order: Vec::new(),
            range: vec![None; bank.len()],
            outputs: BTreeMap::new(),
            visits: vec![0; bank.len()],
        }
    }

    #[inline]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    #[inline]
    pub fn live(&self) -> &Swarm {
        &self.current
    }

    #[inline]
    pub fn outputs(&self) -> &BTreeMap<u32, TokenOutput> {
        &self.outputs
    }

    /// Visits per node since construction. The load-balance diagnostic.
    #[inline]
    pub fn visits(&self) -> &[u64] {
        &self.visits
    }

    /// Tokens whose particles have all been absorbed, in ascending order.
    /// Their output is complete and will not change.
    pub fn settled(&self) -> Vec<u32> {
        let live: std::collections::BTreeSet<u32> = self.current.token.iter().copied().collect();
        self.outputs.keys().copied().filter(|t| !live.contains(t)).collect()
    }

    pub fn take_output(&mut self, token: u32) -> Option<TokenOutput> {
        self.outputs.remove(&token)
    }

    /// Injects one token's shattered particles at the current tick.
    ///
    /// `seeds` is `(node, slot, mass, payload)`; masses should sum to one so
    /// that mass reads as a probability measure downstream.
    pub fn inject(&mut self, token: u32, seeds: &[(u32, u16, f64, Vec<f64>)]) {
        assert!(!self.outputs.contains_key(&token), "token {token} was already injected");
        self.outputs.insert(
            token,
            TokenOutput {
                accumulated: vec![0.0; self.params.slots * self.d_head],
                absorbed_mass: 0.0,
                hop_mass: 0.0,
                visits: 0,
                anchor_cos: [0.0; 2],
                anchor_sin: [0.0; 2],
                injected_tick: self.tick,
            },
        );
        for (node, slot, mass, payload) in seeds {
            assert!((*slot as usize) < self.params.slots, "slot {slot} is out of range");
            self.current.push(*node, token, *slot, *mass, payload, self.next_serial);
            self.next_serial += 1;
        }
    }

    /// Advances one tick. Returns statistics; `processed == 0` means the swarm
    /// is empty and every particle has been absorbed.
    pub fn step(&mut self, topology: &Topology, bank: &mut NodeBank) -> TickStats {
        if self.current.is_empty() {
            self.tick += 1;
            return TickStats { tick: self.tick, ..Default::default() };
        }

        // Charge this tick's work to the tokens that caused it.
        for &token in &self.current.token {
            if let Some(out) = self.outputs.get_mut(&token) {
                out.visits += 1;
            }
        }

        // 1. Deterministic order: by node, then by creation rank.
        self.order.clear();
        self.order.extend(0..self.current.len() as u32);
        let (nodes_of, serial_of) = (&self.current.node, &self.current.serial);
        self.order.sort_unstable_by_key(|&i| (nodes_of[i as usize], serial_of[i as usize]));

        // 2. Contiguous run per node.
        for r in self.range.iter_mut() {
            *r = None;
        }
        let mut start = 0usize;
        while start < self.order.len() {
            let node = self.current.node[self.order[start] as usize];
            let mut end = start + 1;
            while end < self.order.len() && self.current.node[self.order[end] as usize] == node {
                end += 1;
            }
            self.range[node as usize] = Some((start as u32, end as u32));
            start = end;
        }

        // 3. Nodes run in parallel over disjoint state. Sweeping all of them and
        //    skipping the idle ones is what makes the scattered `&mut` safe; the
        //    skip is a predictable branch and the node count is small next to a
        //    tick's real work.
        let params = self.params;
        let (node_slice, expects, node_params) = bank.parts_mut();
        let (order, current, range) = (&self.order, &self.current, &self.range);
        let mut groups: Vec<GroupOutput> = node_slice
            .par_iter_mut()
            .enumerate()
            .filter_map(|(id, node)| {
                let (lo, hi) = range[id]?;
                let node_id = id as u32;
                let edges = topology.out_edges(node_id);
                let mut scratch = Scratch::default();
                let mut out = GroupOutput {
                    node: node_id,
                    children: Vec::new(),
                    absorbed: Vec::new(),
                    surprise_sum: 0.0,
                    visits: 0,
                };
                for &pi in &order[lo as usize..hi as usize] {
                    let i = pi as usize;
                    let lo = id * node_params.d_head;
                    let ctx = Context {
                        params: &node_params,
                        out_edges: edges,
                        expects,
                        self_expect: &expects[lo..lo + node_params.d_head],
                    };
                    let outcome = node.step(&ctx, current.payload(i), &mut scratch);
                    out.surprise_sum += outcome.surprise;
                    out.visits += 1;

                    let (token, slot, mass) =
                        (current.token[i], current.slot[i], current.mass[i]);
                    // The absorb option is the last weight, past every edge.
                    let mut kept = 0.0;
                    for (e, &w) in outcome.weights[..edges.len()].iter().enumerate() {
                        if w <= 0.0 {
                            continue;
                        }
                        let child = mass * w;
                        if child < params.mass_floor {
                            // Too faint to be worth a hop; absorb it here so the
                            // mass is still accounted for.
                            out.absorbed.push((token, slot, child, outcome.emitted.clone()));
                        } else {
                            kept += child;
                            out.children.push((
                                edges[e],
                                token,
                                slot,
                                child,
                                outcome.emitted.clone(),
                            ));
                        }
                    }
                    let _ = kept;
                    let absorb = mass * outcome.weights[edges.len()];
                    if absorb > 0.0 {
                        out.absorbed.push((token, slot, absorb, outcome.emitted.clone()));
                    }
                }
                Some(out)
            })
            .collect();
        // `collect` preserves iteration order, so groups arrive sorted by node.
        groups.sort_unstable_by_key(|g| g.node);

        // 4. Merge in node order, assigning fresh serials.
        self.next.clear();
        let mut stats = TickStats {
            tick: self.tick,
            processed: self.current.len(),
            active_nodes: groups.len(),
            ..Default::default()
        };
        let mut surprise_sum = 0.0;
        let side = topology.grid().side();
        for g in &groups {
            self.visits[g.node as usize] += g.visits;
            surprise_sum += g.surprise_sum;
            for (target, token, slot, mass, payload) in &g.children {
                self.next.push(*target, *token, *slot, *mass, payload, self.next_serial);
                self.next_serial += 1;
                stats.forwarded_mass += mass;
                stats.spawned += 1;
            }
            let (gx, gy) = topology.grid().coords(g.node);
            for (token, slot, mass, payload) in &g.absorbed {
                self.deposit(*token, *slot, *mass, payload, (gx, gy), side);
                stats.absorbed_mass += mass;
            }
        }
        stats.mean_surprise =
            if stats.processed > 0 { surprise_sum / stats.processed as f64 } else { 0.0 };

        // 5. Republish expectations for the nodes that fired, so the next tick
        //    routes against a consistent snapshot.
        for g in &groups {
            bank.publish(g.node);
        }

        std::mem::swap(&mut self.current, &mut self.next);
        self.tick += 1;
        stats
    }

    fn deposit(
        &mut self,
        token: u32,
        slot: u16,
        mass: f64,
        payload: &[f64],
        at: (usize, usize),
        side: usize,
    ) {
        let d = self.d_head;
        let tick = self.tick;
        let out = self.outputs.get_mut(&token).expect("absorbed a particle of an unknown token");
        let lo = slot as usize * d;
        for (a, &p) in out.accumulated[lo..lo + d].iter_mut().zip(payload) {
            *a += mass * p;
        }
        out.absorbed_mass += mass;
        out.hop_mass += mass * (tick - out.injected_tick + 1) as f64;

        let step = std::f64::consts::TAU / side as f64;
        for (axis, coord) in [at.0, at.1].into_iter().enumerate() {
            let theta = coord as f64 * step;
            out.anchor_cos[axis] += mass * theta.cos();
            out.anchor_sin[axis] += mass * theta.sin();
        }
    }

    /// Runs ticks until nothing is left in flight.
    pub fn run_to_quiescence(
        &mut self,
        topology: &Topology,
        bank: &mut NodeBank,
        max_ticks: u64,
    ) -> Vec<TickStats> {
        let mut log = Vec::new();
        while !self.current.is_empty() {
            log.push(self.step(topology, bank));
            assert!(
                (log.len() as u64) < max_ticks,
                "still {} particles in flight after {max_ticks} ticks",
                self.current.len()
            );
        }
        log
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Grid, SmallWorld, Topology};
    use crate::ladder::Schedule;
    use crate::node::{AbsorbRule, NodeParams};
    use crate::rng::Rng;

    const D: usize = 16;
    const SLOTS: usize = 4;

    fn fixture(side: usize, seed: u64) -> (Topology, NodeBank) {
        let mut rng = Rng::new(seed);
        let t = Topology::small_world(Grid::new(side), SmallWorld::default(), &mut rng);
        let bank = NodeBank::new(
            &t,
            NodeParams {
                absorb: AbsorbRule::Relative,
                d_head: D,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                context_scales: 1,
            },
        );
        (t, bank)
    }

    fn seeds(rng: &mut Rng, nodes: usize) -> Vec<(u32, u16, f64, Vec<f64>)> {
        (0..SLOTS)
            .map(|s| {
                let mut v = vec![0.0; D];
                rng.fill_unit_vector(&mut v);
                (rng.next_below(nodes as u64) as u32, s as u16, 1.0 / SLOTS as f64, v)
            })
            .collect()
    }

    fn params() -> EngineParams {
        EngineParams { mass_floor: 1e-3, slots: SLOTS }
    }

    /// Streams `tokens` tokens, one injected per tick, and returns the outputs
    /// plus the per-node visit histogram.
    fn drive(side: usize, seed: u64, tokens: u32, p: EngineParams) -> (Vec<TokenOutput>, Vec<u64>) {
        let (t, mut bank) = fixture(side, seed);
        let mut engine = Engine::new(&bank, p);
        let mut rng = Rng::new(seed ^ 0xBEEF);
        let n = bank.len();
        for token in 0..tokens {
            engine.inject(token, &seeds(&mut rng, n));
            engine.step(&t, &mut bank);
        }
        engine.run_to_quiescence(&t, &mut bank, 10_000);
        let outputs = (0..tokens).map(|k| engine.take_output(k).unwrap()).collect();
        (outputs, engine.visits().to_vec())
    }

    #[test]
    fn the_resting_place_averages_across_the_seam() {
        // The wraparound bug this exists to prevent: mass split between x = 0
        // and x = side-1 is concentrated *at the seam*, and a plain arithmetic
        // mean would report the opposite side of the grid instead.
        let side = 8usize;
        let mut out = TokenOutput {
            accumulated: vec![0.0; D * SLOTS],
            absorbed_mass: 0.0,
            hop_mass: 0.0,
            visits: 0,
            anchor_cos: [0.0; 2],
            anchor_sin: [0.0; 2],
            injected_tick: 0,
        };
        let step = std::f64::consts::TAU / side as f64;
        for (x, mass) in [(0usize, 0.5), (side - 1, 0.5)] {
            let theta = x as f64 * step;
            out.anchor_cos[0] += mass * theta.cos();
            out.anchor_sin[0] += mass * theta.sin();
            // y concentrated at 3, to check the axes are handled independently.
            let phi = 3.0 * step;
            out.anchor_cos[1] += mass * phi.cos();
            out.anchor_sin[1] += mass * phi.sin();
        }
        let (x, y) = out.resting_place(side).expect("resultant is not degenerate");
        let ring_gap = |a: usize, b: usize| (a + side - b) % side;
        let to_seam = ring_gap(x, 0).min(ring_gap(0, x));
        assert!(to_seam <= 1, "mean landed at {x}, which is {to_seam} from the seam");
        assert_eq!(y, 3, "the other axis must be unaffected");
    }

    #[test]
    fn a_resting_place_spread_evenly_round_the_ring_is_undefined() {
        let side = 8usize;
        let mut out = TokenOutput {
            accumulated: vec![0.0; D * SLOTS],
            absorbed_mass: 0.0,
            hop_mass: 0.0,
            visits: 0,
            anchor_cos: [0.0; 2],
            anchor_sin: [0.0; 2],
            injected_tick: 0,
        };
        let step = std::f64::consts::TAU / side as f64;
        for x in 0..side {
            let theta = x as f64 * step;
            for axis in 0..2 {
                out.anchor_cos[axis] += theta.cos() / side as f64;
                out.anchor_sin[axis] += theta.sin() / side as f64;
            }
        }
        assert!(out.resting_place(side).is_none(), "a vanishing resultant has no direction");
    }

    #[test]
    fn every_token_ends_with_exactly_the_mass_it_started_with() {
        // Mass is the conserved measure; if this drifts, the absorbing-chain
        // argument that guarantees halting is not actually implemented.
        let (outputs, _) = drive(8, 1, 24, params());
        for (i, o) in outputs.iter().enumerate() {
            assert!(
                (o.absorbed_mass - 1.0).abs() < 1e-12,
                "token {i} absorbed {} of its mass",
                o.absorbed_mass
            );
        }
    }

    #[test]
    fn everything_halts_without_a_hop_limit() {
        // Nothing bounds the walk except the absorb option and the mass floor.
        let (t, mut bank) = fixture(8, 2);
        let mut engine = Engine::new(&bank, params());
        let mut rng = Rng::new(77);
        engine.inject(0, &seeds(&mut rng, bank.len()));
        let log = engine.run_to_quiescence(&t, &mut bank, 10_000);
        assert!(!log.is_empty());
        assert!(engine.live().is_empty());
        // 1/mass_floor bounds the live count, so the walk cannot run long.
        assert!(log.len() < 200, "took {} ticks", log.len());
    }

    #[test]
    fn live_particle_count_is_bounded_by_the_mass_floor() {
        let p = EngineParams { mass_floor: 1e-2, ..params() };
        let (t, mut bank) = fixture(10, 3);
        let mut engine = Engine::new(&bank, p);
        let mut rng = Rng::new(5);
        let mut peak = 0;
        for token in 0..30 {
            engine.inject(token, &seeds(&mut rng, bank.len()));
            engine.step(&t, &mut bank);
            peak = peak.max(engine.live().len());
        }
        // Each token carries unit mass, so at most 1/floor of its particles can
        // sit above the floor at once; 30 tokens are in flight at the peak.
        assert!(peak <= 30 * (1.0 / p.mass_floor) as usize, "peak {peak}");
    }

    #[test]
    fn identical_under_any_thread_count() {
        // The determinism requirement of DESIGN.md §3, stated as a test. Same
        // seed, different thread pools, byte-identical outputs.
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| drive(10, 9, 20, params()))
        };
        let (a_out, a_visits) = run(1);
        let (b_out, b_visits) = run(8);
        assert_eq!(a_visits, b_visits, "visit histograms diverged");
        for (x, y) in a_out.iter().zip(&b_out) {
            assert_eq!(x.accumulated, y.accumulated, "reassembled payloads diverged");
            assert_eq!(x.absorbed_mass.to_bits(), y.absorbed_mass.to_bits());
            assert_eq!(x.hop_mass.to_bits(), y.hop_mass.to_bits());
        }
    }

    #[test]
    fn identical_across_repeated_runs() {
        let (a, av) = drive(8, 4, 16, params());
        let (b, bv) = drive(8, 4, 16, params());
        assert_eq!(av, bv);
        assert!(a.iter().zip(&b).all(|(x, y)| x.accumulated == y.accumulated));
    }

    #[test]
    fn load_stays_far_from_degenerate() {
        // A sanity floor, not a claim about homeostasis. Measured on a fresh
        // network with uniformly random ingress, the busiest tenth of nodes
        // takes ~16% of visits against 10% for perfect uniformity, and turning
        // homeostasis on or off moves that by under half a point. There is no
        // routing collapse to prevent in the untrained regime, so the value of
        // the mechanism cannot be tested here — only that it is wired in and
        // that nothing has already collapsed. See DESIGN.md §10.
        let (_, visits) = drive(10, 31, 120, params());
        let mut v = visits;
        v.sort_unstable();
        let total: u64 = v.iter().sum();
        assert!(total > 0);
        let cut = v.len() * 9 / 10;
        let top_decile = v[cut..].iter().sum::<u64>() as f64 / total as f64;
        assert!(top_decile < 0.35, "top decile takes {top_decile} of all visits");
        let idle = v.iter().filter(|c| **c == 0).count();
        assert!(idle < v.len() / 2, "{idle} of {} nodes never fired", v.len());
    }
}
