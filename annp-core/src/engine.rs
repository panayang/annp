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
    /// What the particle has gathered so far. Starts as the token's own chunk
    /// and takes in each node's answer.
    payload: Vec<f64>,
    /// What the particle is asking about. Written at injection and never
    /// touched again.
    ///
    /// The payload used to be both: a node read with a key derived from what
    /// had just passed through it, and the arrival's own direction was
    /// overwritten a little at every hop by the answers it collected. Three
    /// hops of that and the question is gone, which is where §36's reach comes
    /// from, and it also means two nodes are answering two different questions
    /// so averaging them blurs rather than ensembles. Kept separate, every node
    /// on a particle's route answers the same question and the answer can
    /// accumulate without eating the question.
    query: Vec<f64>,
    /// Hops remaining, as a fraction of the budget. Spent linearly rather than
    /// divided, so a particle that travels far is linearly cheaper rather than
    /// exponentially irrelevant — which is what made a branching front unable
    /// to use a long chord even when it reached one.
    fuel: Vec<f64>,
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
            query: Vec::new(),
            fuel: Vec::new(),
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
    pub fn query(&self, i: usize) -> &[f64] {
        &self.query[i * self.d_head..(i + 1) * self.d_head]
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
        self.query.clear();
        self.fuel.clear();
        self.mass.clear();
        self.token.clear();
        self.slot.clear();
        self.node.clear();
        self.serial.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        node: u32,
        token: u32,
        slot: u16,
        mass: f64,
        payload: &[f64],
        query: &[f64],
        fuel: f64,
        serial: u64,
    ) {
        debug_assert_eq!(payload.len(), self.d_head);
        debug_assert_eq!(query.len(), self.d_head);
        self.payload.extend_from_slice(payload);
        self.query.extend_from_slice(query);
        self.fuel.push(fuel);
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
    /// Every node this token's mass passed through, with repeats. Deduplicated
    /// by the caller.
    ///
    /// `visits` counts work; this says *where* the work happened. With roughly
    /// 540 visits on a 576-node grid the two questions come apart: a token can
    /// be sampling a selected few nodes many times, or smearing itself over the
    /// whole network. In the second case the reassembled vector is a
    /// network-wide average and cannot be specific to anything.
    pub touched: Vec<u32>,
    /// `sum_i mass_i * payload_i` over the children created on the token's
    /// very first tick, laid out like `accumulated`. Diagnostic only: it is
    /// what the reassembly would have been after exactly one hop, and comparing
    /// probes on it against probes on `accumulated` localises where context
    /// enters or is lost.
    pub after_one_hop: Vec<f64>,
    /// Mass-weighted **circular** sums of the grid coordinates where this
    /// token's mass came to rest, `[x, y]`.
    ///
    /// Circular because the grid is a torus. Averaging raw coordinates would
    /// put the mean of `{0, side-1}` in the middle of the grid instead of at
    /// the seam where it belongs — the classic wraparound bug, and the reason
    /// `resting_place` exists rather than a plain division.
    anchor_cos: f64,
    anchor_sin: f64,
    injected_tick: u64,
}

impl TokenOutput {
    /// Mass-weighted mean number of node visits before absorption. Subtract
    /// one to compare against an edge-counting figure.
    pub fn mean_hops(&self) -> f64 {
        if self.absorbed_mass > 0.0 {
            self.hop_mass / self.absorbed_mass
        } else {
            0.0
        }
    }

    /// Circular mean of where this token's mass was absorbed.
    ///
    /// `None` when the mass is spread evenly enough around either ring that the
    /// resultant vanishes and no direction is meaningful — a genuine
    /// degeneracy, not a tuned threshold.
    pub fn resting_place(&self, nodes: usize) -> Option<usize> {
        // Circular mean, because the ring wraps: averaging cell indices would
        // put mass split across the seam in the middle of the ring instead of
        // at the seam.
        if (self.anchor_cos * self.anchor_cos + self.anchor_sin * self.anchor_sin).sqrt() < 1e-12 {
            return None;
        }
        let theta = self.anchor_sin.atan2(self.anchor_cos);
        let unit = theta.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU;
        Some(((unit * nodes as f64).round() as usize) % nodes)
    }
}

/// How a particle spends its mass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    /// One child per edge, each carrying a share of the mass. Mass conserves,
    /// but a particle that travels `h` hops has divided its mass `h` times, so
    /// distance and weight are anti-correlated by construction and
    /// exponentially: at a peak routing weight of 0.5, eight hops leaves four
    /// parts in a thousand. On a torus where everything useful sat within three
    /// hops that coupling was invisible. On a ring whose chords put useful
    /// things eight hops away it is fatal — the far mass arrives, and its vote
    /// is negligible (DESIGN.md §38).
    Branch,
    /// One child, down whichever option the routing weights favour most,
    /// carrying the whole mass. The weights are the same ones `Branch` divides
    /// along; the only change is taking the largest rather than splitting among
    /// all, so this adds no rule and no constant. Distance now costs nothing in
    /// weight, `1 / mass_floor` no longer has to cap a branching front, and the
    /// process is an absorbing Markov chain in the literal sense rather than a
    /// branching one.
    Walk,
}

#[derive(Clone, Copy, Debug)]
pub struct EngineParams {
    /// Children below this mass are absorbed instead of forwarded, which caps
    /// the live particle count at `1 / mass_floor` per unit of injected mass.
    pub mass_floor: f64,
    /// Payload slots per token, needed to lay out the reassembly buffer.
    pub slots: usize,
    pub motion: Motion,
    /// Weight a deposit by the depositing node's confidence as well as its mass.
    ///
    /// Mass is a flow quantity — how much stopped here — and confidence is a
    /// relevance one. The readout has been averaging by flow, which is the
    /// mixture-of-experts arrangement with the gate score replaced by traffic.
    /// Carry an immutable query and a fuel tank on each particle; see
    /// `Swarm::query` and `Swarm::fuel`.
    pub carry_query: bool,
    pub confidence_weighted: bool,
    /// Deposit payload and node answer into separate halves; see
    /// `ModelParams::split_deposit`.
    pub split_deposit: bool,
    /// Hops after which a walker is absorbed wherever it stands.
    ///
    /// **Placeholder, not a design.** `Branch` terminates on its own because
    /// mass decays; a walk has no such guarantee and can circle. Every
    /// termination rule that suggests itself — stop on revisiting, stop when
    /// surprise stops falling — needs the particle to remember something, which
    /// is the fifth semantic field the design forbids. This cap exists so the
    /// decoupling can be measured before that question is settled, and any
    /// result that depends on its value is a result about the cap.
    pub hop_cap: u64,
}

/// One particle coming to rest, with everything the reassembly needs to weigh
/// it: what arrived, what the node answered, and how much of the arrival that
/// node's memory could account for.
/// One particle moving on: where to, and everything it carries.
struct Child {
    target: u32,
    token: u32,
    slot: u16,
    mass: f64,
    payload: Vec<f64>,
    query: Vec<f64>,
    fuel: f64,
}

struct Absorbed {
    token: u32,
    slot: u16,
    mass: f64,
    payload: Vec<f64>,
    answer: Vec<f64>,
    confidence: f64,
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
    children: Vec<Child>,
    /// `(token, slot, mass, payload)`
    absorbed: Vec<Absorbed>,
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
        assert!(
            params.mass_floor > 0.0,
            "mass floor must be positive or particles never die"
        );
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
        self.outputs
            .keys()
            .copied()
            .filter(|t| !live.contains(t))
            .collect()
    }

    pub fn take_output(&mut self, token: u32) -> Option<TokenOutput> {
        self.outputs.remove(&token)
    }

    /// Injects one token's shattered particles at the current tick.
    ///
    /// `seeds` is `(node, slot, mass, payload)`; masses should sum to one so
    /// that mass reads as a probability measure downstream.
    pub fn inject(&mut self, token: u32, seeds: &[(u32, u16, f64, Vec<f64>)]) {
        assert!(
            !self.outputs.contains_key(&token),
            "token {token} was already injected"
        );
        self.outputs.insert(
            token,
            TokenOutput {
                accumulated: vec![0.0; self.readout_width()],
                after_one_hop: vec![0.0; self.readout_width()],
                absorbed_mass: 0.0,
                hop_mass: 0.0,
                visits: 0,
                touched: Vec::new(),
                anchor_cos: 0.0,
                anchor_sin: 0.0,
                injected_tick: self.tick,
            },
        );
        for (node, slot, mass, payload) in seeds {
            assert!(
                (*slot as usize) < self.params.slots,
                "slot {slot} is out of range"
            );
            // The query starts as the token's own chunk: what a particle is
            // asking about is what it is.
            self.current.push(
                *node,
                token,
                *slot,
                *mass,
                payload,
                payload,
                1.0,
                self.next_serial,
            );
            self.next_serial += 1;
        }
    }

    /// Advances one tick. Returns statistics; `processed == 0` means the swarm
    /// is empty and every particle has been absorbed.
    pub fn step(&mut self, topology: &Topology, bank: &mut NodeBank) -> TickStats {
        if self.current.is_empty() {
            self.tick += 1;
            return TickStats {
                tick: self.tick,
                ..Default::default()
            };
        }

        // Charge this tick's work to the tokens that caused it.
        for (&token, &node) in self.current.token.iter().zip(&self.current.node) {
            if let Some(out) = self.outputs.get_mut(&token) {
                out.visits += 1;
                out.touched.push(node);
            }
        }

        // A hop's share of the fuel. The budget is `(ln N)^2`, the scale at
        // which greedy routing on a Kleinberg lattice at the critical exponent
        // delivers — §38.2 measured hops/(ln N)^2 between 0.47 and 0.52 — so it
        // is read off the topology rather than chosen, and a particle gets
        // roughly enough to cross the graph once.
        let budget = {
            let ln_n = (topology.ring().len() as f64).ln();
            (ln_n * ln_n).max(1.0)
        };
        let fuel_per_hop = 1.0 / budget;

        // Age of the particles about to move, in hops. Serial mode has one
        // token in flight and injects all of its particles on one tick, so the
        // whole swarm shares an age and it can be read off the tick instead of
        // riding along on every particle.
        let oldest = self
            .current
            .token
            .iter()
            .filter_map(|t| self.outputs.get(t))
            .map(|o| o.injected_tick)
            .min()
            .unwrap_or(self.tick);
        let age = self.tick.saturating_sub(oldest);

        // 1. Deterministic order: by node, then by creation rank.
        self.order.clear();
        self.order.extend(0..self.current.len() as u32);
        let (nodes_of, serial_of) = (&self.current.node, &self.current.serial);
        self.order
            .sort_unstable_by_key(|&i| (nodes_of[i as usize], serial_of[i as usize]));

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
                    let outcome =
                        node.step(&ctx, current.payload(i), current.query(i), &mut scratch);
                    out.surprise_sum += outcome.surprise;
                    out.visits += 1;

                    let (token, slot, mass) = (current.token[i], current.slot[i], current.mass[i]);
                    // Fuel is spent per hop, so a route that goes far costs
                    // linearly rather than dividing its own weight away. At zero
                    // the particle stops wherever it stands, which is the
                    // termination guarantee a walk otherwise lacks.
                    let query = current.query(i);
                    let spent = current.fuel[i] - fuel_per_hop;
                    let out_of_fuel = spent <= 0.0;
                    // The absorb option is the last weight, past every edge.
                    if params.motion == Motion::Walk {
                        // Serial mode injects a token's particles on one tick and
                        // advances them one hop per tick, so age is the hop count
                        // and no counter has to ride along on the particle.
                        let capped = age >= params.hop_cap || out_of_fuel;
                        let best = outcome
                            .weights
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(e, _)| e)
                            .unwrap_or(edges.len());
                        if capped || best >= edges.len() {
                            out.absorbed.push(Absorbed {
                                token,
                                slot,
                                mass,
                                payload: outcome.emitted.clone(),
                                answer: outcome.answer.clone(),
                                confidence: outcome.confidence,
                            });
                        } else {
                            out.children.push(Child {
                                target: edges[best],
                                token,
                                slot,
                                mass,
                                payload: outcome.emitted.clone(),
                                query: query.to_vec(),
                                fuel: spent,
                            });
                        }
                        continue;
                    }
                    if out_of_fuel && params.carry_query {
                        out.absorbed.push(Absorbed {
                            token,
                            slot,
                            mass,
                            payload: outcome.emitted.clone(),
                            answer: outcome.answer.clone(),
                            confidence: outcome.confidence,
                        });
                        continue;
                    }
                    let mut kept = 0.0;
                    for (e, &w) in outcome.weights[..edges.len()].iter().enumerate() {
                        if w <= 0.0 {
                            continue;
                        }
                        let child = mass * w;
                        if child < params.mass_floor {
                            // Too faint to be worth a hop; absorb it here so the
                            // mass is still accounted for.
                            out.absorbed.push(Absorbed {
                                token,
                                slot,
                                mass: child,
                                payload: outcome.emitted.clone(),
                                answer: outcome.answer.clone(),
                                confidence: outcome.confidence,
                            });
                        } else {
                            kept += child;
                            out.children.push(Child {
                                target: edges[e],
                                token,
                                slot,
                                mass: child,
                                payload: outcome.emitted.clone(),
                                query: query.to_vec(),
                                fuel: spent,
                            });
                        }
                    }
                    let _ = kept;
                    let absorb = mass * outcome.weights[edges.len()];
                    if absorb > 0.0 {
                        out.absorbed.push(Absorbed {
                            token,
                            slot,
                            mass: absorb,
                            payload: outcome.emitted.clone(),
                            answer: outcome.answer.clone(),
                            confidence: outcome.confidence,
                        });
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
        let nodes = topology.ring().len();
        let d_head = self.d_head;
        for g in &groups {
            self.visits[g.node as usize] += g.visits;
            surprise_sum += g.surprise_sum;
            for c in &g.children {
                if let Some(out) = self
                    .outputs
                    .get_mut(&c.token)
                    .filter(|o| self.tick == o.injected_tick)
                {
                    let lo = c.slot as usize * d_head;
                    for (a, &p) in out.after_one_hop[lo..lo + d_head]
                        .iter_mut()
                        .zip(&c.payload)
                    {
                        *a += c.mass * p;
                    }
                }
                self.next.push(
                    c.target,
                    c.token,
                    c.slot,
                    c.mass,
                    &c.payload,
                    &c.query,
                    c.fuel,
                    self.next_serial,
                );
                self.next_serial += 1;
                stats.forwarded_mass += c.mass;
                stats.spawned += 1;
            }
            for a in &g.absorbed {
                stats.absorbed_mass += a.mass;
                self.deposit(a, g.node as usize, nodes);
            }
        }
        stats.mean_surprise = if stats.processed > 0 {
            surprise_sum / stats.processed as f64
        } else {
            0.0
        };

        // 5. Republish expectations for the nodes that fired, so the next tick
        //    routes against a consistent snapshot.
        for g in &groups {
            bank.publish(g.node);
        }

        std::mem::swap(&mut self.current, &mut self.next);
        self.tick += 1;
        stats
    }

    /// Width of a reassembly buffer: one `d_model` block, or two when payload
    /// and answer are kept apart.
    fn readout_width(&self) -> usize {
        self.params.slots * self.d_head * if self.params.split_deposit { 2 } else { 1 }
    }

    fn deposit(&mut self, a: &Absorbed, at: usize, nodes: usize) {
        let d = self.d_head;
        let tick = self.tick;
        let split = self.params.split_deposit;
        let width = self.params.slots * d;
        // Confidence multiplies the mass rather than replacing it: mass is still
        // what conserves, and a node that explained nothing simply stops voting.
        let weight = if self.params.confidence_weighted {
            a.mass * a.confidence
        } else {
            a.mass
        };
        let (mass, token, slot, payload) = (a.mass, a.token, a.slot, &a.payload);
        let out = self
            .outputs
            .get_mut(&token)
            .expect("absorbed a particle of an unknown token");
        let lo = slot as usize * d;
        for (acc, &p) in out.accumulated[lo..lo + d].iter_mut().zip(payload) {
            *acc += weight * p;
        }
        if split {
            for (acc, &p) in out.accumulated[width + lo..width + lo + d]
                .iter_mut()
                .zip(&a.answer)
            {
                *acc += weight * p;
            }
        }
        out.absorbed_mass += mass;
        out.hop_mass += mass * (tick - out.injected_tick + 1) as f64;

        let theta = at as f64 * std::f64::consts::TAU / nodes as f64;
        out.anchor_cos += mass * theta.cos();
        out.anchor_sin += mass * theta.sin();
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
    use crate::graph::{Ring, SmallWorld, Topology};
    use crate::ladder::Schedule;
    use crate::node::{AbsorbRule, NodeParams};
    use crate::rng::Rng;

    const D: usize = 16;
    const SLOTS: usize = 4;

    fn fixture(side: usize, seed: u64) -> (Topology, NodeBank) {
        let mut rng = Rng::new(seed);
        let t = Topology::small_world(Ring::new(side), SmallWorld::default(), &mut rng);
        let bank = NodeBank::new(
            &t,
            NodeParams {
                absorb: AbsorbRule::Relative,
                d_head: D,
                eta: 1.0,
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
                context_scales: 1,
                key_echo: 0,
                read_rung: 0,
                query_read: false,
            },
        );
        (t, bank)
    }

    fn seeds(rng: &mut Rng, nodes: usize) -> Vec<(u32, u16, f64, Vec<f64>)> {
        (0..SLOTS)
            .map(|s| {
                let mut v = vec![0.0; D];
                rng.fill_unit_vector(&mut v);
                (
                    rng.next_below(nodes as u64) as u32,
                    s as u16,
                    1.0 / SLOTS as f64,
                    v,
                )
            })
            .collect()
    }

    fn params() -> EngineParams {
        EngineParams {
            mass_floor: 1e-3,
            slots: SLOTS,
            motion: Motion::Branch,
            carry_query: false,
            confidence_weighted: false,
            split_deposit: false,
            hop_cap: u64::MAX,
        }
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
        let outputs = (0..tokens)
            .map(|k| engine.take_output(k).unwrap())
            .collect();
        (outputs, engine.visits().to_vec())
    }

    #[test]
    fn carrying_a_query_changes_nothing_until_it_is_consulted() {
        // A particle carries a query and a fuel tank in every mode now; the flag
        // only decides whether they are read. If that is not exactly true, every
        // number recorded before they existed stops being comparable to one
        // recorded after.
        let base = EngineParams {
            mass_floor: 1e-3,
            slots: SLOTS,
            motion: Motion::Branch,
            hop_cap: u64::MAX,
            carry_query: false,
            confidence_weighted: false,
            split_deposit: false,
        };
        let (a, av) = drive(64, 5, 40, base);
        let (b, bv) = drive(64, 5, 40, base);
        let sum = |o: &[TokenOutput]| -> u64 {
            o.iter()
                .flat_map(|x| x.accumulated.iter())
                .fold(0.0f64, |s, v| s + v)
                .to_bits()
        };
        assert_eq!(sum(&a), sum(&b));
        assert_eq!(av, bv);

        // With it on, the run stays well formed: mass still conserves, and the
        // fuel tank means no particle can circle forever.
        let carried = EngineParams {
            carry_query: true,
            ..base
        };
        let (c, _) = drive(64, 5, 40, carried);
        for o in &c {
            assert!(
                (o.absorbed_mass - 1.0).abs() < 1e-6,
                "mass not conserved with a query carried: {}",
                o.absorbed_mass
            );
        }
    }

    #[test]
    fn a_resting_place_split_across_the_seam_lands_on_the_seam() {
        // Mass at cells 0 and N-1 is adjacent on a ring; an arithmetic mean of
        // the indices would report the opposite side instead.
        let nodes = 8usize;
        let mut out = TokenOutput {
            accumulated: vec![0.0; D * SLOTS],
            after_one_hop: vec![0.0; D * SLOTS],
            absorbed_mass: 0.0,
            hop_mass: 0.0,
            visits: 0,
            touched: Vec::new(),
            anchor_cos: 0.0,
            anchor_sin: 0.0,
            injected_tick: 0,
        };
        let step = std::f64::consts::TAU / nodes as f64;
        for (cell, mass) in [(0usize, 0.5), (nodes - 1, 0.5)] {
            let theta = cell as f64 * step;
            out.anchor_cos += mass * theta.cos();
            out.anchor_sin += mass * theta.sin();
        }
        let at = out
            .resting_place(nodes)
            .expect("resultant is not degenerate");
        let gap = |a: usize, b: usize| (a + nodes - b) % nodes;
        let to_seam = gap(at, 0).min(gap(0, at));
        assert!(
            to_seam <= 1,
            "mean landed at {at}, which is {to_seam} from the seam"
        );
    }

    #[test]
    fn a_resting_place_spread_evenly_round_the_ring_is_undefined() {
        let nodes = 8usize;
        let mut out = TokenOutput {
            accumulated: vec![0.0; D * SLOTS],
            after_one_hop: vec![0.0; D * SLOTS],
            absorbed_mass: 0.0,
            hop_mass: 0.0,
            visits: 0,
            touched: Vec::new(),
            anchor_cos: 0.0,
            anchor_sin: 0.0,
            injected_tick: 0,
        };
        let step = std::f64::consts::TAU / nodes as f64;
        for cell in 0..nodes {
            let theta = cell as f64 * step;
            out.anchor_cos += theta.cos() / nodes as f64;
            out.anchor_sin += theta.sin() / nodes as f64;
        }
        assert!(
            out.resting_place(nodes).is_none(),
            "a vanishing resultant has no direction"
        );
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
        let p = EngineParams {
            mass_floor: 1e-2,
            ..params()
        };
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
            assert_eq!(
                x.accumulated, y.accumulated,
                "reassembled payloads diverged"
            );
            assert_eq!(x.absorbed_mass.to_bits(), y.absorbed_mass.to_bits());
            assert_eq!(x.hop_mass.to_bits(), y.hop_mass.to_bits());
        }
    }

    #[test]
    fn identical_across_repeated_runs() {
        let (a, av) = drive(8, 4, 16, params());
        let (b, bv) = drive(8, 4, 16, params());
        assert_eq!(av, bv);
        assert!(
            a.iter()
                .zip(&b)
                .all(|(x, y)| x.accumulated == y.accumulated)
        );
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
        assert!(
            top_decile < 0.35,
            "top decile takes {top_decile} of all visits"
        );
        let idle = v.iter().filter(|c| **c == 0).count();
        assert!(
            idle < v.len() / 2,
            "{idle} of {} nodes never fired",
            v.len()
        );
    }
}
