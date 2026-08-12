//! The whole system, running.
//!
//! Ties the model to the engine: shatter a token into particles, let them
//! diffuse, and when every one of a token's particles has been absorbed,
//! reassemble the result and score it against the token that actually came
//! next.
//!
//! There is no training mode. A token is processed exactly once, scored once,
//! and learned from once, which makes the accumulated loss a *prequential*
//! quantity — every prediction is made before its target was ever seen, so it
//! is a valid compression bound with no train/test split to leak across. That
//! is not a convenience: DESIGN.md §5 makes it the evaluation protocol,
//! because an architecture with no separation between training and running has
//! nothing else it could honestly report.
//!
//! **Serialisation is not optional.** Node state is a shared mutable resource:
//! whichever particles land on a node write to its matrix, regardless of which
//! token they belong to. If token `p + 3` is injected while `p` is still in
//! flight, `p`'s output can be transformed by a matrix `p + 3` has already
//! written to, so the loss for predicting `p + 1` depends on `p + 3`. That is
//! not bidirectionality, it is reading the answer, and it destroys the
//! compression bound. `Mode::Serial` injects one token, lets it settle, scores
//! it, and only then injects the next.
//!
//! `Mode::Overlapped` keeps the old behaviour, not because it is defensible but
//! because the difference between the two *is* the size of the leak, and that
//! number has to be reported.
//!
//! What stays genuinely unordered is the fragment level: a token's `P`
//! particles diffuse concurrently, arrive in whatever order the routing
//! produces, and the node state they build imposes no order among them. Token
//! boundaries are ordered because the prediction task defines them that way —
//! asking "what follows `p`" is already an assertion that `p` precedes it.
//!
//! The arrival order itself is free. Any permutation gives a valid code length,
//! since the model predicts whatever is next *in arrival order*. Left-to-right
//! is the order that makes text predictable and a causal Transformer
//! comparable; an order that walks in from both ends and stops at `q` lets the
//! model condition on both sides of `q`, which a causal Transformer cannot do
//! at all. A uniformly random order is equally valid and equally useless — it
//! makes the next arrival unpredictable for any model, ours included.

use std::collections::BTreeMap;

use crate::engine::{Engine, EngineParams};
use crate::graph::Topology;
use crate::model::{Model, ModelParams};
use crate::node::{NodeBank, NodeParams};

/// How tokens are admitted relative to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One token in flight at a time. The only mode whose loss is a valid
    /// compression bound.
    Serial,
    /// One token injected per tick regardless of what is still in flight.
    /// Leaks future tokens into earlier predictions; kept only to measure by
    /// how much.
    Overlapped,
}

/// Representations captured along one token's path, for diagnostics.
///
/// Enabled with `Runtime::capture_representations`; `None` otherwise, so the
/// vectors cost nothing when nobody is probing.
#[derive(Clone, Debug)]
pub struct Representations {
    /// The token's embedding, before anything happened to it. Contains no
    /// information about earlier tokens by construction, so a probe on it is
    /// the chance-level control.
    pub ingress: Vec<f64>,
    /// Reassembly as it would have been after exactly one hop.
    pub after_one_hop: Vec<f64>,
    /// The reassembly the output head actually sees.
    pub assembled: Vec<f64>,
}

/// One token, after its particles have all come home.
#[derive(Clone, Debug)]
pub struct Scored {
    pub position: u32,
    pub token: u32,
    pub target: u32,
    /// Cross-entropy in nats, measured before the update it produced.
    pub loss: f64,
    /// Loss of the tied table reading the token's own embedding — what a
    /// perfectly transparent network would have scored. The network has to beat
    /// this to have contributed anything at all (DESIGN.md §11.2 ①).
    pub passthrough_loss: f64,
    pub visits: u64,
    /// Distinct nodes this token's mass passed through, ascending.
    pub touched: Vec<u32>,
    pub mean_hops: f64,
    pub absorbed_mass: f64,
    /// How far this token's ingress anchor moved when its resting place was
    /// recorded. A map that is forming has this decaying; one that is merely
    /// jittering does not.
    pub anchor_move: usize,
    pub reps: Option<Representations>,
}

pub struct Runtime {
    model: Model,
    topology: Topology,
    bank: NodeBank,
    engine: Engine,
    /// Positions in flight: `position -> (vocabulary id, shatter scale)`.
    pending: BTreeMap<u32, (u32, f64)>,
    /// Vocabulary id at each position still needed as somebody's target.
    stream: BTreeMap<u32, u32>,
    /// Settled outputs waiting for their turn to be scored.
    ready: BTreeMap<u32, crate::engine::TokenOutput>,
    /// The next position that may be scored. Scoring is strictly in order.
    next_to_score: u32,
    position: u32,
    mode: Mode,
    /// Whether long-range contacts are rewired as the run proceeds.
    turnover: bool,
    /// Take the first candidate drawn instead of the least-visited of several.
    /// The control that separates "turnover found useful structure" from
    /// "turnover randomised the graph, which happens to spread load".
    blind_turnover: bool,
    /// Its own generator, so enabling turnover does not shift any other stream.
    turnover_rng: crate::rng::Rng,
    /// Rewirings applied so far.
    rewirings: u64,
    /// Capture representations along each token's path for probing.
    capture: bool,
    /// Score the token's own embedding instead of what came back from the
    /// network, while leaving every other timing identical.
    ///
    /// This is the control that says what the network is worth. Without it the
    /// only baselines available are uniform and an *untrained* passthrough, and
    /// neither answers the question that matters: a tied output head trained on
    /// the current token's embedding is already a linear next-token predictor,
    /// and the network has to beat that, not the trivial baselines.
    bypass: bool,
}

impl Runtime {
    /// `structure_seed` is the one seed every structural stream derives from.
    /// `rng` initialises the model; turnover gets a stream of its own salted
    /// from the same seed, so that replicate runs vary *all* structural
    /// randomness rather than all of it but one.
    pub fn new(
        topology: Topology,
        model_params: ModelParams,
        node_params: NodeParams,
        engine_params: EngineParams,
        structure_seed: u64,
        rng: &mut crate::rng::Rng,
    ) -> Self {
        assert_eq!(
            model_params.grid_side,
            topology.grid().side(),
            "the model's grid side must match the topology it routes over"
        );
        assert_eq!(
            model_params.d_head, node_params.d_head,
            "payload width must agree between model and nodes"
        );
        assert_eq!(
            model_params.slots, engine_params.slots,
            "the engine's reassembly buffer must have one slot per particle"
        );
        let model = Model::new(model_params, rng);
        let bank = NodeBank::new(&topology, node_params);
        let engine = Engine::new(&bank, engine_params);
        Self {
            model,
            topology,
            bank,
            engine,
            pending: BTreeMap::new(),
            stream: BTreeMap::new(),
            ready: BTreeMap::new(),
            next_to_score: 0,
            position: 0,
            mode: Mode::Serial,
            turnover: true,
            blind_turnover: false,
            turnover_rng: crate::rng::Rng::new(structure_seed ^ 0x7C_9E_2B_41_A0_53_D6_18),
            rewirings: 0,
            capture: false,
            bypass: false,
        }
    }

    /// Replaces the network's output with the input embedding, keeping every
    /// other part of the loop — routing, plasticity, update timing — unchanged.
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    pub fn set_ingress_mode(&mut self, mode: crate::model::IngressMode) {
        self.model.set_ingress_mode(mode);
    }

    pub fn capture_representations(&mut self, capture: bool) {
        self.capture = capture;
    }

    pub fn set_turnover(&mut self, turnover: bool) {
        self.turnover = turnover;
    }

    pub fn set_blind_turnover(&mut self, blind: bool) {
        self.blind_turnover = blind;
    }

    #[inline]
    pub fn rewirings(&self) -> u64 {
        self.rewirings
    }

    /// Synaptic turnover: every node periodically repoints its least-used
    /// long-range contact at an under-used part of the grid.
    ///
    /// Three things make this fit rather than bolt on. Degree never changes, so
    /// the CSR needs no reallocation and per-hop compute is exactly constant —
    /// this reallocates a fixed budget, it does not grow one. New edges need no
    /// initialisation, because routing has no per-edge parameters left. And the
    /// target is chosen by *low traffic* rather than low surprise: an unvisited
    /// node publishes a zero expectation, which under the surprise rule clears
    /// the reference exactly when the source node is surprised, so a fresh
    /// contact opens only for the content its source could not place.
    ///
    /// The cadence is not a free parameter: a node rewires once it has been
    /// visited `d_head` times, which is how many patterns its matrix can hold.
    /// The number of candidates sampled is its own plastic degree.
    fn turn_over(&mut self) {
        let lattice = self.topology.lattice_degree();
        let cadence = self.bank.params().d_head as u64;
        let nodes = self.bank.len() as u32;
        for node in 0..nodes {
            let Some(visits) = self.bank.turnover_visits(node) else {
                continue;
            };
            if visits < cadence {
                continue;
            }
            let Some(slot) = self.bank.coldest_plastic_slot(node, lattice) else {
                continue;
            };
            let candidates = if self.blind_turnover {
                1
            } else {
                self.topology.plastic_slots(node).len().max(1)
            };
            let mut best: Option<(u32, u64)> = None;
            for _ in 0..candidates {
                let Some(v) = self.topology.sample_contact(node, &mut self.turnover_rng) else {
                    continue;
                };
                // Traffic, not surprise: low surprise means a node is already
                // satisfied, whereas low traffic means it has room.
                let load = self.engine.visits().get(v as usize).copied().unwrap_or(0);
                if best.is_none_or(|(_, b)| load < b) {
                    best = Some((v, load));
                }
            }
            if let Some((v, _)) = best {
                self.topology.rewire(node, slot, v);
            }
            self.bank.reset_turnover(node);
            self.rewirings += 1;
        }
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    #[inline]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    #[inline]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[inline]
    /// See `NodeBank::visit_stats`.
    pub fn visit_stats(&self) -> Option<(f64, f64)> {
        self.bank.visit_stats()
    }

    /// See `NodeBank::key_echo`.
    pub fn key_echo(&self) -> Option<(f64, f64)> {
        self.bank.key_echo()
    }

    /// See `Model::readout_coverage`.
    pub fn readout_coverage(&self) -> Option<f64> {
        self.model.readout_coverage()
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    #[inline]
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    #[inline]
    pub fn live_particles(&self) -> usize {
        self.engine.live().len()
    }

    /// Injects `token` if given, advances one tick, and returns whatever
    /// finished. Pass `None` to keep ticking without feeding anything new.
    pub fn advance(&mut self, token: Option<u32>) -> Vec<Scored> {
        if let Some(token) = token {
            let position = self.position;
            let shattered = self
                .model
                .shatter(self.topology.grid(), token, position as u64);
            self.engine.inject(position, &shattered.seeds);
            self.pending.insert(position, (token, shattered.scale));
            self.stream.insert(position, token);
            self.position += 1;
        }
        self.engine.step(&self.topology, &mut self.bank);
        if self.turnover {
            self.turn_over();
        }
        if self.mode == Mode::Serial {
            // Settle completely before anything else is admitted. The output
            // being scored is then a function of this token and its
            // predecessors only.
            let mut ticks = 0u64;
            while self.live_particles() > 0 {
                self.engine.step(&self.topology, &mut self.bank);
                ticks += 1;
                assert!(ticks < 100_000, "a single token failed to settle");
            }
        }
        self.collect()
    }

    /// Ticks until nothing is in flight, scoring as tokens land.
    pub fn drain(&mut self, max_ticks: u64) -> Vec<Scored> {
        let mut out = Vec::new();
        let mut ticks = 0;
        while self.live_particles() > 0 {
            out.extend(self.advance(None));
            ticks += 1;
            assert!(ticks < max_ticks, "still in flight after {max_ticks} ticks");
        }
        out
    }

    /// Scores settled positions, strictly in stream order.
    ///
    /// Positions do **not** settle in order: a token whose particles take more
    /// hops lands later. Scoring them as they land would be a leak, not just an
    /// untidiness — position `p`'s loss would be measured against a model that
    /// had already absorbed the update from `p + 1`, which depends on the token
    /// at `p + 2`. The prequential loss would then no longer be a compression
    /// bound, and the headline evaluation of DESIGN.md §5 would be quietly
    /// invalid. So settled outputs wait in `ready` until every earlier position
    /// has been scored.
    fn collect(&mut self) -> Vec<Scored> {
        for position in self.engine.settled() {
            let output = self
                .engine
                .take_output(position)
                .expect("settled without output");
            self.ready.insert(position, output);
        }

        let mut scored = Vec::new();
        while let Some(mut output) = self.ready.remove(&self.next_to_score) {
            let position = self.next_to_score;
            // The last token of a stream never gets a target; hold its output
            // in case more tokens arrive later.
            let Some(&target) = self.stream.get(&(position + 1)) else {
                self.ready.insert(position, output);
                break;
            };
            let (token, scale) = self
                .pending
                .remove(&position)
                .expect("scored without pending");

            let assembled = if self.bypass {
                self.model.embedding_of(token).to_vec()
            } else {
                self.model.assemble(&output.accumulated, scale)
            };
            let passthrough_loss = {
                let e = self.model.embedding_of(token).to_vec();
                self.model.cross_entropy(&e, target)
            };
            let loss = self.model.learn_centred(&assembled, token, target);

            // The anchor is a readout of the dynamics, not a second learner:
            // this token next enters wherever its own mass just came to rest.
            let side = self.topology.grid().side();
            let anchor_move = match output.resting_place(side) {
                Some(resting) => self.model.observe_resting_place(token, resting),
                None => 0,
            };

            let reps = self.capture.then(|| Representations {
                ingress: self.model.embedding_of(token).to_vec(),
                after_one_hop: self.model.assemble(&output.after_one_hop, scale),
                assembled: assembled.clone(),
            });
            scored.push(Scored {
                position,
                token,
                target,
                loss,
                passthrough_loss,
                visits: output.visits,
                touched: {
                    let mut t = std::mem::take(&mut output.touched);
                    t.sort_unstable();
                    t.dedup();
                    t
                },
                mean_hops: output.mean_hops(),
                absorbed_mass: output.absorbed_mass,
                anchor_move,
                reps,
            });
            self.next_to_score += 1;
        }

        // A position's id is the target of the position before it, so ids stay
        // until everything that could still want them has been scored.
        self.stream.retain(|k, _| *k >= self.next_to_score);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Grid, SmallWorld};
    use crate::ladder::Schedule;
    use crate::node::AbsorbRule;
    use crate::rng::Rng;

    fn build(vocab: usize, seed: u64) -> Runtime {
        let side = 16;
        let mut rng = Rng::new(seed);
        let topology = Topology::small_world(Grid::new(side), SmallWorld::default(), &mut rng);
        let schedule = Schedule::Geometric { r: 4.0, g1: 0.5 };
        Runtime::new(
            topology,
            ModelParams {
                centre_readout: false,
                head_top_k: 0,
                vocab,
                d_head: 8,
                slots: 8,
                grid_side: side,
                schedule,
                embed_rungs: 3,
                learning_rate: 0.05,
                tied: true,
                head_kind: crate::model::HeadKind::Linear,
            },
            NodeParams {
                absorb: AbsorbRule::RelativeSurprise,
                d_head: 8,
                eta: 1.0,
                schedule,
                rungs: 4,
                context_scales: 1,
                key_echo: 0,
            },
            EngineParams {
                mass_floor: 1e-3,
                slots: 8,
            },
            seed,
            &mut rng,
        )
    }

    /// Feeds a stream and returns every scored position.
    fn feed(rt: &mut Runtime, stream: &[u32]) -> Vec<Scored> {
        let mut out = Vec::new();
        for &t in stream {
            out.extend(rt.advance(Some(t)));
        }
        out.extend(rt.drain(10_000));
        out
    }

    #[test]
    fn every_token_is_scored_exactly_once_and_in_order() {
        let mut rt = build(32, 1);
        let stream: Vec<u32> = (0..200).map(|i| (i * 7 % 32) as u32).collect();
        let scored = feed(&mut rt, &stream);
        // The final position has no successor, so it is never scored.
        assert_eq!(scored.len(), stream.len() - 1);
        for (i, s) in scored.iter().enumerate() {
            assert_eq!(
                s.position as usize, i,
                "scoring must be strictly in stream order"
            );
            assert_eq!(s.token, stream[i]);
            assert_eq!(s.target, stream[i + 1]);
        }
    }

    #[test]
    #[ignore = "readout ingress is measured worse (§13); kept for the record"]
    fn anchors_move_without_collapsing() {
        // The readout's failure mode is every token migrating to one region;
        // routing homeostasis does not guard against it, so it is checked here.
        let mut rt = build(32, 11);
        rt.set_ingress_mode(crate::model::IngressMode::Readout);
        let stream: Vec<u32> = (0..400).map(|i| (i * 7 % 32) as u32).collect();
        feed(&mut rt, &stream);
        let (known, distinct, mean_move) = rt.model().ingress().drift();
        assert_eq!(known, 32, "every token should have come to rest somewhere");
        assert!(distinct >= 8, "anchors collapsed onto {distinct} cells");
        assert!(mean_move.is_finite());
    }

    #[test]
    fn fixed_ingress_anchors_never_move() {
        let mut rt = build(32, 11);
        rt.set_ingress_mode(crate::model::IngressMode::Content);
        let stream: Vec<u32> = (0..200).map(|i| (i * 7 % 32) as u32).collect();
        feed(&mut rt, &stream);
        let (known, _, mean_move) = rt.model().ingress().drift();
        assert_eq!(
            known, 0,
            "nothing should be remembered with the readout off"
        );
        assert_eq!(mean_move, 0.0);
    }

    #[test]
    fn turnover_preserves_degree_the_lattice_and_connectivity() {
        // The three invariants the whole scheme rests on. Degree fixed means
        // per-hop compute is exactly constant; the lattice intact means §9's
        // no-stall guarantee survives; reachable means we have not quietly
        // partitioned the graph, which is the failure this project has hit
        // before.
        let mut rt = build(32, 41);
        let before: Vec<usize> = (0..rt.topology().grid().len() as u32)
            .map(|n| rt.topology().degree(n))
            .collect();
        let stream: Vec<u32> = (0..1_500).map(|i| (i * 7 % 32) as u32).collect();
        feed(&mut rt, &stream);
        assert!(rt.rewirings() > 0, "no rewiring happened at all");

        let t = rt.topology();
        for node in 0..t.grid().len() as u32 {
            assert_eq!(
                t.degree(node),
                before[node as usize],
                "degree changed at {node}"
            );
            let lattice = t.grid().axis_neighbours(node);
            assert_eq!(
                &t.out_edges(node)[..4],
                &lattice,
                "lattice edge rewired at {node}"
            );
            assert!(!t.out_edges(node).contains(&node), "self-loop at {node}");
            let mut seen = t.out_edges(node).to_vec();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), t.degree(node), "duplicate edge at {node}");
        }
        assert_eq!(
            t.reachable_count(0),
            t.grid().len(),
            "turnover partitioned the graph"
        );
    }

    #[test]
    fn turnover_can_be_switched_off() {
        let mut rt = build(32, 41);
        rt.set_turnover(false);
        let before: Vec<u32> = (0..rt.topology().grid().len() as u32)
            .flat_map(|n| rt.topology().out_edges(n).to_vec())
            .collect();
        let stream: Vec<u32> = (0..1_000).map(|i| (i * 7 % 32) as u32).collect();
        feed(&mut rt, &stream);
        let after: Vec<u32> = (0..rt.topology().grid().len() as u32)
            .flat_map(|n| rt.topology().out_edges(n).to_vec())
            .collect();
        assert_eq!(before, after);
        assert_eq!(rt.rewirings(), 0);
    }

    #[test]
    fn every_token_recovers_all_of_its_mass() {
        let mut rt = build(32, 2);
        let stream: Vec<u32> = (0..120).map(|i| (i % 5) as u32).collect();
        for s in feed(&mut rt, &stream) {
            assert!(
                (s.absorbed_mass - 1.0).abs() < 1e-12,
                "position {} kept only {}",
                s.position,
                s.absorbed_mass
            );
        }
    }

    #[test]
    fn serial_mode_leaves_nothing_in_flight_between_tokens() {
        // The property the compression bound rests on: when a token is scored,
        // no later token has touched any node it visited, because no later
        // token exists yet.
        let mut rt = build(32, 3);
        assert_eq!(rt.mode(), Mode::Serial, "serial is the default");
        for i in 0..60u32 {
            rt.advance(Some(i % 32));
            assert_eq!(rt.live_particles(), 0, "token {i} was still in flight");
        }
    }

    #[test]
    fn overlapped_mode_really_does_overlap() {
        // Kept so the leak has something to be measured against.
        let mut rt = build(32, 3);
        rt.set_mode(Mode::Overlapped);
        let mut overlapped = false;
        for i in 0..60u32 {
            rt.advance(Some(i % 32));
            if rt.live_particles() > rt.model().params().slots {
                overlapped = true;
            }
        }
        assert!(overlapped, "overlapped mode kept only one token in flight");
        rt.drain(10_000);
    }

    #[test]
    fn per_token_compute_settles_rather_than_growing_with_position() {
        // DESIGN.md §1.6 claims per-token cost does not grow with sequence
        // length. Under the relative absorption rule there is now a genuine
        // warm-up: an untrained network absorbs everything immediately and
        // grows its paths as nodes learn, so cost rises before it settles.
        // The claim is about the settled regime, so both windows are taken
        // after the warm-up; the warm-up curve itself is reported by `annp run`.
        let mut rt = build(48, 4);
        let stream: Vec<u32> = (0..3_000).map(|i| (i * 13 % 48) as u32).collect();
        let scored = feed(&mut rt, &stream);
        let mean = |s: &[Scored]| s.iter().map(|x| x.visits as f64).sum::<f64>() / s.len() as f64;
        let middle = mean(&scored[1_400..1_700]);
        let late = mean(&scored[2_600..2_900]);
        assert!(
            (late / middle - 1.0).abs() < 0.25,
            "compute per token drifted from {middle} to {late} after warm-up"
        );
    }

    #[test]
    fn results_are_identical_across_thread_counts() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut rt = build(32, 5);
                    let stream: Vec<u32> = (0..150).map(|i| (i * 11 % 32) as u32).collect();
                    feed(&mut rt, &stream)
                        .iter()
                        .map(|s| (s.position, s.loss.to_bits(), s.visits))
                        .collect::<Vec<_>>()
                })
        };
        assert_eq!(run(1), run(8));
    }

    #[test]
    fn a_repeating_motif_is_learned() {
        // The weakest end-to-end claim that is still a claim: on a periodic
        // stream the loss must fall, and it must fall below what a transparent
        // network scores, since beating passthrough is the real bar.
        let mut rt = build(16, 6);
        let motif = [3u32, 9, 1, 14, 7];
        let stream: Vec<u32> = (0..1_500).map(|i| motif[i % motif.len()]).collect();
        let scored = feed(&mut rt, &stream);

        let mean = |s: &[Scored]| s.iter().map(|x| x.loss).sum::<f64>() / s.len() as f64;
        let first = mean(&scored[..200]);
        let last = mean(&scored[scored.len() - 200..]);
        assert!(last < first, "loss went {first} -> {last}");

        let passthrough = scored[scored.len() - 200..]
            .iter()
            .map(|x| x.passthrough_loss)
            .sum::<f64>()
            / 200.0;
        assert!(
            last < passthrough,
            "network scored {last} against passthrough {passthrough}"
        );
    }
}
