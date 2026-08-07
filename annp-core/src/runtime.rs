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

/// One token, after its particles have all come home.
#[derive(Clone, Copy, Debug)]
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
    pub mean_hops: f64,
    pub absorbed_mass: f64,
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
    pub fn new(
        topology: Topology,
        model_params: ModelParams,
        node_params: NodeParams,
        engine_params: EngineParams,
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
            bypass: false,
        }
    }

    /// Replaces the network's output with the input embedding, keeping every
    /// other part of the loop — routing, plasticity, update timing — unchanged.
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
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
            let shattered = self.model.shatter(self.topology.grid(), token);
            self.engine.inject(position, &shattered.seeds);
            self.pending.insert(position, (token, shattered.scale));
            self.stream.insert(position, token);
            self.position += 1;
        }
        self.engine.step(&self.topology, &mut self.bank);
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
            let output = self.engine.take_output(position).expect("settled without output");
            self.ready.insert(position, output);
        }

        let mut scored = Vec::new();
        while let Some(output) = self.ready.remove(&self.next_to_score) {
            let position = self.next_to_score;
            // The last token of a stream never gets a target; hold its output
            // in case more tokens arrive later.
            let Some(&target) = self.stream.get(&(position + 1)) else {
                self.ready.insert(position, output);
                break;
            };
            let (token, scale) = self.pending.remove(&position).expect("scored without pending");

            let assembled = if self.bypass {
                self.model.embedding_of(token).to_vec()
            } else {
                self.model.assemble(&output.accumulated, scale)
            };
            let passthrough_loss = {
                let e = self.model.embedding_of(token).to_vec();
                self.model.cross_entropy(&e, target)
            };
            let loss = self.model.learn(&assembled, target);

            scored.push(Scored {
                position,
                token,
                target,
                loss,
                passthrough_loss,
                visits: output.visits,
                mean_hops: output.mean_hops(),
                absorbed_mass: output.absorbed_mass,
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
    use crate::rng::Rng;

    fn build(vocab: usize, seed: u64) -> Runtime {
        let side = 16;
        let mut rng = Rng::new(seed);
        let topology = Topology::small_world(Grid::new(side), SmallWorld::default(), &mut rng);
        let schedule = Schedule::Geometric { r: 4.0, g1: 0.5 };
        Runtime::new(
            topology,
            ModelParams {
                vocab,
                d_head: 8,
                slots: 8,
                grid_side: side,
                schedule,
                embed_rungs: 3,
                learning_rate: 0.05,
            },
            NodeParams { d_head: 8, eta: 1.0, schedule, rungs: 4, homeostasis: 0.05 },
            EngineParams { top_k: 2, mass_floor: 1e-3, slots: 8 },
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
            assert_eq!(s.position as usize, i, "scoring must be strictly in stream order");
            assert_eq!(s.token, stream[i]);
            assert_eq!(s.target, stream[i + 1]);
        }
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
    fn per_token_compute_does_not_grow_with_position() {
        // DESIGN.md §1.6's central claim, as a test. Compare the visit count
        // charged to tokens early in a stream against tokens far later.
        let mut rt = build(48, 4);
        let stream: Vec<u32> = (0..1_200).map(|i| (i * 13 % 48) as u32).collect();
        let scored = feed(&mut rt, &stream);
        let mean = |s: &[Scored]| s.iter().map(|x| x.visits as f64).sum::<f64>() / s.len() as f64;
        // Skip the first few, where the network is still empty and everything
        // is absorbed immediately.
        let early = mean(&scored[100..300]);
        let late = mean(&scored[900..1_100]);
        assert!(
            (late / early - 1.0).abs() < 0.25,
            "compute per token drifted from {early} to {late}"
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

        let passthrough =
            scored[scored.len() - 200..].iter().map(|x| x.passthrough_loss).sum::<f64>() / 200.0;
        assert!(last < passthrough, "network scored {last} against passthrough {passthrough}");
    }
}
