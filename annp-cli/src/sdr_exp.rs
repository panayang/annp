//! Command-line experiment runner and benchmark suite for SDR Continual Learning.
//!
//! Evaluates 5 control arms under the unified input-weight multi-timescale architecture:
//! - Arm 1: SDR + Ladder-8 (m=8, activity-gated event diffusion, pure surface U1 readout)
//! - Arm 2: SDR + Ladder-4 (m=4, activity-gated event diffusion, pure surface U1 readout)
//! - Arm 3: SDR + Ladder-2 (m=2, activity-gated event diffusion, pure surface U1 readout)
//! - Arm 4: SDR + Plain (m=1, pure plastic weights)
//! - Arm 5: Online Proximal EWC
//!
//! Features:
//! 1. Hierarchical Scale-Free Zipf-Graph Random Walk Stream (power-law hubs, community clustering).
//! 2. Stratified Zipf Rank Evaluation (Top-20% Hubs, Mid-30% Domain, Tail-50% Leaves).
//! 3. Multi-Range Few-Shot Recovery evaluation (0, 5, 10, 20, 50, 200 steps).
//! 4. Forward Plasticity & Catastrophic Intransigence monitoring.
//! 5. Capacity-Contraction Stress Regime (high synaptic interference).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use annp_core::ladder::Schedule;
use annp_core::linalg::Mat;
use annp_core::rng::Rng;
use annp_core::sdr::{InputContextLadder, RandomProjection, SdrMemory};

/// Stream mode: Mode A (orthogonal disjoint domains) or Mode B (shared-entity semantic collision).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamMode {
    ModeA,
    ModeB,
}

impl StreamMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "b" | "mode_b" | "modeb" => StreamMode::ModeB,
            _ => StreamMode::ModeA,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            StreamMode::ModeA => "Mode A (Orthogonal Domains)",
            StreamMode::ModeB => "Mode B (Shared-Entity Semantic Collision)",
        }
    }
}

/// A relational triplet fact: `[entity, relation] -> target` with Zipf rank tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelationalFact {
    pub domain: usize,
    pub entity: usize,
    pub relation: usize,
    pub target: usize,
    pub rank_tier: usize, // 0: Hub (Top-20%), 1: Mid (30%), 2: Tail (50%)
}

/// Generator and container for multi-domain scale-free relational graphs and continuous random walk streams.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct RelationalFactStream {
    pub mode: StreamMode,
    pub domains: usize,
    pub facts_per_domain: usize,
    pub span_tokens: usize,
    pub rounds: usize,
    pub zipf_s: f64,
    pub hub_ratio: f64,
    pub vocab: usize,
    pub facts: Vec<Vec<RelationalFact>>, // [domain][probe_facts]
    pub walks: Vec<Vec<RelationalFact>>, // [domain][walk_steps]
    pub prefixes: Vec<Vec<usize>>,       // [domain][prefix_tokens]
    /// One token per domain, emitted at the start of that domain's walk.
    ///
    /// The benchmark has never required long-range association: a target is
    /// fully determined by the current (entity, relation), so nothing has to
    /// be carried across the stream and the architecture has never been
    /// penalised for carrying nothing. The cue makes regime identity available
    /// ONLY as a token seen at the start of the visit, so a probe that shows
    /// cue, then a gap, then the query, can only be answered by something that
    /// held the cue across the gap.
    pub cues: Vec<usize>,
}

impl RelationalFactStream {
    /// Constructs a stream generator under either Mode A (orthogonal) or Mode B (semantic collision).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: StreamMode,
        domains: usize,
        facts_per_domain: usize,
        span_tokens: usize,
        rounds: usize,
        zipf_s: f64,
        hub_ratio: f64,
        vocab: usize,
        seed: u64,
        target_overlap: f64,
        target_zipf: bool,
    ) -> Self {
        assert!(domains > 0, "domains must be positive");
        assert!(
            (0.0..=1.0).contains(&target_overlap),
            "target_overlap must be in [0, 1], got {target_overlap}"
        );
        assert!(
            target_overlap == 0.0 || matches!(mode, StreamMode::ModeB),
            "--target-overlap only affects Mode B, whose targets are otherwise \
             disjoint by construction. Mode A already shares its hub targets \
             (union/domains falls 97 -> 51 over twelve domains), so passing it \
             here would silently do nothing."
        );
        assert!(facts_per_domain >= 3, "facts_per_domain must be >= 3");
        assert!(span_tokens >= 3, "span_tokens must be >= 3");
        assert!(rounds > 0, "rounds must be positive");

        match mode {
            StreamMode::ModeA => Self::new_mode_a(
                domains,
                facts_per_domain,
                span_tokens,
                rounds,
                zipf_s,
                hub_ratio,
                vocab,
                seed,
                target_zipf,
            ),
            StreamMode::ModeB => Self::new_mode_b(
                domains,
                facts_per_domain,
                span_tokens,
                rounds,
                zipf_s,
                hub_ratio,
                vocab,
                seed,
                target_overlap,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_mode_a(
        domains: usize,
        facts_per_domain: usize,
        span_tokens: usize,
        rounds: usize,
        zipf_s: f64,
        hub_ratio: f64,
        vocab: usize,
        seed: u64,
        target_zipf: bool,
    ) -> Self {
        let total_entities = (domains * facts_per_domain).max(128);
        // Clamp bounds match section 2.1's stated range so a caller's value is
        // honoured rather than silently overridden. The old floor of 0.15
        // exceeded the CLI default of 0.10, so every run using the default
        // was actually running at 0.15 without any indication of it.
        let hub_count = ((total_entities as f64) * hub_ratio.clamp(0.05, 0.40))
            .round()
            .max(4.0) as usize;
        let domain_entity_count = (total_entities - hub_count) / domains;
        let rel_base = total_entities;
        let hub_rels = [rel_base, rel_base + 1]; // 2 global hub relations
        let rel_count = 2 + domains * 2; // 2 global + 2 per domain

        let required_vocab = total_entities + rel_count;
        assert!(
            vocab >= required_vocab,
            "vocab size {vocab} too small for entities ({total_entities}) + relations ({rel_count})"
        );

        let mut rng = Rng::new(seed);

        // Entity zipfian weights: w(e_i) = 1 / (i + 1)^s
        let entity_weights: Vec<f64> = (0..total_entities)
            .map(|i| 1.0 / ((i + 1) as f64).powf(zipf_s))
            .collect();
        let all_leaf_entities: Vec<usize> = (hub_count..total_entities).collect();
        let all_leaf_weights: Vec<f64> =
            all_leaf_entities.iter().map(|&e| entity_weights[e]).collect();
        let sum_all_leaf_w: f64 = all_leaf_weights.iter().sum();

        // 1. Construct Global Hub Edges (Tier 0: Hub -> Hub): Shared identically across all domains
        let hub_entities: Vec<usize> = (0..hub_count).collect();
        let hub_weights: Vec<f64> = hub_entities.iter().map(|&e| entity_weights[e]).collect();
        let sum_hub_w: f64 = hub_weights.iter().sum();

        let mut global_hub_edges = Vec::new();
        for &u in &hub_entities {
            for (rel_idx, &r) in hub_rels.iter().enumerate() {
                let mut pick_val = rng.next_f64() * sum_hub_w;
                let mut chosen_v = hub_entities[0];
                for (&v, &w) in hub_entities.iter().zip(&hub_weights) {
                    if pick_val <= w {
                        chosen_v = v;
                        break;
                    }
                    pick_val -= w;
                }
                if chosen_v == u && hub_entities.len() > 1 {
                    chosen_v = hub_entities[(rel_idx + 1) % hub_entities.len()];
                }
                global_hub_edges.push((u, r, chosen_v, 0)); // Tier 0
            }
        }

        let mut facts = Vec::with_capacity(domains);
        let mut walks = Vec::with_capacity(domains);
        let mut prefixes = Vec::with_capacity(domains);
        // Cue ids live above every token either mode allocates.
        let cues: Vec<usize> = (0..domains).map(|d| vocab - 1 - d).collect();

        for d in 0..domains {
            let spec_start = hub_count + d * domain_entity_count;
            let spec_end = spec_start + domain_entity_count;
            let domain_leaf_entities: Vec<usize> = (spec_start..spec_end).collect();
            let d_leaf_weights: Vec<f64> = domain_leaf_entities
                .iter()
                .map(|&e| entity_weights[e])
                .collect();
            let sum_leaf_w: f64 = d_leaf_weights.iter().sum();

            let domain_r_mid = rel_base + 2 + d * 2; // Leaf -> Hub
            let domain_r_tail = rel_base + 2 + d * 2 + 1; // Leaf -> Leaf

            // 2. Construct Mid Edges (Tier 1: Leaf -> Hub)
            let mut mid_edges = Vec::new();
            for &u in &domain_leaf_entities {
                let mut pick_val = rng.next_f64() * sum_hub_w;
                let mut chosen_v = hub_entities[0];
                for (&v, &w) in hub_entities.iter().zip(&hub_weights) {
                    if pick_val <= w {
                        chosen_v = v;
                        break;
                    }
                    pick_val -= w;
                }
                mid_edges.push((u, domain_r_mid, chosen_v, 1)); // Tier 1
            }

            // 3. Construct Tail Edges (Tier 2: Leaf -> Leaf)
            //
            // Tail targets were drawn from this domain's OWN leaves, so every
            // domain contributed a fixed quota of entirely fresh targets and
            // the union grew strictly linearly -- about 46 new per domain,
            // measured. Capacity that follows content then cannot grow
            // sublinearly whatever the mechanism does, and an earlier reading
            // of union/domains falling 97 -> 51 as "sublinear" was wrong: a
            // falling average with a constant increment is exactly linear.
            //
            // Real corpora are Zipf, so distinct types follow Heaps' law and
            // the new-type rate DECAYS. Drawing tail targets from the global
            // pool by Zipf weight restores that. This makes the source more
            // realistic rather than more permissive, which is the only reason
            // it is legitimate to change it.
            let (pool, pool_w, pool_sum): (&[usize], &[f64], f64) = if target_zipf {
                (&all_leaf_entities, &all_leaf_weights, sum_all_leaf_w)
            } else {
                (&domain_leaf_entities, &d_leaf_weights, sum_leaf_w)
            };
            let mut tail_edges = Vec::new();
            for (leaf_idx, &u) in domain_leaf_entities.iter().enumerate() {
                let mut pick_val = rng.next_f64() * pool_sum;
                let mut chosen_v = pool[0];
                for (&v, &w) in pool.iter().zip(pool_w) {
                    if pick_val <= w {
                        chosen_v = v;
                        break;
                    }
                    pick_val -= w;
                }
                if chosen_v == u && pool.len() > 1 {
                    chosen_v = pool[(leaf_idx + 1) % pool.len()];
                }
                tail_edges.push((u, domain_r_tail, chosen_v, 2)); // Tier 2
            }

            // 3b. Entry edges (Hub -> Leaf), without which the walk cannot exist.
            //
            // The other three sets are Hub->Hub, Leaf->Hub and Leaf->Leaf, so no
            // edge leads from the backbone into a domain's leaves. The walk
            // starts at a hub and follows targets, which meant it was absorbed
            // in the hub component permanently: mid and tail probe facts were
            // presented exactly zero times, every domain traversed the same
            // shared hub subgraph, and accuracy sat at the one third of the
            // probe set that had been seen at all. Measured before this: domain
            // activation overlap 0.973, 11.9% of neurons ever used, accuracy
            // 0.335-0.365 against a structural ceiling of 0.333.
            //
            // Section 2.2 of DESIGN-SDR describes a walker spending about half
            // its time on the backbone and half among the domain's leaves, which
            // requires this set to exist. The relation symbol is reused from the
            // mid tier, so no new vocabulary is introduced and no (entity,
            // relation) pair gains a second target.
            let mut entry_edges = Vec::new();
            for &u in &hub_entities {
                let mut pick_val = rng.next_f64() * sum_leaf_w;
                let mut chosen_v = domain_leaf_entities[0];
                for (&v, &w) in domain_leaf_entities.iter().zip(&d_leaf_weights) {
                    if pick_val <= w {
                        chosen_v = v;
                        break;
                    }
                    pick_val -= w;
                }
                entry_edges.push((u, domain_r_mid, chosen_v, 1));
            }

            // All valid edges in Domain d:
            let mut all_domain_edges = Vec::new();
            all_domain_edges.extend_from_slice(&global_hub_edges);
            all_domain_edges.extend_from_slice(&entry_edges);
            all_domain_edges.extend_from_slice(&mid_edges);
            all_domain_edges.extend_from_slice(&tail_edges);

            // Construct strictly balanced probe facts for Domain d:
            let n_hub_probe = (facts_per_domain / 3).max(1);
            let n_mid_probe = (facts_per_domain / 3).max(1);
            let n_tail_probe = facts_per_domain - n_hub_probe - n_mid_probe;

            let mut p_facts = Vec::with_capacity(facts_per_domain);
            for &(u, r, v, tier) in global_hub_edges.iter().take(n_hub_probe) {
                p_facts.push(RelationalFact {
                    domain: d,
                    entity: u,
                    relation: r,
                    target: v,
                    rank_tier: tier,
                });
            }
            for &(u, r, v, tier) in mid_edges.iter().take(n_mid_probe) {
                p_facts.push(RelationalFact {
                    domain: d,
                    entity: u,
                    relation: r,
                    target: v,
                    rank_tier: tier,
                });
            }
            for &(u, r, v, tier) in tail_edges.iter().take(n_tail_probe) {
                p_facts.push(RelationalFact {
                    domain: d,
                    entity: u,
                    relation: r,
                    target: v,
                    rank_tier: tier,
                });
            }

            // Continuous biased random walk in Domain d:
            let walk_steps = span_tokens / 3;
            let mut walk = Vec::with_capacity(walk_steps);
            let mut curr_node = hub_entities[0];

            for _ in 0..walk_steps {
                let outgoing: Vec<_> = all_domain_edges
                    .iter()
                    .filter(|(u, _, _, _)| *u == curr_node)
                    .collect();
                let edge = if !outgoing.is_empty() {
                    let idx = rng.next_below(outgoing.len() as u64) as usize;
                    *outgoing[idx]
                } else {
                    let idx = rng.next_below(all_domain_edges.len() as u64) as usize;
                    all_domain_edges[idx]
                };

                walk.push(RelationalFact {
                    domain: d,
                    entity: edge.0,
                    relation: edge.1,
                    target: edge.2,
                    rank_tier: edge.3,
                });

                curr_node = edge.2;
            }

            facts.push(p_facts);
            walks.push(walk);
            prefixes.push(Vec::new()); // Mode A does not require prefix conditioning
        }

        Self {
            mode: StreamMode::ModeA,
            domains,
            facts_per_domain,
            span_tokens,
            rounds,
            zipf_s,
            hub_ratio,
            vocab,
            facts,
            walks,
            prefixes,
            cues,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_mode_b(
        domains: usize,
        facts_per_domain: usize,
        span_tokens: usize,
        rounds: usize,
        zipf_s: f64,
        hub_ratio: f64,
        vocab: usize,
        seed: u64,
        target_overlap: f64,
    ) -> Self {
        let shared_entities_count = facts_per_domain.max(32);
        let shared_rel_count = 4;
        let rel_base = shared_entities_count;
        let targets_base = rel_base + shared_rel_count;
        // One shared pool ahead of the per-domain blocks.
        let private_base = targets_base + shared_entities_count;
        let total_required = private_base + domains * shared_entities_count;

        assert!(
            vocab >= total_required,
            "vocab {vocab} too small for Mode B (needs {total_required})"
        );

        // Stream (entities, edges, probe facts, walk) was hardcoded to seed
        // 20260817 regardless of --seed, so every "multi-seed" run so far
        // would have varied only embeddings and weight init while probing the
        // exact same facts with the exact same walk. That matters specifically
        // because tail_0shot is a small-sample statistic (~136 facts at
        // facts_per_domain=100) where a handful of facts changes the number by
        // several points -- multi-seeding without this fix would not have
        // averaged out that noise at all.
        let mut rng = Rng::new(seed);

        let mut facts = Vec::with_capacity(domains);
        let mut walks = Vec::with_capacity(domains);
        let mut prefixes = Vec::with_capacity(domains);
        let cues: Vec<usize> = (0..domains).map(|d| vocab - 1 - d).collect();

        // Was hardcoded to 0.20 regardless of what was passed in, so
        // --hub-ratio was a no-op for Mode B: both benches run before this fix
        // produced byte-identical results across a change that only touched
        // hub_ratio, because Mode B never saw it. Clamp matches Mode A's.
        let hub_count = (shared_entities_count as f64 * hub_ratio.clamp(0.05, 0.40))
            .round()
            .max(2.0) as usize;
        let mid_count = (shared_entities_count as f64 * 0.30).round().max(2.0) as usize;

        for d in 0..domains {
            let mut domain_facts = Vec::with_capacity(facts_per_domain);
            let mut domain_all_edges = Vec::new();

            for e in 0..shared_entities_count {
                let rank_tier = if e < hub_count {
                    0 // Hub
                } else if e < hub_count + mid_count {
                    1 // Mid
                } else {
                    2 // Tail
                };

                let rel = rel_base + (e % shared_rel_count);
                // Targets were disjoint across domains by construction -- the
                // `d * count` term gave every domain its own block -- so the
                // union of targets grew exactly linearly (measured: 150 new
                // per domain, zero overlap at 24 domains). Nothing at the
                // target level could ever be amortised, which is why savings
                // could not appear regardless of the architecture.
                //
                // `target_overlap` is the fraction of facts whose target is
                // drawn from a pool common to every domain, so the same
                // (entity, relation) means the same thing everywhere. It
                // interpolates between the adversarial corner (0: domains
                // share surface form and nothing else) and redundancy (1).
                // Real streams sit inside that range; the benchmark sat at the
                // endpoint.
                let shared_target = ((e * 17 + 5) % 1000) < (target_overlap * 1000.0) as usize;
                let target = if shared_target {
                    targets_base + ((e * 7 + 13) % shared_entities_count)
                } else {
                    private_base
                        + d * shared_entities_count
                        + ((e * 7 + 13 + d * 31) % shared_entities_count)
                };

                let fact = RelationalFact {
                    domain: d,
                    entity: e,
                    relation: rel,
                    target,
                    rank_tier,
                };
                domain_facts.push(fact);
                domain_all_edges.push((e, rel, target, rank_tier));
            }

            // Continuous walk in Domain d with Zipfian selection:
            let walk_steps = span_tokens / 3;
            let mut walk = Vec::with_capacity(walk_steps);

            let entity_weights: Vec<f64> = (0..shared_entities_count)
                .map(|i| 1.0 / ((i + 1) as f64).powf(zipf_s))
                .collect();
            let sum_w: f64 = entity_weights.iter().sum();

            for _ in 0..walk_steps {
                let mut pick = rng.next_f64() * sum_w;
                let mut chosen_e = 0;
                for (idx, &w) in entity_weights.iter().enumerate() {
                    if pick <= w {
                        chosen_e = idx;
                        break;
                    }
                    pick -= w;
                }
                walk.push(domain_facts[chosen_e]);
            }

            // Construct 16-token domain prefix sequence:
            let mut prefix = Vec::with_capacity(16);
            for fact in walk.iter().take(6) {
                prefix.push(fact.entity);
                prefix.push(fact.relation);
                prefix.push(fact.target);
            }
            prefix.truncate(16);

            facts.push(domain_facts);
            walks.push(walk);
            prefixes.push(prefix);
        }

        Self {
            mode: StreamMode::ModeB,
            domains,
            facts_per_domain,
            span_tokens,
            rounds,
            zipf_s,
            hub_ratio,
            vocab,
            facts,
            walks,
            prefixes,
            cues,
        }
    }
}

/// A zero-learning-rate probe evaluation slice.
pub struct FrozenProbeSlice;

/// Results of a single frozen probe evaluation across a domain fact set.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct ProbeResult {
    pub domain: usize,
    pub accuracy: f64,
    pub acc_hub: f64,
    pub acc_mid: f64,
    pub acc_tail: f64,
    pub mean_loss: f64,
    /// How many distinct targets the arm actually predicted across the whole
    /// probe set. A model whose ranking is dominated by a component that is
    /// constant within a domain -- the restored context is identical for
    /// every fact probed in that domain -- predicts the same target for all
    /// of them and scores near zero while its loss still falls, because the
    /// probability mass on the right answer rises without ever taking the
    /// top slot. Accuracy alone cannot tell that apart from "the mechanism
    /// is weak"; this can.
    pub distinct_predictions: usize,
    /// Mean loss restricted to mid+tail facts -- the domain-specific ones.
    /// Hub facts (Tier 0) are shared identically across every domain in
    /// Mode A, so other domains keep training them while this domain is
    /// "away"; a retention probe that includes hub facts is not measuring
    /// forgetting for roughly a third of its own sample.
    pub mean_loss_domain_specific: f64,
    pub count: usize,
}

/// Results of Multi-Range Few-Shot recovery evaluation upon revisiting a domain.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct FewShotResult {
    pub domain: usize,
    pub acc_0shot: f64,
    pub loss_0shot: f64,
    pub acc_few5: f64,
    pub acc_few10: f64,
    pub acc_few20: f64,
    pub acc_few50: f64,
    pub acc_few200: f64,
    pub spec_hub: [f64; 6],
    pub spec_mid: [f64; 6],
    pub spec_tail: [f64; 6],
}

impl FrozenProbeSlice {
    /// Evaluates accuracy and cross-entropy loss over a set of relational facts without modifying memory state.
    /// `ctx` must already be sitting in the probed domain's context -- see
    /// `InputContextLadder::snapshot_state`. Each fact is scored from that
    /// state plus its own entity and relation, so the slow rungs carry the
    /// domain history the model was trained under instead of the 18 steps a
    /// reset-and-replay-a-short-prefix probe could deliver.
    pub fn evaluate(
        facts: &[RelationalFact],
        ctx: &InputContextLadder,
        bank: &mut ExpertBank,
    ) -> ProbeResult {
        Self::evaluate_with_episode(facts, ctx, bank, None, 0)
    }

    /// Scores `facts`, optionally rebuilding the context from a presented
    /// episode instead of a restored per-domain snapshot.
    ///
    /// The snapshot is keyed by the true domain index, so any arm that reads
    /// the input trace would be handed its regime identity at evaluation time
    /// -- the same shape of leak as s29.1, which only stayed harmless because
    /// the edge arm ignored the trace. With an episode the context comes from
    /// what the model was shown: cue, then `gap` filler tokens, then the
    /// query. Nothing but something that carried the cue across the gap can
    /// answer, and no task id is involved anywhere.
    pub fn evaluate_with_episode(
        facts: &[RelationalFact],
        ctx: &InputContextLadder,
        bank: &mut ExpertBank,
        cue: Option<usize>,
        gap: usize,
    ) -> ProbeResult {
        let mut predicted: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut correct = 0;
        let mut correct_hub = 0;
        let mut count_hub = 0;
        let mut correct_mid = 0;
        let mut count_mid = 0;
        let mut correct_tail = 0;
        let mut count_tail = 0;

        let mut total_loss = 0.0;
        let mut domain_specific_loss = 0.0;
        let mut domain_specific_count = 0usize;
        let mut online_ladder = ctx.clone();
        // With an episode the probe builds its own context and the incoming
        // snapshot is deliberately unused.
        let snap = if let Some(c) = cue {
            let mut ep = ctx.clone();
            ep.reset();
            ep.step(c);
            for i in 0..gap {
                // Fillers must carry no regime information of their own.
                ep.step(i % 8);
            }
            ep.snapshot_state()
        } else {
            ctx.snapshot_state()
        };

        for fact in facts {
            online_ladder.restore_state(&snap);
            online_ladder.step(fact.entity);
            online_ladder.step(fact.relation);

            let (loss, is_correct) = bank.predict_fact(
                fact.entity,
                fact.relation,
                online_ladder.normalized_trace(),
                fact.target,
            );
            bank.note_consistency(fact.entity, fact.relation);
            predicted.insert(bank.last_argmax());
            if is_correct {
                correct += 1;
                match fact.rank_tier {
                    0 => correct_hub += 1,
                    1 => correct_mid += 1,
                    _ => correct_tail += 1,
                }
            }
            match fact.rank_tier {
                0 => count_hub += 1,
                1 => count_mid += 1,
                _ => count_tail += 1,
            }
            if fact.rank_tier != 0 {
                domain_specific_loss += loss;
                domain_specific_count += 1;
            }
            total_loss += loss;
        }

        let count = facts.len().max(1);
        let accuracy = correct as f64 / count as f64;
        let acc_hub = if count_hub > 0 {
            correct_hub as f64 / count_hub as f64
        } else {
            accuracy
        };
        let acc_mid = if count_mid > 0 {
            correct_mid as f64 / count_mid as f64
        } else {
            accuracy
        };
        let acc_tail = if count_tail > 0 {
            correct_tail as f64 / count_tail as f64
        } else {
            accuracy
        };
        let mean_loss = total_loss / count as f64;
        let mean_loss_domain_specific = if domain_specific_count > 0 {
            domain_specific_loss / domain_specific_count as f64
        } else {
            mean_loss
        };
        let domain = facts.first().map(|f| f.domain).unwrap_or(0);

        ProbeResult {
            domain,
            accuracy,
            acc_hub,
            acc_mid,
            acc_tail,
            mean_loss,
            mean_loss_domain_specific,
            distinct_predictions: predicted.len(),
            count,
        }
    }

    /// Evaluates multi-timescale few-shot recovery curve (0, 5, 10, 20, 50, 200 steps)
    /// across stratified Zipf rank tiers on isolated clones of memory state.
    pub fn evaluate_few_shot(
        facts: &[RelationalFact],
        ctx: &InputContextLadder,
        bank: &ExpertBank,
        eta: f64,
        seed: u64,
    ) -> FewShotResult {
        let mut sim_rng = Rng::new(seed);
        let base_probe = Self::evaluate(facts, ctx, &mut bank.clone());
        let snap = ctx.snapshot_state();

        let budgets = [5, 10, 20, 50, 200];
        let mut few_accs = [base_probe.accuracy; 5];
        let mut few_hub = [base_probe.acc_hub; 5];
        let mut few_mid = [base_probe.acc_mid; 5];
        let mut few_tail = [base_probe.acc_tail; 5];

        for (idx, &budget) in budgets.iter().enumerate() {
            let mut temp_bank = bank.clone();
            let mut online_ladder = ctx.clone();

            for _ in 0..budget {
                let fact_idx = sim_rng.next_below(facts.len() as u64) as usize;
                let fact = facts[fact_idx];

                online_ladder.restore_state(&snap);
                online_ladder.step(fact.entity);
                online_ladder.step(fact.relation);

                temp_bank.observe_fact(
                    fact.entity,
                    fact.relation,
                    online_ladder.normalized_trace(),
                    fact.target,
                    eta,
                );
            }

            let probe = Self::evaluate(facts, ctx, &mut temp_bank);
            few_accs[idx] = probe.accuracy;
            few_hub[idx] = probe.acc_hub;
            few_mid[idx] = probe.acc_mid;
            few_tail[idx] = probe.acc_tail;
        }

        FewShotResult {
            domain: facts.first().map(|f| f.domain).unwrap_or(0),
            acc_0shot: base_probe.accuracy,
            loss_0shot: base_probe.mean_loss,
            acc_few5: few_accs[0],
            acc_few10: few_accs[1],
            acc_few20: few_accs[2],
            acc_few50: few_accs[3],
            acc_few200: few_accs[4],
            spec_hub: [
                base_probe.acc_hub,
                few_hub[0],
                few_hub[1],
                few_hub[2],
                few_hub[3],
                few_hub[4],
            ],
            spec_mid: [
                base_probe.acc_mid,
                few_mid[0],
                few_mid[1],
                few_mid[2],
                few_mid[3],
                few_mid[4],
            ],
            spec_tail: [
                base_probe.acc_tail,
                few_tail[0],
                few_tail[1],
                few_tail[2],
                few_tail[3],
                few_tail[4],
            ],
        }
    }
}

/// Identifiers for the 5 experimental control arms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArmKind {
    Ladder8,
    Ladder4,
    Ladder2,
    Plain,
    /// No projection, no k-WTA, no sparse addressing: a linear softmax model
    /// read straight off the same input trace. This is the baseline the
    /// "Plain" arm is not -- Plain still carries the 8-rung input ladder, the
    /// calibrated projection, the top-k code and the cosine weights, and only
    /// drops the weight-side consolidation ladder. Linear drops all of it and
    /// uses HALF the parameters (V x m_in*d_input against V x d_sdr).
    Linear,
    /// As `Linear`, but reading only rung 0 -- roughly "the last token" as
    /// context. Isolates what the eight input timescales are worth.
    LinearFast,
    ProximalEwc { lambda: f64 },
}

impl ArmKind {
    pub fn name(&self) -> String {
        match self {
            ArmKind::Ladder8 => "Ladder-8 (m=8)".to_string(),
            ArmKind::Ladder4 => "Ladder-4 (m=4)".to_string(),
            ArmKind::Ladder2 => "Ladder-2 (m=2)".to_string(),
            ArmKind::Plain => "Plain (m=1)".to_string(),
            ArmKind::Linear => "Linear (no SDR)".to_string(),
            ArmKind::LinearFast => "Linear-fast (rung 0)".to_string(),
            ArmKind::ProximalEwc { lambda } => format!("Proximal EWC (lam={:.1})", lambda),
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            ArmKind::Ladder8 => "ladder_8",
            ArmKind::Ladder4 => "ladder_4",
            ArmKind::Ladder2 => "ladder_2",
            ArmKind::Plain => "plain_m1",
            ArmKind::Linear => "linear",
            ArmKind::LinearFast => "linear_fast",
            ArmKind::ProximalEwc { .. } => "proximal_ewc",
        }
    }
}

/// Trajectory record for a single probe observation.
#[derive(Clone, Debug)]
pub struct TrajectoryRecord {
    pub arm_name: String,
    pub arm_short: String,
    pub eta: f64,
    pub ewc_lambda: f64,
    pub round: usize,
    pub domain: usize,
    pub pre_revisit_acc: f64,
    pub pre_revisit_loss: f64,
    pub acc_few5: f64,
    pub acc_few10: f64,
    pub acc_few20: f64,
    pub acc_few50: f64,
    pub acc_few200: f64,
    pub spec_hub: [f64; 6],
    pub spec_mid: [f64; 6],
    pub spec_tail: [f64; 6],
    pub post_train_acc: f64,
    /// Entropy of the node-visit distribution in bits, and the maximum the
    /// topology allows. Zero for the non-topological arms. Printed with the
    /// results rather than checked afterwards: a run whose routing collapsed
    /// still yields a perfectly plausible accuracy number, so the mechanism
    /// has to be shown present in the same table that reports the outcome.
    /// Fraction of probed facts that route back to the node training left
    /// them at. Spread without this is meaningless: a code that scatters
    /// every fact to a fresh node each time scores maximal entropy and
    /// retrieves nothing.
    pub routing_consistency: f64,
    pub class_switch_rate: f64,
    pub class_collision: f64,
    pub classes_live: f64,
    pub growth_steal: f64,
    pub path_entropy_bits: f64,
    pub path_entropy_max: f64,
    /// Distinct targets predicted across the probe set, from the
    /// post-training probe. Collapse to a handful means the ranking is
    /// driven by something constant within the domain, not by the fact.
    pub post_distinct_predictions: usize,
    pub post_train_loss: f64,
    pub retention_gap: f64,
}

/// Aggregate summary metrics for an arm across all rounds.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ArmSummary {
    pub arm: ArmKind,
    pub best_eta: f64,
    pub best_ewc_lambda: f64,
    pub eta_boundary_hit: bool,
    pub ewc_zero_lambda_won: bool,
    pub final_retention_acc: f64,
    pub final_retention_loss: f64,
    pub final_few_spectrum: [f64; 6],  // 0, 5, 10, 20, 50, 200
    pub final_hub_spectrum: [f64; 6],  // Top-20% Hubs
    pub final_mid_spectrum: [f64; 6],  // Mid-30% Domain
    pub final_tail_spectrum: [f64; 6], // Tail-50% Leaves
    pub r1_post_acc: f64,
    pub final_post_acc: f64,
    pub plasticity_ratio: f64,
    pub mean_retention_gap: f64,
    pub round_retention_accs: Vec<f64>,
    pub round_retention_losses: Vec<f64>,
    pub trajectories: Vec<TrajectoryRecord>,
}

/// Configuration parameters for SDR experiments.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SdrConfig {
    pub mode: StreamMode,
    pub domains: usize,
    pub facts_per_domain: usize,
    pub span_tokens: usize,
    pub rounds: usize,
    pub vocab: usize,
    pub d_input: usize,
    pub m_in: usize,
    pub d_sdr: usize,
    pub k_active: usize,
    pub ladder_r: f64,
    pub zipf_s: f64,
    pub hub_ratio: f64,
    pub eta: Option<f64>,
    pub ewc_lambda: Option<f64>,
    pub seed: u64,
    /// Number of product-of-experts groups the input rungs are split across.
    /// 1 reproduces the single-projection architecture exactly.
    pub experts: usize,
    /// Skip the EWC sweep entirely. It is two thirds of the wall clock and
    /// contributes nothing to a content-vs-context question.
    /// Fraction of facts whose target is common to every domain.
    pub target_overlap: f64,
    /// Draw Mode A tail targets from the global Zipf pool, so distinct types
    /// follow Heaps' law as in real corpora rather than a fixed fresh quota.
    pub target_zipf: bool,
    /// Tokens between the cue and the query in the probe episode. 0 keeps the
    /// old protocol, in which context is restored from a per-domain snapshot.
    pub long_range_gap: usize,
    /// Cycle the probe gap over several horizons within one run.
    pub long_range_mix: bool,
    pub no_ewc: bool,
    /// Explicit eta grid, for cutting the seven-point default down when the
    /// useful range is already known from a previous sweep.
    pub etas: Option<Vec<f64>>,
    /// Weight-ladder base conductance. tau_k = r^(2k) / g1, so this sets the
    /// whole timescale range. It has to be matched to the period of whatever
    /// the stream asks the ladder to hold: a rung whose tau exceeds one
    /// domain visit averages across domains, which is exactly what destroys
    /// a Mode B mapping. Stability needs g1 < 1 (rung 0's outflow rate).
    pub ladder_g1: f64,
    /// Context-axis width for outer-product addressing. 0 disables it and
    /// leaves the flat/product-of-experts addressing in place.
    pub tensor_d2: usize,
    /// Active columns on the context axis under outer-product addressing.
    pub tensor_k2: usize,
    /// Rung index where context begins; content is [0, split), context is
    /// [split, m_in). 0 means m_in / 2.
    pub tensor_split: usize,
    /// Enable context-conditioned rotation of the content code.
    pub rotate: bool,
    /// Topological distributed memory: node count (0 = off) and its shape.
    pub topo_nodes: usize,
    pub topo_shortcuts: usize,
    pub topo_hops: usize,
    pub topo_payload: usize,
    pub topo_forget: f64,
    pub topo_expect: f64,
    pub topo_crowd: f64,
    pub topo_keep: usize,
    pub edge_nodes: usize,
    pub edge_shortcuts: usize,
    pub edge_hops: usize,
    pub edge_classes: usize,
    pub edge_dim: usize,
    pub edge_forget: f64,
    pub edge_hash_class: bool,
    pub edge_class_readout: bool,
    /// Ceiling on total class slices when capacity may be expanded at run
    /// time. 0 = fixed budget, the old behaviour.
    pub edge_expand: usize,
    /// Gain on the multi-timescale trace added to the payload. 0 = off.
    pub edge_ctx_gain: f64,
    /// Orthogonalise a new prototype against existing ones.
    pub edge_grow_orth: bool,
    /// Select each hop's edge from the previous hop's output.
    pub edge_route_compose: bool,
    /// Share one edge memory across all classes; readout stays per-class.
    pub edge_share: bool,
    /// Add a class-common readout block to the per-class one.
    pub edge_share_readout: bool,
    /// Route each write by which head scores the observed target best.
    pub edge_posterior: bool,
    /// Negatives sampled per write; 0 keeps the dense delta rule.
    pub edge_neg_samples: usize,
    /// Debounce length in units of a transition, not observations.
    pub edge_grow_hold: usize,
    /// Scales each write by addressing confidence. 0 disables.
    pub edge_gate: f64,
    /// Addressing-blind control: caps the per-write update norm. 0 disables.
    pub edge_clip: f64,
    /// Benna-Fusi rungs behind each class readout slice; 1 disables it.
    pub edge_rungs: usize,
    /// How many domain visits the first hidden rung should average over.
    ///
    /// Not a conductance. g1 is derived from this and the visit length in
    /// `run_arm_trial`, because every time this project has let a time
    /// constant be typed in as a literal it has ended up expressed in the
    /// wrong units (DESIGN-EDGE.md s12 lists six). A slice's clock is
    /// activity-gated, so it advances only while its own domain is being
    /// written -- one visit is span/3 activations, and an absence is zero.
    pub edge_ladder_visits: f64,
    /// Classes live at the start; the rest are grown on demand.
    pub edge_init_classes: usize,
    /// Novelty threshold for growing a class, in standard deviations below
    /// the running mean best-match. Self-scaling, so no similarity constant.
    pub edge_grow_k: f64,
    /// Retire the first half of the domains after this round, and start
    /// training the second half.
    ///
    /// Without it the four domains cycle for ever and nothing ever becomes
    /// permanently obsolete, so directed forgetting has no situation in which
    /// to pay: its premise is that capacity is genuinely exhausted and the
    /// knowledge being dropped is genuinely no longer wanted, and the endless
    /// cycle never presents that. 0 disables retirement.
    /// Rounds between successive domains entering the stream. 0 = all present
    /// from the start, the previous behaviour.
    pub arrival: usize,
    pub retire_after: usize,
    /// Rotation gain. Angles are gamma * (a_p . z_ctx) with unit-norm a_p, so
    /// this is roughly the maximum rotation in radians. 0 reduces the arm to
    /// a content-only projection, which is the internal control: it isolates
    /// what the rotation itself contributes from what merely dropping the
    /// context out of the additive projection does.
    pub rotate_gain: f64,
    pub out: PathBuf,
}

/// A product-of-experts bank: the input ladder's rungs are partitioned into
/// `n` contiguous groups, each group gets its own calibrated projection and
/// its own memory, and the per-expert logits are summed before one shared
/// softmax.
///
/// Summing logits is multiplying probabilities. Each expert contributes
/// `P(target | its own slice of the trace)` and the product keeps what they
/// agree on. What motivates it is the measured per-rung profile: the fast
/// rungs carry content (high cross-domain overlap, and correctly so in Mode
/// B, where the entity genuinely is the same entity in every domain), a mid
/// rung carries a near-clean domain code (overlap 0.105 at rung 5, against
/// 0.446 for the joint projection over the same stream), and concatenating
/// everything into one projection destroys that code by letting the
/// high-magnitude fast rungs dominate the top-k. The joint projection is
/// four times worse at separating domains than its own best component.
///
/// `n = 1` is exactly the previous architecture, so the expert count is an
/// ablation axis rather than a separate code path.
///
/// The split holds the budget fixed: `D_b = d_sdr / n` columns and
/// `k_b = k_active / n` active per expert, so every setting has the same
/// total column count, active count and parameter count as the n=1
/// baseline. That keeps the comparison fair (lesson 11) and stays well away
/// from the `k=1` quantisation that actually sank the band-split experiment
/// behind lesson 17 -- at d_sdr=512, k=16 even n=8 leaves k_b=2.
/// Outer-product (tensor) addressing: content and context are projected
/// separately and the memory cell is indexed by the *pair* rather than by a
/// single column.
///
/// The cell index is flattened to `i * d2 + j`, so the memory stays a
/// `V x (d1*d2)` matrix and every existing mechanism -- Benna-Fusi
/// diffusion, activity gating, the EWC anchor -- operates unchanged on the
/// flattened columns. The only thing that changes is how the active set and
/// its weights are computed, which is why this costs no new memory code.
///
/// This is the strictly more expressive sibling of the product-of-experts
/// bank. Summing logits constrains the joint to a rank-1 factorisation
/// `P(v|i)P(v|j)`; a cell per `(i,j)` holds an arbitrary joint, at the cost
/// of address space. Budget is held fixed the same way: `d1*d2 = d_sdr` and
/// `k1*k2 = k_active`, so the parameter and active-unit counts match the
/// flat baseline exactly.
/// Context-conditioned rotation of the content code.
///
/// Enlarging D was supposed to answer "is addressing the bottleneck" and did
/// not, because D moves two things at once. Measured at k=16, span 8000,
/// Mode B: going from D=512 to D=8192 halved cross-domain overlap
/// (0.463 -> 0.256, the separation we wanted) and simultaneously wrecked the
/// stability of a fact's own code (active-set retention at lag 1
/// 0.378 -> 0.221, at lag 100 0.120 -> 0.039). Retention came out flat
/// (38.1% -> 37.1%). Top-k's sensitivity to input perturbation is set by the
/// single scalar k/D, so that one knob separates what should be apart and
/// what should stay together by the same amount. They cannot be set
/// independently.
///
/// A rotation decouples them. `R` is a product of Givens rotations, hence
/// orthogonal: the content space keeps every dimension it had and k/D is
/// untouched, so within-fact stability is whatever it already was. The
/// angles come from the context, which is near-constant inside a domain
/// visit (measured active-set retention 0.92 on the context rungs) and
/// differs across domains -- so R is near-constant where codes should hold
/// together and differs where they should come apart. Separation is a
/// continuous function of context distance, with no threshold anywhere.
///
/// The address stays a function of (context, key) alone, which is what makes
/// it retrievable: surprise cannot enter here, because at probe time there is
/// no target to compute it from, and an address that cannot be recomputed at
/// read time cannot be read. The angles also depend only on the current
/// context and not on the path taken to reach it, so cycling the domains
/// returns a domain to its own addresses -- a path-dependent transport would
/// leave last lap's writes somewhere this lap does not look.
#[derive(Clone)]
struct ContextRotation {
    /// (d_content / 2) x d_context, unit-norm rows.
    a: Mat,
    gain: f64,
    n_pairs: usize,
    d_content: usize,
    theta: Vec<f64>,
    rotated: Vec<f64>,
}

impl ContextRotation {
    fn new(d_content: usize, d_context: usize, gain: f64, rng: &mut Rng) -> Self {
        let n_pairs = d_content / 2;
        let mut a = Mat::zeros(n_pairs.max(1), d_context);
        let slice = a.as_mut_slice();
        for r in 0..n_pairs.max(1) {
            rng.fill_unit_vector(&mut slice[r * d_context..(r + 1) * d_context]);
        }
        Self {
            a,
            gain,
            n_pairs,
            d_content,
            theta: vec![0.0; n_pairs.max(1)],
            rotated: vec![0.0; d_content],
        }
    }

    /// Writes `R(z_ctx) z_content` into the internal buffer and returns it.
    fn apply(&mut self, z_content: &[f64], z_context: &[f64]) -> &[f64] {
        self.a.mul_vec(z_context, &mut self.theta);
        self.rotated.copy_from_slice(z_content);
        for p in 0..self.n_pairs {
            let (s, c) = (self.gain * self.theta[p]).sin_cos();
            let (x, y) = (self.rotated[2 * p], self.rotated[2 * p + 1]);
            self.rotated[2 * p] = c * x - s * y;
            self.rotated[2 * p + 1] = s * x + c * y;
        }
        debug_assert_eq!(self.rotated.len(), self.d_content);
        &self.rotated
    }
}

#[derive(Clone)]
struct TensorAddr {
    proj_c: RandomProjection,
    proj_x: RandomProjection,
    d2: usize,
    split: usize,
    u_c: Vec<f64>,
    calib_c: Vec<f64>,
    pairs_c: Vec<(usize, f64)>,
    act_c: Vec<usize>,
    alp_c: Vec<f64>,
    u_x: Vec<f64>,
    calib_x: Vec<f64>,
    pairs_x: Vec<(usize, f64)>,
    act_x: Vec<usize>,
    alp_x: Vec<f64>,
}

/// State displaced by `ExpertBank::topo_borrow`, owed back to the bank.
///
/// Dropping it silently is exactly the bug in DESIGN-EDGE.md s29, so it is
/// `#[must_use]`: under `-D warnings` an unpaired borrow will not compile.
#[must_use = "the displaced state must go back via topo_return before training \
              resumes -- an unpaired restore lets the probe steer the run"]
pub struct Displaced(Option<Vec<f64>>);

#[derive(Clone)]
pub struct ExpertBank {
    /// Non-zero for the Linear arms: every input dimension is "active", the
    /// weights are the trace itself, and there is no projection at all.
    edge: Option<crate::edge::EdgeMemory>,
    topo: Option<crate::topo::TopoMemory>,
    linear_dim: usize,
    rotation: Option<ContextRotation>,
    rot_split: usize,
    tensor: Option<TensorAddr>,
    groups: Vec<(usize, usize)>,
    projs: Vec<RandomProjection>,
    mems: Vec<SdrMemory>,
    d_input: usize,
    u_buf: Vec<f64>,
    calib_buf: Vec<f64>,
    pairs_buf: Vec<(usize, f64)>,
    active: Vec<Vec<usize>>,
    alphas: Vec<Vec<f64>>,
    logits: Vec<f64>,
    partial: Vec<f64>,
    probs: Vec<f64>,
}

impl ExpertBank {
    pub fn new(cfg: &SdrConfig, arm: ArmKind, rng: &mut Rng) -> Self {
        let n = cfg.experts.clamp(1, cfg.m_in);
        let d_b = (cfg.d_sdr / n).max(1);
        let k_b = (cfg.k_active / n).max(1);
        let schedule = Schedule::Geometric {
            r: cfg.ladder_r,
            g1: cfg.ladder_g1,
        };

        if cfg.edge_nodes > 0 && !matches!(arm, ArmKind::Linear | ArmKind::LinearFast) {
            let em = crate::edge::EdgeMemory::new(
                cfg.edge_nodes,
                cfg.edge_shortcuts,
                cfg.edge_hops,
                cfg.edge_classes,
                cfg.edge_dim,
                cfg.vocab,
                // Slow enough to hold inside a visit, quick enough to follow
                // a switch: tau = span/5.
                5.0 / cfg.span_tokens.max(1) as f64,
                cfg.edge_forget,
                cfg.edge_hash_class,
                cfg.edge_class_readout,
                cfg.edge_init_classes,
                cfg.edge_grow_k,
                // One full rotation of the stream, in observations.
                (cfg.domains * (cfg.span_tokens / 3)) as f64,
                cfg.edge_rungs,
                cfg.ladder_r,
                {
                    // tau_1 = r^2 / g1 activations, and we want that to equal
                    // `edge_ladder_visits` visits' worth of activations. The
                    // slice advances once per observe, and a visit is span/3
                    // observes, all of them landing on the live domain's class.
                    let per_visit = (cfg.span_tokens / 3) as f64;
                    cfg.ladder_r.powi(2) / (cfg.edge_ladder_visits * per_visit)
                },
                cfg.edge_gate,
                cfg.edge_clip,
                cfg.edge_share,
                cfg.edge_share_readout,
                cfg.edge_posterior,
                cfg.edge_neg_samples,
                cfg.edge_expand,
                {
                    // Debounce length must exceed a transition, and the
                    // transition length is measured, not chosen: intrusion
                    // decays to zero by roughly a third of a visit. Typing a
                    // raw count here is the units mistake of s12, committed
                    // six times and once more in the commit that added this
                    // flag. `edge_grow_hold` is now a MULTIPLE of that.
                    let transition = (cfg.span_tokens / 3) as f64 * 0.25;
                    (cfg.edge_grow_hold as f64 * transition).round() as usize
                },
                cfg.edge_grow_orth,
                cfg.edge_route_compose,
                cfg.edge_ctx_gain,
                rng,
            );
            return Self {
                edge: Some(em),
                topo: None,
                linear_dim: 0,
                rotation: None,
                rot_split: 0,
                tensor: None,
                groups: Vec::new(),
                projs: Vec::new(),
                mems: Vec::new(),
                d_input: cfg.d_input,
                u_buf: Vec::new(),
                calib_buf: Vec::new(),
                pairs_buf: Vec::new(),
                active: Vec::new(),
                alphas: Vec::new(),
                logits: vec![0.0; cfg.vocab],
                partial: Vec::new(),
                probs: vec![0.0; cfg.vocab],
            };
        }

        if cfg.topo_nodes > 0 && !matches!(arm, ArmKind::Linear | ArmKind::LinearFast) {
            let topo = crate::topo::TopoMemory::new(
                cfg.topo_nodes,
                cfg.topo_shortcuts,
                cfg.topo_hops,
                cfg.topo_payload,
                cfg.vocab,
                cfg.topo_forget,
                cfg.topo_expect,
                cfg.topo_crowd,
                cfg.topo_keep,
                rng,
            );
            return Self {
                edge: None,
                topo: Some(topo),
                linear_dim: 0,
                rotation: None,
                rot_split: 0,
                tensor: None,
                groups: Vec::new(),
                projs: Vec::new(),
                mems: Vec::new(),
                d_input: cfg.d_input,
                u_buf: Vec::new(),
                calib_buf: Vec::new(),
                pairs_buf: Vec::new(),
                active: Vec::new(),
                alphas: Vec::new(),
                logits: vec![0.0; cfg.vocab],
                partial: Vec::new(),
                probs: vec![0.0; cfg.vocab],
            };
        }

        if matches!(arm, ArmKind::Linear | ArmKind::LinearFast) {
            let dim = if matches!(arm, ArmKind::LinearFast) {
                cfg.d_input
            } else {
                cfg.m_in * cfg.d_input
            };
            return Self {
                edge: None,
                topo: None,
                rotation: None,
                rot_split: 0,
                tensor: None,
                groups: vec![(0, if matches!(arm, ArmKind::LinearFast) { 1 } else { cfg.m_in })],
                projs: Vec::new(),
                mems: vec![SdrMemory::new_plain(cfg.vocab, dim)],
                d_input: cfg.d_input,
                u_buf: Vec::new(),
                calib_buf: Vec::new(),
                pairs_buf: Vec::new(),
                active: vec![(0..dim).collect()],
                alphas: vec![vec![0.0; dim]],
                logits: vec![0.0; cfg.vocab],
                partial: vec![0.0; cfg.vocab],
                probs: vec![0.0; cfg.vocab],
                linear_dim: dim,
            };
        }

        if cfg.rotate {
            let split = if cfg.tensor_split == 0 {
                cfg.m_in / 2
            } else {
                cfg.tensor_split
            }
            .clamp(1, cfg.m_in - 1);
            let d_content = split * cfg.d_input;
            let d_context = (cfg.m_in - split) * cfg.d_input;
            let mem = match arm {
                ArmKind::Ladder8 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 8, schedule),
                ArmKind::Ladder4 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 4, schedule),
                ArmKind::Ladder2 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 2, schedule),
                ArmKind::Plain => SdrMemory::new_plain(cfg.vocab, cfg.d_sdr),
                ArmKind::ProximalEwc { lambda } => {
                    let cycle = (cfg.domains * (cfg.span_tokens / 3)).max(1) as f64;
                    SdrMemory::new_ewc(cfg.vocab, cfg.d_sdr, lambda, 1.0 / cycle)
                }
                ArmKind::Linear | ArmKind::LinearFast => unreachable!("handled above"),
            };
            return Self {
                edge: None,
                topo: None,
                linear_dim: 0,
                rotation: Some(ContextRotation::new(
                    d_content,
                    d_context,
                    cfg.rotate_gain,
                    rng,
                )),
                rot_split: split,
                tensor: None,
                groups: vec![(0, split)],
                // Phi now sees only the content rungs; the context enters
                // through the rotation, not as extra additive dimensions.
                projs: vec![RandomProjection::new(
                    d_content,
                    cfg.d_sdr,
                    cfg.k_active,
                    calib_rate(cfg),
                    rng,
                )],
                mems: vec![mem],
                d_input: cfg.d_input,
                u_buf: vec![0.0; cfg.d_sdr],
                calib_buf: vec![0.0; cfg.d_sdr],
                pairs_buf: Vec::with_capacity(cfg.d_sdr),
                active: vec![Vec::with_capacity(cfg.k_active)],
                alphas: vec![Vec::with_capacity(cfg.k_active)],
                logits: vec![0.0; cfg.vocab],
                partial: vec![0.0; cfg.vocab],
                probs: vec![0.0; cfg.vocab],
            };
        }

        if cfg.tensor_d2 > 0 {
            let d2 = cfg.tensor_d2;
            let k2 = cfg.tensor_k2.clamp(1, d2);
            let d1 = (cfg.d_sdr / d2).max(1);
            let k1 = (cfg.k_active / k2).max(1);
            let split = if cfg.tensor_split == 0 {
                cfg.m_in / 2
            } else {
                cfg.tensor_split
            }
            .clamp(1, cfg.m_in - 1);

            let tensor = TensorAddr {
                proj_c: RandomProjection::new(
                    split * cfg.d_input,
                    d1,
                    k1.min(d1),
                    calib_rate(cfg),
                    rng,
                ),
                proj_x: RandomProjection::new(
                    (cfg.m_in - split) * cfg.d_input,
                    d2,
                    k2,
                    calib_rate(cfg),
                    rng,
                ),
                d2,
                split,
                u_c: vec![0.0; d1],
                calib_c: vec![0.0; d1],
                pairs_c: Vec::with_capacity(d1),
                act_c: Vec::with_capacity(k1),
                alp_c: Vec::with_capacity(k1),
                u_x: vec![0.0; d2],
                calib_x: vec![0.0; d2],
                pairs_x: Vec::with_capacity(d2),
                act_x: Vec::with_capacity(k2),
                alp_x: Vec::with_capacity(k2),
            };
            let flat = d1 * d2;
            let mem = match arm {
                ArmKind::Ladder8 => SdrMemory::new_ladder(cfg.vocab, flat, 8, schedule),
                ArmKind::Ladder4 => SdrMemory::new_ladder(cfg.vocab, flat, 4, schedule),
                ArmKind::Ladder2 => SdrMemory::new_ladder(cfg.vocab, flat, 2, schedule),
                ArmKind::Plain => SdrMemory::new_plain(cfg.vocab, flat),
                ArmKind::ProximalEwc { lambda } => {
                    let cycle = (cfg.domains * (cfg.span_tokens / 3)).max(1) as f64;
                    SdrMemory::new_ewc(cfg.vocab, flat, lambda, 1.0 / cycle)
                }
                ArmKind::Linear | ArmKind::LinearFast => unreachable!("handled above"),
            };
            return Self {
                edge: None,
                topo: None,
                linear_dim: 0,
                rotation: None,
                rot_split: 0,
                tensor: Some(tensor),
                groups: Vec::new(),
                projs: Vec::new(),
                mems: vec![mem],
                d_input: cfg.d_input,
                u_buf: Vec::new(),
                calib_buf: Vec::new(),
                pairs_buf: Vec::new(),
                active: vec![Vec::with_capacity(k1 * k2)],
                alphas: vec![Vec::with_capacity(k1 * k2)],
                logits: vec![0.0; cfg.vocab],
                partial: vec![0.0; cfg.vocab],
                probs: vec![0.0; cfg.vocab],
            };
        }

        let mut groups = Vec::with_capacity(n);
        let mut projs = Vec::with_capacity(n);
        let mut mems = Vec::with_capacity(n);
        for b in 0..n {
            let lo = b * cfg.m_in / n;
            let hi = (b + 1) * cfg.m_in / n;
            groups.push((lo, hi));
            projs.push(RandomProjection::new(
                (hi - lo) * cfg.d_input,
                d_b,
                k_b,
                calib_rate(cfg),
                rng,
            ));
            mems.push(match arm {
                ArmKind::Ladder8 => SdrMemory::new_ladder(cfg.vocab, d_b, 8, schedule),
                ArmKind::Ladder4 => SdrMemory::new_ladder(cfg.vocab, d_b, 4, schedule),
                ArmKind::Ladder2 => SdrMemory::new_ladder(cfg.vocab, d_b, 2, schedule),
                ArmKind::Plain => SdrMemory::new_plain(cfg.vocab, d_b),
                ArmKind::ProximalEwc { lambda } => {
                    let cycle = (cfg.domains * (cfg.span_tokens / 3)).max(1) as f64;
                    SdrMemory::new_ewc(cfg.vocab, d_b, lambda, 1.0 / cycle)
                }
                ArmKind::Linear | ArmKind::LinearFast => unreachable!("handled above"),
            });
        }

        Self {
            edge: None,
            topo: None,
            linear_dim: 0,
            rotation: None,
            rot_split: 0,
            tensor: None,
            groups,
            projs,
            mems,
            d_input: cfg.d_input,
            u_buf: vec![0.0; d_b],
            calib_buf: vec![0.0; d_b],
            pairs_buf: Vec::with_capacity(d_b),
            active: vec![Vec::with_capacity(k_b); n],
            alphas: vec![Vec::with_capacity(k_b); n],
            logits: vec![0.0; cfg.vocab],
            partial: vec![0.0; cfg.vocab],
            probs: vec![0.0; cfg.vocab],
        }
    }

    /// Selects each expert's active set from its own slice of the trace and
    /// accumulates the summed logits. Calibration updates here as everywhere
    /// else, including on probe paths.
    fn select_and_forward(&mut self, trace: &[f64]) {
        if self.linear_dim > 0 {
            let n = self.linear_dim;
            // alphas are the trace itself, L1-normalised. Signs are kept --
            // a linear model needs them, and this is the one place the
            // architecture's non-negative cosine weighting is deliberately
            // not imposed. The normalisation only matches the scale
            // convention the sparse arms already satisfy (their alphas sum
            // to 1), so one eta grid covers both; it is a fixed gain on the
            // logits, not a change of model class.
            let l1: f64 = trace[..n].iter().map(|z| z.abs()).sum::<f64>().max(1e-12);
            for (a, z) in self.alphas[0].iter_mut().zip(&trace[..n]) {
                *a = z / l1;
            }
            self.mems[0].forward(&self.active[0], &self.alphas[0], &mut self.logits);
            return;
        }

        if self.rotation.is_some() {
            let Self {
                rotation,
                rot_split,
                projs,
                mems,
                d_input,
                u_buf,
                calib_buf,
                pairs_buf,
                active,
                alphas,
                logits,
                ..
            } = self;
            let cut = *rot_split * *d_input;
            let rot = rotation.as_mut().expect("rotation branch");
            let z = rot.apply(&trace[..cut], &trace[cut..]);
            projs[0].project_and_select(
                z,
                u_buf,
                calib_buf,
                pairs_buf,
                &mut active[0],
                &mut alphas[0],
            );
            mems[0].forward(&active[0], &alphas[0], logits);
            return;
        }

        if self.tensor.is_some() {
            let Self {
                tensor,
                mems,
                d_input,
                active,
                alphas,
                logits,
                ..
            } = self;
            let t = tensor.as_mut().expect("tensor branch");
            let cut = t.split * *d_input;
            t.proj_c.project_and_select(
                &trace[..cut],
                &mut t.u_c,
                &mut t.calib_c,
                &mut t.pairs_c,
                &mut t.act_c,
                &mut t.alp_c,
            );
            t.proj_x.project_and_select(
                &trace[cut..],
                &mut t.u_x,
                &mut t.calib_x,
                &mut t.pairs_x,
                &mut t.act_x,
                &mut t.alp_x,
            );
            // Every (content, context) pair is one cell. Both alpha vectors
            // already sum to 1, so their products do too -- no renormalising.
            let a = &mut active[0];
            let al = &mut alphas[0];
            a.clear();
            al.clear();
            for (ii, &i) in t.act_c.iter().enumerate() {
                for (jj, &j) in t.act_x.iter().enumerate() {
                    a.push(i * t.d2 + j);
                    al.push(t.alp_c[ii] * t.alp_x[jj]);
                }
            }
            mems[0].forward(a, al, logits);
            return;
        }

        let Self {
            groups,
            projs,
            mems,
            d_input,
            u_buf,
            calib_buf,
            pairs_buf,
            active,
            alphas,
            logits,
            partial,
            ..
        } = self;

        logits.fill(0.0);
        for b in 0..projs.len() {
            let (lo, hi) = groups[b];
            let slice = &trace[lo * *d_input..hi * *d_input];
            projs[b].project_and_select(
                slice,
                u_buf,
                calib_buf,
                pairs_buf,
                &mut active[b],
                &mut alphas[b],
            );
            mems[b].forward(&active[b], &alphas[b], partial);
            for (l, p) in logits.iter_mut().zip(partial.iter()) {
                *l += *p;
            }
        }
    }

    /// Reads without writing any weight, matching `SdrMemory::predict`'s
    /// contract -- the 0-shot protocol needs the probed fact not to teach
    /// itself before it is scored.
    /// Index of the highest logit from the last `predict`. Used by the
    /// prediction-collapse check, not by the model itself.
    pub fn path_entropy(&self) -> Option<(f64, f64)> {
        self.edge
            .as_ref()
            .map(|e| e.edge_entropy())
            .or_else(|| self.topo.as_ref().map(|t| t.path_entropy()))
    }

    pub fn last_argmax(&self) -> usize {
        if let Some(e) = self.edge.as_ref() {
            return e.last_argmax();
        }
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &l) in self.logits.iter().enumerate() {
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        best
    }

    /// Reads a fact. The topological arm needs the tokens themselves, since
    /// it builds its own payload and keeps its context in the nodes rather
    /// than reading a globally computed trace.
    pub fn predict_fact(
        &mut self,
        entity: usize,
        relation: usize,
        trace: &[f64],
        target: usize,
    ) -> (f64, bool) {
        if let Some(e) = self.edge.as_mut() {
            e.set_pending_context(trace);
            return e.predict_fact(entity, relation, target);
        }
        if let Some(t) = self.topo.as_mut() {
            return t.predict_fact(entity, relation, target);
        }
        self.predict(trace, target)
    }

    pub fn predict(&mut self, trace: &[f64], target: usize) -> (f64, bool) {
        assert!(
            self.topo.is_none() && self.edge.is_none(),
            "the topological and edge arms must be read through predict_fact"
        );
        self.select_and_forward(trace);
        let mut probs = std::mem::take(&mut self.probs);
        let out = SdrMemory::score_logits(&self.logits, target, &mut probs);
        self.probs = probs;
        out
    }

    /// Every expert's surface weights concatenated, for tests that assert a
    /// read path left the memory untouched.
    #[cfg(test)]
    pub fn fast_weights_snapshot(&self) -> Vec<f64> {
        self.mems
            .iter()
            .flat_map(|m| m.read_fast_weights().as_slice().to_vec())
            .collect()
    }

    /// Records where this fact's mass landed, so the probe can be asked
    /// whether it returns there.
    pub fn note_consistency(&mut self, entity: usize, relation: usize) {
        if let Some(e) = self.edge.as_mut() {
            e.note_consistency(entity, relation);
        }
        if let Some(t) = self.topo.as_mut() {
            t.note_consistency(entity, relation);
        }
    }

    /// The node states, so the probe can put the network back where the
    /// domain left it -- the topological counterpart of restoring the input
    /// ladder, and what makes a written fact findable again.
    pub fn topo_snapshot(&self) -> Option<Vec<f64>> {
        self.edge
            .as_ref()
            .map(|e| e.snapshot())
            .or_else(|| self.topo.as_ref().map(|t| t.snapshot()))
    }

    /// Installs `snap` and hands back the state it displaced.
    ///
    /// The probe shares the bank with training, so a restore that is never
    /// undone does not measure the run, it steers it. That went wrong twice
    /// and in opposite directions (DESIGN-EDGE.md s29): one path handed each
    /// domain its class back from a slot keyed by the true domain index and
    /// inflated the headline from 17.6% to 96%; the other dragged training to
    /// domain 0 four times a visit and left the class mechanism measuring as
    /// completely inert.
    ///
    /// `Displaced` is `#[must_use]`, and this crate is gated on
    /// `clippy -D warnings`, so forgetting to pair the restore is a build
    /// failure rather than something to remember.
    pub fn topo_borrow(&mut self, snap: &[f64]) -> Displaced {
        let natural = self.topo_snapshot();
        self.topo_restore(snap);
        Displaced(natural)
    }

    /// Puts back what `topo_borrow` displaced. Call before training resumes.
    pub fn topo_return(&mut self, d: Displaced) {
        if let Some(nat) = d.0 {
            self.topo_restore(&nat);
        }
    }

    fn topo_restore(&mut self, snap: &[f64]) {
        if let Some(e) = self.edge.as_mut() {
            e.restore(snap);
            return;
        }
        if let Some(t) = self.topo.as_mut() {
            t.restore(snap);
        }
    }

    pub fn routing_consistency(&self) -> Option<f64> {
        self.edge
            .as_ref()
            .map(|e| e.routing_consistency())
            .or_else(|| self.topo.as_ref().map(|t| t.routing_consistency()))
    }

    /// Class switches per token. Near zero inside a visit and non-zero across
    /// them is what a working class axis looks like; never switching, or
    /// switching constantly, means it is decoration.
    pub fn reset_stream_ctx(&mut self) {
        if let Some(e) = self.edge.as_mut() {
            e.reset_ctx();
        }
    }

    /// Feeds a raw stream token to whichever memory is active.
    pub fn absorb_stream_token(&mut self, token: usize) {
        if let Some(e) = self.edge.as_mut() {
            e.absorb_token(token);
        }
    }

    pub fn set_domain(&mut self, d: usize) {
        if let Some(e) = self.edge.as_mut() {
            e.set_domain(d);
        }
    }

    pub fn growth_stats(&self) -> Option<(f64, f64)> {
        self.edge.as_ref().map(|e| e.growth_stats())
    }

    pub fn class_collision(&self) -> Option<f64> {
        self.edge.as_ref().map(|e| e.class_collision())
    }

    pub fn posterior_move_rate(&self) -> Option<f64> {
        self.edge.as_ref().map(|e| e.posterior_move_rate())
    }

    pub fn readout_occupancy(&self) -> Option<(usize, usize, usize)> {
        self.edge.as_ref().map(|e| e.readout_occupancy())
    }

    pub fn context_ratio(&self) -> Option<f64> {
        self.edge.as_ref().map(|e| e.context_ratio())
    }

    pub fn activation(&self) -> Option<(usize, usize)> {
        self.edge.as_ref().map(|e| e.activation())
    }

    pub fn edge_write_spread(&self) -> Option<(f64, f64)> {
        self.edge.as_ref().map(|e| e.edge_write_spread())
    }

    pub fn growth_timing(&self) -> Option<[usize; 4]> {
        self.edge.as_ref().map(|e| e.growth_timing())
    }

    pub fn write_for_domain(&self, d: usize) -> f64 {
        self.edge.as_ref().map(|e| e.write_for_domain(d)).unwrap_or(0.0)
    }

    pub fn intrusion_stats(&self) -> Option<(f64, f64, [f64; 4])> {
        self.edge.as_ref().map(|e| e.intrusion_stats())
    }

    pub fn collision_stats(&self) -> Option<(usize, usize, f64)> {
        self.edge.as_ref().map(|e| e.collision_stats())
    }

    pub fn domain_class_matrix(&self, domains: usize) -> Option<Vec<Vec<f64>>> {
        self.edge.as_ref().map(|e| e.domain_class_matrix(domains))
    }

    pub fn take_write_norms(&mut self) -> Option<(f64, f64)> {
        self.edge.as_mut().map(|e| e.take_write_norms())
    }

    pub fn class_switch_rate(&self) -> Option<f64> {
        self.edge.as_ref().map(|e| e.class_switch_rate())
    }

    pub fn observe_fact(
        &mut self,
        entity: usize,
        relation: usize,
        trace: &[f64],
        target: usize,
        eta: f64,
    ) {
        if let Some(t) = self.topo.as_mut() {
            t.observe_fact(entity, relation, target, eta);
            return;
        }
        if let Some(e) = self.edge.as_mut() {
            e.set_pending_context(trace);
            e.observe_fact(entity, relation, target, eta);
            // The target is a stream token too, and in Mode B it is the only
            // one that differs by domain -- absorbing it is what lets the
            // class tell the domains apart.
            e.absorb_token(target);
            return;
        }
        self.observe(trace, target, eta);
    }

    pub fn observe(&mut self, trace: &[f64], target: usize, eta: f64) {
        assert!(
            self.topo.is_none() && self.edge.is_none(),
            "the topological and edge arms must be written through observe_fact: \
             routing them here silently trains nothing, because their projection \
             list is empty and the loop below runs zero times"
        );
        self.select_and_forward(trace);
        let mut probs = std::mem::take(&mut self.probs);
        let _ = SdrMemory::score_logits(&self.logits, target, &mut probs);
        // One shared residual, each expert charged on its own active set.
        // That shared residual is what makes this a product of experts and
        // not an ensemble of separately-trained models: each expert is
        // trained on what the *combined* prediction still gets wrong, so
        // they specialise against each other instead of all learning the
        // same marginal.
        for b in 0..self.mems.len() {
            self.mems[b].update_sparse(&self.active[b], &self.alphas[b], target, eta, &probs);
        }
        self.probs = probs;
    }
}

/// Runs a single trial of an arm under a specific learning rate eta.
/// Manipulation checks for the source and the projection, computed once.
///
/// The active set depends only on the input ladder and the fixed projection, so
/// it is identical across arms and is a property of the stream rather than of
/// any memory. Two things are checked here, and neither was measured before.
///
/// **Domain activation overlap.** DESIGN-SDR's lesson 1 stakes the whole design
/// on sparse spatial isolation: consolidation is claimed to work only when
/// different domains land on near-disjoint neurons. That premise had never been
/// tested. Reported as shared activation mass, 1.0 meaning the domains use the
/// same neurons and 0.0 meaning they are disjoint. A null result cannot be read
/// without it -- an ineffective mechanism and an isolation that never formed
/// look identical from the outside.
///
/// **Fact exposure.** Under a Zipf walk the tail facts may appear only a
/// handful of times per visit, in which case nothing has been learned to
/// forget, the accuracy curve sits flat, and the learning-rate search is driven
/// to its upper bound trying to extract signal from too few exposures.
pub struct SourceChecks {
    pub overlap: Vec<Vec<f64>>,
    pub mean_overlap: f64,
    /// Same shared-activation-mass overlap, but the activity histogram only
    /// counts steps on domain-specific (mid+tail, rank_tier != 0) facts --
    /// hub-tier (rank_tier == 0) steps are excluded from the histogram
    /// entirely, not just from the probe loss. Separates "domains overlap
    /// because they share a hub layer by design" from "domains overlap
    /// because distinct leaf entities collide in D_sdr space regardless".
    pub mean_overlap_leaf_only: f64,
    pub exposure_by_tier: [(f64, f64, f64); 3],
    pub active_fraction: f64,
    /// Fraction of the active set retained from one step to the next, and from
    /// a hundred steps earlier. A code that does not turn over is not encoding
    /// the input, and then a high domain overlap and a small set of neurons
    /// ever used are both consequences of that rather than separate problems.
    pub retention_1: f64,
    pub retention_100: f64,
    /// Share of total win-mass (activation count, pooled across all domains)
    /// held by the single most-selected column, and by the top 10. If the
    /// fixed random projection has hubs -- some columns winning top-k for a
    /// disproportionate share of contexts purely from concentration of
    /// measure in high dimensions, independent of content -- this is where
    /// it would show up: skew far beyond what 44.6%-of-D-ever-used alone
    /// tells you, since that stat is silent on how uneven the used half is.
    pub top1_win_share: f64,
    pub top10_win_share: f64,
    /// How many columns (out of D_sdr, sorted by win count descending)
    /// account for half of all activation mass. A uniform code would need
    /// close to D_sdr/2; a hub-dominated one needs far fewer.
    pub cols_for_half_mass: usize,
    /// Per-input-rung cross-domain overlap: each rung projected on its own
    /// through an independent calibrated projection, at the same D and k as
    /// the joint one so the numbers are directly comparable.
    ///
    /// Read together with `rung_retention_100`, never alone. A context code
    /// worth building an expert on needs BOTH a low cross-domain overlap (it
    /// separates domains) AND a high within-domain retention (it is a stable
    /// code rather than noise). A rung carrying nothing but noise scores a
    /// low overlap for entirely the wrong reason, and the pair is what tells
    /// the two apart.
    pub rung_overlap: Vec<f64>,
    /// Per-rung fraction of the active set still active 100 steps later,
    /// measured only within a domain visit -- comparing across a domain
    /// boundary would score the domain switch itself as instability.
    pub rung_retention_100: Vec<f64>,
    /// Relaxation time of each rung, in events, for labelling the profile.
    pub rung_tau: Vec<f64>,
    /// Overlap between a domain's activation histogram in consecutive rounds,
    /// averaged over domains and round pairs.
    ///
    /// Domain cycling is a closed loop in context space. If the address map
    /// does not return a domain to its own addresses after one full lap, then
    /// what was written on the previous lap is not where the next lap looks
    /// for it, and retrieval fails structurally -- no memory mechanism can
    /// repair that, because nothing is wrong with the memory. 1.0 means the
    /// addresses come back exactly; low values mean the code drifts lap over
    /// lap. Worth measuring on our own account: the online z-score
    /// calibration is itself a slowly-moving state, so it can introduce
    /// exactly this drift.
    pub cycle_return: f64,
}

/// EMA follow rate for `RandomProjection`'s per-column hub-correction
/// statistics. Unlike the EWC anchor/Fisher trail, this stat chases a
/// roughly stationary target (Phi is fixed and the trace's marginal
/// distribution doesn't shift at domain boundaries the way a per-domain
/// anchor would need to track), so there's no moving-target risk in
/// picking a window shorter than a full cycle -- one domain visit is enough
/// for it to lock in and then stay a smooth running statistic for the rest
/// of the run.
fn calib_rate(cfg: &SdrConfig) -> f64 {
    1.0 / cfg.span_tokens.max(1) as f64
}

pub fn measure_source(cfg: &SdrConfig, stream: &RelationalFactStream) -> SourceChecks {
    let mut rng = Rng::new(cfg.seed);
    let ladder = InputContextLadder::new(cfg.d_input, cfg.vocab, cfg.m_in, cfg.ladder_r, &mut rng);
    let mut proj = RandomProjection::new(
        ladder.total_dim(),
        cfg.d_sdr,
        cfg.k_active,
        calib_rate(cfg),
        &mut rng,
    );

    let mut activity = vec![vec![0.0f64; cfg.d_sdr]; cfg.domains];
    let mut activity_round = vec![vec![vec![0.0f64; cfg.d_sdr]; cfg.domains]; cfg.rounds];
    let mut activity_leaf = vec![vec![0.0f64; cfg.d_sdr]; cfg.domains];
    let mut u_buf = vec![0.0; cfg.d_sdr];
    let mut calib_buf = vec![0.0; cfg.d_sdr];
    let mut pairs = Vec::with_capacity(cfg.d_sdr);
    let mut active = Vec::with_capacity(cfg.k_active);
    let mut alphas = Vec::with_capacity(cfg.k_active);
    let mut walk_ladder = ladder.clone();
    let mut history: std::collections::VecDeque<Vec<usize>> = std::collections::VecDeque::new();
    let (mut keep1, mut keep100, mut n1, mut n100) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    // One independent projection per input rung, each seeing only that rung's
    // slice of the trace, at the same D and k as the joint projection. This
    // measures how much domain information each timescale carries on its own,
    // which is the question a per-timescale expert decomposition turns on --
    // and which the band-split experiment behind lesson 17 could not answer,
    // since it never separated domains and ran k=1 per band.
    let mut rung_projs: Vec<RandomProjection> = (0..cfg.m_in)
        .map(|_| {
            RandomProjection::new(
                cfg.d_input,
                cfg.d_sdr,
                cfg.k_active,
                calib_rate(cfg),
                &mut rng,
            )
        })
        .collect();
    let mut rung_activity = vec![vec![vec![0.0f64; cfg.d_sdr]; cfg.domains]; cfg.m_in];
    let mut rung_history: Vec<std::collections::VecDeque<Vec<usize>>> =
        vec![std::collections::VecDeque::new(); cfg.m_in];
    let mut rung_keep100 = vec![0.0f64; cfg.m_in];
    let mut rung_n100 = vec![0.0f64; cfg.m_in];
    let mut r_active = Vec::with_capacity(cfg.k_active);
    let mut r_alphas = Vec::with_capacity(cfg.k_active);

    // One full pass of the cycling stream, exactly as training sees it.
    // round_idx and d index several parallel collections; the index reads
    // better here than zipping four of them.
    #[allow(clippy::needless_range_loop)]
    for round_idx in 0..cfg.rounds {
        for d in 0..cfg.domains {
            // Per-rung stability is measured within a domain visit only; a
            // pair straddling a boundary would count the domain switch itself
            // as the code failing to hold, which is the opposite of what this
            // number is for.
            for h in rung_history.iter_mut() {
                h.clear();
            }
            for fact in &stream.walks[d] {
                walk_ladder.step(fact.entity);
                walk_ladder.step(fact.relation);
                let trace = walk_ladder.normalized_trace();
                proj.project_and_select(
                    trace,
                    &mut u_buf,
                    &mut calib_buf,
                    &mut pairs,
                    &mut active,
                    &mut alphas,
                );
                for &i in &active {
                    activity[d][i] += 1.0;
                    activity_round[round_idx][d][i] += 1.0;
                }
                if fact.rank_tier != 0 {
                    for &i in &active {
                        activity_leaf[d][i] += 1.0;
                    }
                }
                let now: Vec<usize> = {
                    let mut v = active.clone();
                    v.sort_unstable();
                    v
                };
                if let Some(prev) = history.back() {
                    let shared = now.iter().filter(|i| prev.binary_search(i).is_ok()).count();
                    keep1 += shared as f64 / now.len() as f64;
                    n1 += 1.0;
                }
                if history.len() >= 100 {
                    let old = &history[history.len() - 100];
                    let shared = now.iter().filter(|i| old.binary_search(i).is_ok()).count();
                    keep100 += shared as f64 / now.len() as f64;
                    n100 += 1.0;
                }
                history.push_back(now);
                if history.len() > 101 {
                    history.pop_front();
                }

                for k in 0..cfg.m_in {
                    let slice = &trace[k * cfg.d_input..(k + 1) * cfg.d_input];
                    rung_projs[k].project_and_select(
                        slice,
                        &mut u_buf,
                        &mut calib_buf,
                        &mut pairs,
                        &mut r_active,
                        &mut r_alphas,
                    );
                    for &i in &r_active {
                        rung_activity[k][d][i] += 1.0;
                    }
                    let r_now: Vec<usize> = {
                        let mut v = r_active.clone();
                        v.sort_unstable();
                        v
                    };
                    if rung_history[k].len() >= 100 {
                        let old = &rung_history[k][rung_history[k].len() - 100];
                        let shared = r_now.iter().filter(|i| old.binary_search(i).is_ok()).count();
                        rung_keep100[k] += shared as f64 / r_now.len() as f64;
                        rung_n100[k] += 1.0;
                    }
                    rung_history[k].push_back(r_now);
                    if rung_history[k].len() > 101 {
                        rung_history[k].pop_front();
                    }
                }
                walk_ladder.step(fact.target);
            }
        }
    }

    // Shared activation mass between each pair of domains.
    let pairwise_overlap = |raw: &[Vec<f64>]| -> (Vec<Vec<f64>>, f64) {
        let norm: Vec<Vec<f64>> = raw
            .iter()
            .map(|a| {
                let s: f64 = a.iter().sum::<f64>().max(f64::MIN_POSITIVE);
                a.iter().map(|v| v / s).collect()
            })
            .collect();
        let mut overlap = vec![vec![0.0; raw.len()]; raw.len()];
        let (mut off_sum, mut off_n) = (0.0, 0.0);
        for a in 0..raw.len() {
            for b in 0..raw.len() {
                let o: f64 = norm[a].iter().zip(&norm[b]).map(|(x, y)| x.min(*y)).sum();
                overlap[a][b] = o;
                if a != b {
                    off_sum += o;
                    off_n += 1.0;
                }
            }
        }
        (overlap, if off_n > 0.0 { off_sum / off_n } else { 0.0 })
    };
    let (overlap, mean_overlap) = pairwise_overlap(&activity);
    let (_, mean_overlap_leaf_only) = pairwise_overlap(&activity_leaf);
    let used: f64 = activity
        .iter()
        .map(|a| a.iter().filter(|v| **v > 0.0).count() as f64)
        .sum::<f64>()
        / (cfg.domains * cfg.d_sdr) as f64;

    // Win-count concentration, pooled across domains -- does a fixed random
    // column win top-k for a disproportionate share of contexts regardless
    // of which domain/content is driving it (hubness), or is the used half
    // of D roughly evenly shared?
    let mut pooled_wins = vec![0.0f64; cfg.d_sdr];
    for a in &activity {
        for (p, &v) in pooled_wins.iter_mut().zip(a) {
            *p += v;
        }
    }
    let total_wins: f64 = pooled_wins.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let mut sorted_wins = pooled_wins.clone();
    sorted_wins.sort_by(|a, b| b.total_cmp(a));
    let top1_win_share = sorted_wins[0] / total_wins;
    let top10_win_share = sorted_wins.iter().take(10).sum::<f64>() / total_wins;
    let mut cols_for_half_mass = 0usize;
    let mut running = 0.0;
    for &w in &sorted_wins {
        running += w;
        cols_for_half_mass += 1;
        if running >= 0.5 * total_wins {
            break;
        }
    }

    // Exposure counts per probe fact, split by Zipf tier.
    let mut tier: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for d in 0..cfg.domains {
        for f in &stream.facts[d] {
            let n = stream.walks[d]
                .iter()
                .filter(|w| w.entity == f.entity && w.relation == f.relation)
                .count() as f64;
            tier[f.rank_tier.min(2)].push(n * cfg.rounds as f64);
        }
    }
    let stat = |v: &mut Vec<f64>| -> (f64, f64, f64) {
        if v.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let exposure_by_tier = [stat(&mut tier[0]), stat(&mut tier[1]), stat(&mut tier[2])];

    let rung_overlap: Vec<f64> = rung_activity
        .iter()
        .map(|per_domain| pairwise_overlap(per_domain).1)
        .collect();
    let rung_retention_100: Vec<f64> = rung_keep100
        .iter()
        .zip(&rung_n100)
        .map(|(s, n)| if *n > 0.0 { s / n } else { f64::NAN })
        .collect();
    let rung_tau: Vec<f64> = (0..cfg.m_in).map(|k| ladder.relaxation_time(k)).collect();

    let norm1 = |a: &[f64]| -> Vec<f64> {
        let s: f64 = a.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        a.iter().map(|v| v / s).collect()
    };
    let (mut cyc_sum, mut cyc_n) = (0.0f64, 0.0f64);
    for pair in activity_round.windows(2) {
        for (prev, next) in pair[0].iter().zip(&pair[1]) {
            let a = norm1(prev);
            let b = norm1(next);
            cyc_sum += a.iter().zip(&b).map(|(x, y)| x.min(*y)).sum::<f64>();
            cyc_n += 1.0;
        }
    }
    let cycle_return = if cyc_n > 0.0 { cyc_sum / cyc_n } else { f64::NAN };

    SourceChecks {
        overlap,
        mean_overlap,
        mean_overlap_leaf_only,
        exposure_by_tier,
        active_fraction: used,
        retention_1: if n1 > 0.0 { keep1 / n1 } else { f64::NAN },
        retention_100: if n100 > 0.0 { keep100 / n100 } else { f64::NAN },
        top1_win_share,
        top10_win_share,
        cols_for_half_mass,
        rung_overlap,
        rung_retention_100,
        rung_tau,
        cycle_return,
    }
}

/// Whether domain `d` is still being trained in round `r`.
fn trains(cfg: &SdrConfig, r: usize, d: usize) -> bool {
    // Sequential arrival. With every domain present from round 1 there is no
    // k-th regime to price, so the marginal-cost curve cannot be measured at
    // all; domain d simply waits until its round.
    if cfg.arrival > 0 && r < 1 + d * cfg.arrival {
        return false;
    }
    if cfg.retire_after == 0 {
        return true;
    }
    let half = cfg.domains / 2;
    if r <= cfg.retire_after {
        d < half
    } else {
        d >= half
    }
}

pub fn run_arm_trial(
    arm: ArmKind,
    eta: f64,
    cfg: &SdrConfig,
    stream: &RelationalFactStream,
) -> (Vec<TrajectoryRecord>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(cfg.seed);
    let ladder_template =
        InputContextLadder::new(cfg.d_input, cfg.vocab, cfg.m_in, cfg.ladder_r, &mut rng);
    // The EWC anchor/Fisher follow rate lives in ExpertBank::new now. Its
    // inverse is EWC's memory timescale and it has to span what the stream
    // asks it to hold: the old value 0.05 gives tau=20 steps against a domain
    // visit of span_tokens/3 and a full cycle of domains*span_tokens/3, so by
    // five time constants into a new domain the anchor had already converged
    // onto it and forgotten the one before -- a penalty pulling toward a
    // moving target rather than toward old-domain knowledge.
    let mut bank = ExpertBank::new(cfg, arm, &mut rng);

    let mut records = Vec::new();
    let mut round_accs = Vec::with_capacity(cfg.rounds);
    let mut round_losses = Vec::with_capacity(cfg.rounds);

    let mut online_ladder = ladder_template.clone();
    // The input-ladder state each domain was left in, saved at the end of its
    // visit and restored to ask the retention question in that domain's own
    // context. Before a domain's first visit there is nothing to restore, so
    // the short prefix stands in -- nothing has been learned to retain then
    // anyway.
    let mut domain_ctx: Vec<Option<Vec<f64>>> = vec![None; cfg.domains];
    let mut domain_nodes: Vec<Option<Vec<f64>>> = vec![None; cfg.domains];
    let mut probe_ladder = ladder_template.clone();

    for r in 1..=cfg.rounds {
        let mut round_pre_acc_sum = 0.0;
        let mut round_pre_loss_sum = 0.0;

        // d indexes facts, walks, prefixes and the saved contexts together;
        // zipping four collections reads worse than the index does.
        #[allow(clippy::needless_range_loop)]
        for d in 0..cfg.domains {
            let domain_facts = &stream.facts[d];
            bank.set_domain(d);
            // The frozen 0-shot probe needs this domain's own context, so the
            // per-domain snapshot is restored before it. But that snapshot
            // carries `class_now` itself, out of a slot keyed by the true
            // domain index -- so letting it stand would hand the system its
            // class at every visit instead of making it infer one. The natural
            // state is kept here and put back before training, so the restore
            // serves the measurement only, which is what it was introduced for.
            let displaced = domain_nodes[d].clone().map(|snap| bank.topo_borrow(&snap));
            match &domain_ctx[d] {
                Some(snap) => probe_ladder.restore_state(snap),
                None => {
                    probe_ladder.reset();
                    for &tok in &stream.prefixes[d] {
                        probe_ladder.step(tok);
                    }
                }
            }

            // 1. Frozen 0-Shot probe (and full Multi-Range Few-Shot recovery probe in final round)
            let (
                pre_acc,
                pre_loss,
                acc_few5,
                acc_few10,
                acc_few20,
                acc_few50,
                acc_few200,
                spec_hub,
                spec_mid,
                spec_tail,
            ) = if r == cfg.rounds {
                let few = FrozenProbeSlice::evaluate_few_shot(
                    domain_facts,
                    &probe_ladder,
                    &bank,
                    0.1,
                    cfg.seed + (r * 100 + d) as u64,
                );
                (
                    few.acc_0shot,
                    few.loss_0shot,
                    few.acc_few5,
                    few.acc_few10,
                    few.acc_few20,
                    few.acc_few50,
                    few.acc_few200,
                    few.spec_hub,
                    few.spec_mid,
                    few.spec_tail,
                )
            } else {
                // One fixed gap is best served by whichever single time constant
                // matches it, so multi-scale context can show no advantage
                // under it -- the evaluation decides the outcome, not the
                // mechanism. Mixing horizons within a run is fairer and closer
                // to a real stream.
                let gap_now = if cfg.long_range_mix {
                    const MIX: [usize; 4] = [1, 16, 64, 256];
                    MIX[(r + d) % MIX.len()]
                } else {
                    cfg.long_range_gap
                };
                if gap_now > 0 {
                    // The edge memory's class comes from its own slow context,
                    // not from the input ladder, so rebuilding only the ladder
                    // left the restored -- and therefore oracle -- class in
                    // place. Rebuild both from the presented episode.
                    bank.reset_stream_ctx();
                    bank.absorb_stream_token(stream.cues[d]);
                    for i in 0..gap_now {
                        bank.absorb_stream_token(i % 8);
                    }
                }
                let p = FrozenProbeSlice::evaluate_with_episode(
                    domain_facts,
                    &probe_ladder,
                    &mut bank,
                    if gap_now > 0 { Some(stream.cues[d]) } else { None },
                    gap_now,
                );
                let zero_spec = [p.accuracy; 6];
                (
                    p.accuracy,
                    p.mean_loss,
                    p.accuracy,
                    p.accuracy,
                    p.accuracy,
                    p.accuracy,
                    p.accuracy,
                    zero_spec,
                    zero_spec,
                    zero_spec,
                )
            };

            round_pre_acc_sum += pre_acc;
            round_pre_loss_sum += pre_loss;
            // One machine-readable row per (round, domain): what this regime
            // has cost so far and what it knows. The marginal-cost curve is
            // built from these, so keep the format stable.
            println!(
                "MARGINAL arm={arm:?} round={r} domain={d} writes={:.3} acc={pre_acc:.4}",
                bank.write_for_domain(d)
            );

            // Training resumes from where the stream actually left it, not
            // from the probe's reinstated context.
            if let Some(d) = displaced {
                bank.topo_return(d);
            }

            // The cue opens the visit, so the association between it and this
            // regime is available to anything that integrates the stream.
            if cfg.long_range_gap > 0 && trains(cfg, r, d) {
                online_ladder.step(stream.cues[d]);
                bank.absorb_stream_token(stream.cues[d]);
            }

            // 2. Stream online learning in current domain via continuous random walk
            let domain_walk: &[RelationalFact] = if trains(cfg, r, d) {
                &stream.walks[d]
            } else {
                &[]
            };
            for fact in domain_walk {
                online_ladder.step(fact.entity);
                online_ladder.step(fact.relation);

                bank.observe_fact(
                    fact.entity,
                    fact.relation,
                    online_ladder.normalized_trace(),
                    fact.target,
                    eta,
                );
                online_ladder.step(fact.target);
            }

            // The context this domain is now in, kept for the next round's
            // retention probe as well as the post-training one below.
            // A retired domain trains no further, so it has no new context to
            // save and the post-probe reads it in whatever context the
            // pre-probe already established -- which is the right question to
            // ask of a domain that has stopped being visited.
            if trains(cfg, r, d) {
                domain_ctx[d] = Some(online_ladder.snapshot_state());
                domain_nodes[d] = bank.topo_snapshot();
                if let Some(snap) = &domain_ctx[d] {
                    probe_ladder.restore_state(snap);
                }
            }

            // 3. Frozen probe immediately after training
            let post_probe = FrozenProbeSlice::evaluate(domain_facts, &probe_ladder, &mut bank);

            let retention_gap = post_probe.accuracy - pre_acc;
            let ewc_lambda = match arm {
                ArmKind::ProximalEwc { lambda } => lambda,
                _ => 0.0,
            };

            records.push(TrajectoryRecord {
                arm_name: arm.name(),
                arm_short: arm.short_name().to_string(),
                eta,
                ewc_lambda,
                round: r,
                domain: d,
                pre_revisit_acc: pre_acc,
                pre_revisit_loss: pre_loss,
                acc_few5,
                acc_few10,
                acc_few20,
                acc_few50,
                acc_few200,
                spec_hub,
                spec_mid,
                spec_tail,
                post_train_acc: post_probe.accuracy,
                routing_consistency: bank.routing_consistency().unwrap_or(f64::NAN),
                class_switch_rate: bank.class_switch_rate().unwrap_or(f64::NAN),
                class_collision: bank.class_collision().unwrap_or(f64::NAN),
                classes_live: bank.growth_stats().map(|g| g.0).unwrap_or(f64::NAN),
                growth_steal: bank.growth_stats().map(|g| g.1).unwrap_or(f64::NAN),
                path_entropy_bits: bank.path_entropy().map(|e| e.0).unwrap_or(0.0),
                path_entropy_max: bank.path_entropy().map(|e| e.1).unwrap_or(0.0),
                post_distinct_predictions: post_probe.distinct_predictions,
                post_train_loss: post_probe.mean_loss,
                retention_gap,
            });
        }

        let mean_pre_acc = round_pre_acc_sum / cfg.domains as f64;
        let mean_pre_loss = round_pre_loss_sum / cfg.domains as f64;
        round_accs.push(mean_pre_acc);
        round_losses.push(mean_pre_loss);
    }

    // This path restores a per-domain snapshot at each visit, unlike the
    // Ebbinghaus probe where context flows continuously. If domains separate
    // here but not there, separation is being handed over by the harness
    // rather than inferred, and "allocation" would not be the system's own.
    if let Some((nz, alloc, live)) = bank.readout_occupancy().filter(|(_, a, _)| *a > 0) {
        {
            println!(
                "  [{arm:?}] readout rows occupied {nz}/{alloc} ({:.1}%) across {live} live classes; \
                 {:.0} rows per class",
                100.0 * nz as f64 / alloc as f64,
                nz as f64 / live.max(1) as f64
            );
        }
    }
    if let Some(r) = bank.posterior_move_rate().filter(|r| *r > 0.0) {
        println!(
            "  [{arm:?}] posterior re-routed {:.1}% of writes away from the prior",
            100.0 * r
        );
    }
    if let Some((read, total)) = bank.activation() {
        println!(
            "  [{arm:?}] activation {read}/{total} memory params per token ({:.3}%)",
            100.0 * read as f64 / total.max(1) as f64
        );
    }
    if let Some((mean_w, live)) = bank.edge_write_spread() {
        println!("  [{arm:?}] edge traffic: {live:.0} edges used, {mean_w:.0} writes each");
    }
    if let Some(r) = bank.context_ratio().filter(|r| *r > 0.0) {
        println!("  [{arm:?}] injected context is {r:.2}x the content magnitude");
    }
    if let Some(g) = bank.growth_timing() {
        let tot: usize = g.iter().sum();
        println!(
            "  [{arm:?}] growth events {tot}; since switch: <100 {}  100-499 {}  500-1999 {}  2000+ {}",
            g[0], g[1], g[2], g[3]
        );
    }
    if let Some((rate, wshare, by_bucket)) = bank.intrusion_stats() {
        println!(
            "  [{arm:?}] intrusion {:.1}% of writes / {:.1}% of write magnitude; \
             since switch: <100 {:.1}%  100-499 {:.1}%  500-1999 {:.1}%  2000+ {:.1}%",
            100.0 * rate,
            100.0 * wshare,
            100.0 * by_bucket[0],
            100.0 * by_bucket[1],
            100.0 * by_bucket[2],
            100.0 * by_bucket[3]
        );
    }
    if let Some(m) = bank.domain_class_matrix(cfg.domains) {
        println!("  [{arm:?}] write share by (domain, class):");
        for (d, row) in m.iter().enumerate() {
            let cells: Vec<String> = row.iter().map(|x| format!("{x:5.2}")).collect();
            println!("    domain {d}: {}", cells.join(" "));
        }
    }

    (records, round_accs, round_losses)
}

/// One trial's outputs. Named because the tuple was wide enough to be unreadable.
type TrialResult = (ArmKind, f64, Vec<TrajectoryRecord>, Vec<f64>, Vec<f64>);

/// The same for an EWC trial, keyed by lambda and eta rather than by arm.
type EwcTrialResult = (f64, f64, Vec<TrajectoryRecord>, Vec<f64>, Vec<f64>);

/// Runs the complete SDR benchmark suite across all 5 arms.
/// Distinct targets per domain and cumulative across domains.
///
/// Decides whether sparse per-domain allocation can grow sublinearly. A
/// readout row is only useful to a domain for targets that domain actually
/// produces; at 400 facts against a 4096 vocabulary, 90% of each private block
/// is for tokens the domain never emits. Whether allocating only the used rows
/// buys sublinear growth depends on how fast the UNION of targets grows --
/// Zipf-distributed targets should give Heaps' law, k^alpha with alpha < 1.
pub fn print_target_growth(stream: &RelationalFactStream) {
    use std::collections::HashSet;
    let mut union: HashSet<usize> = HashSet::new();
    println!("  target-set growth (decides whether sparse allocation is sublinear):");
    println!("    domains  own targets  cumulative union  union/domains");
    for (k, facts) in stream.facts.iter().enumerate() {
        let own: HashSet<usize> = facts.iter().map(|f| f.target).collect();
        union.extend(own.iter().copied());
        println!(
            "    {:>7}  {:>11}  {:>16}  {:>13.1}",
            k + 1,
            own.len(),
            union.len(),
            union.len() as f64 / (k + 1) as f64
        );
    }
}

pub fn print_source_checks(c: &SourceChecks) {
    println!();
    println!("SOURCE CHECKS  (properties of the stream and projection, arm-independent)");
    println!(
        "  domain activation overlap: mean {:.3}   (1.000 = same neurons, 0.000 = disjoint)",
        c.mean_overlap
    );
    println!(
        "  domain activation overlap, leaf-only (hub-tier steps excluded): mean {:.3}",
        c.mean_overlap_leaf_only
    );
    if c.mean_overlap > 0.5 {
        println!("  !! domains share most of their activation mass -- the sparse isolation that");
        println!("     DESIGN-SDR lesson 1 makes the whole design depend on has not formed, and a");
        println!(
            "     null result below cannot distinguish a weak mechanism from a missing premise"
        );
    }
    for (a, row) in c.overlap.iter().enumerate() {
        println!(
            "    domain {a}: {}",
            row.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!(
        "  active set retained: {:.3} after 1 step, {:.3} after 100 steps  (chance {:.3})",
        c.retention_1, c.retention_100, 0.0
    );
    if c.retention_1 > 0.9 {
        println!("  !! the code barely turns over -- it is not encoding the input, and the");
        println!("     domain overlap above is a consequence of that rather than a separate fact");
    }
    println!(
        "  neurons ever used: {:.1}% of D",
        100.0 * c.active_fraction
    );
    println!(
        "  win-count concentration: top-1 column {:.1}% of all wins, top-10 {:.1}%, {} columns hold half the mass",
        100.0 * c.top1_win_share,
        100.0 * c.top10_win_share,
        c.cols_for_half_mass
    );
    if c.top10_win_share > 0.10 {
        println!("  !! a handful of fixed columns win top-k for a disproportionate share of all");
        println!("     contexts -- a hubness effect in the random projection, not a domain-specific");
        println!("     signal, and likely the same mechanism behind the domain-overlap numbers above");
    }
    println!("  per-input-rung profile (each rung projected alone, same D and k as the joint one):");
    println!("    rung      tau   cross-domain overlap   within-domain retention@100");
    for k in 0..c.rung_overlap.len() {
        println!(
            "    {:>4}  {:>7.0}   {:>19.3}   {:>27.3}",
            k, c.rung_tau[k], c.rung_overlap[k], c.rung_retention_100[k]
        );
    }
    println!(
        "    (joint, for reference: overlap {:.3}, retention@100 {:.3})",
        c.mean_overlap, c.retention_100
    );
    println!("    a rung worth building a context expert on needs BOTH a low overlap and a high");
    println!("    retention -- low overlap with low retention is noise, not a context code");
    println!(
        "  address return after one domain cycle: {:.3}   (1.000 = a domain comes back to its own addresses)",
        c.cycle_return
    );
    if c.cycle_return < 0.7 {
        println!("  !! addresses drift lap over lap -- what was written last lap is not where this");
        println!("     lap looks for it, so part of the retention loss is structural and no memory");
        println!("     mechanism can repair it");
    }
    println!("  probe-fact exposures over the whole run (min / median / max):");
    for (i, name) in ["hub ", "mid ", "tail"].iter().enumerate() {
        let (lo, md, hi) = c.exposure_by_tier[i];
        println!("    {name}  {lo:>8.0} {md:>8.0} {hi:>8.0}");
    }
    let (_, tail_median, _) = c.exposure_by_tier[2];
    if tail_median < 30.0 {
        println!(
            "  !! tail facts are seen ~{tail_median:.0} times in total -- too few to learn, so a"
        );
        println!("     flat accuracy curve and a learning rate pinned at the grid's top are");
        println!("     expected regardless of architecture");
    }
}

/// One probe reading: which round, tokens elapsed since domain 0 was last
/// trained, accuracy, and mean bits, over domain 0's probe facts.
pub struct EbbinghausPoint {
    pub round: usize,
    pub tokens_since_visit: usize,
    pub accuracy: f64,
    pub bits: f64,
    /// Bits restricted to mid+tail (domain-specific) facts. Mode A's hub
    /// facts are shared, identical edges across every domain, so other
    /// domains keep training them while this one is "away" -- `bits` is
    /// contaminated by that for roughly a third of the probe set, this is
    /// not.
    pub bits_domain_specific: f64,
}

/// Ebbinghaus-shaped decay and savings probe, for one arm at a time.
///
/// v1 probed only at domain boundaries (4 points per cycle at `domains=4`)
/// and read raw accuracy at a fixed delay across rounds. Both were wrong.
///
/// **Resolution.** Domain-boundary delay is measured in "how many domains
/// intervened", not "how much time passed" -- Ebbinghaus's own axis is time
/// since last study. This checkpoints domain 0 at `CHECKPOINTS_PER_DOMAIN`
/// even points *inside* each intervening domain's walk as well as at its
/// end, so delay is reported in actual tokens elapsed and a cycle carries
/// roughly `(domains-1)*CHECKPOINTS_PER_DOMAIN + 1` points instead of
/// `domains`.
///
/// **The confound.** v1's four delay curves moved in lockstep across
/// rounds -- delay 0 (just trained) and delay "one full cycle" were nearly
/// identical at every round -- which means the numbers were dominated by
/// how much of domain 0 had been learned *overall* by that point in
/// training, not by how much was forgotten *this cycle*. Reading a fixed
/// delay's absolute accuracy cannot separate those two. The fix is the same
/// one this project has already learned elsewhere: read the gap, not the
/// absolute number. Every round now also reports
/// `bits(max delay) - bits(delay 0)`, the within-round forgetting, and it is
/// *that* trend across rounds -- not the raw accuracy trend -- that tests
/// savings: forgetting shrinking round over round is the Ebbinghaus-savings
/// signature, independent of how much overall competence is still rising.
pub fn ebbinghaus_probe(
    arm: ArmKind,
    eta: f64,
    cfg: &SdrConfig,
    stream: &RelationalFactStream,
) -> Vec<EbbinghausPoint> {
    const CHECKPOINTS_PER_DOMAIN: usize = 4;

    let mut rng = Rng::new(cfg.seed);
    let ladder_template =
        InputContextLadder::new(cfg.d_input, cfg.vocab, cfg.m_in, cfg.ladder_r, &mut rng);
    let mut bank = ExpertBank::new(cfg, arm, &mut rng);
    let mut online_ladder = ladder_template.clone();
    // Domain 0's own context, saved whenever its visit ends. Probing it from
    // a reset ladder would leave the slow rungs -- the ones carrying the
    // domain code -- effectively uncharged, so the delay curve would measure
    // an off-distribution cue rather than retention.
    let mut ctx0: Option<Vec<f64>> = None;
    // The class state has to come back too, not just the input ladder.
    //
    // Restoring only the ladder left the class to whatever the global context
    // EMA had drifted to -- which, at maximum delay, is the *last* domain
    // trained. Domain 0's facts were then looked up through domain 3's slice,
    // so the number being reported was "what happens when you read a fact
    // under the wrong context", not retention. That is why it swung either
    // way by seed and went negative at d=16: it was reading whatever happened
    // to be in the wrong slice. Same failure as the input-ladder probe bug in
    // DESIGN-SDR appendix C.1, on the axis that was added later.
    let mut nodes0: Option<Vec<f64>> = None;
    let mut probe_ladder = ladder_template.clone();

    let mut points = Vec::new();
    let mut tokens_since_visit = 0usize;
    let started = std::time::Instant::now();
    for r in 1..=cfg.rounds {
        // Progress per round, not silence for the whole run. A round here
        // does a full training pass over every domain plus roughly
        // domains*CHECKPOINTS_PER_DOMAIN frozen evaluations, and at the
        // scale this probe needs to resolve savings (many facts, many
        // rounds) that is not fast enough to run unannounced.
        let secs = started.elapsed().as_secs_f64();
        let eta_secs = if r > 1 {
            secs / (r - 1) as f64 * (cfg.rounds - r + 1) as f64
        } else {
            0.0
        };
        // Write magnitude alongside progress: forgetting that falls only
        // because every write is getting smaller is convergence, not savings.
        let wn = bank
            .take_write_norms()
            .map(|(r, e)| format!(", write |readout| {r:.1} |edge| {e:.1}"))
            .unwrap_or_default();
        println!(
            "  round {r}/{}, {secs:.0}s elapsed, ~{eta_secs:.0}s left{wn}",
            cfg.rounds
        );
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
        for d in 0..cfg.domains {
            // Without this the edge memory's cur_domain stays 0 for the whole
            // probe, so every per-domain diagnostic silently describes domain
            // 0 alone. It changes no behaviour -- cur_domain is read only by
            // instrumentation and by the growth stranding sample.
            bank.set_domain(d);
            let walk = &stream.walks[d];
            // Evenly spaced checkpoints inside this domain's walk, plus its
            // end, at which to probe domain 0 -- but only while domain 0 is
            // NOT the one training (d==0 gets a single delay-0 reading once
            // it finishes, same as before, since there is nothing to
            // checkpoint mid-walk for the domain currently being taught).
            let mut checkpoints: Vec<usize> = if d == 0 {
                vec![walk.len()]
            } else {
                (1..=CHECKPOINTS_PER_DOMAIN)
                    .map(|k| walk.len() * k / CHECKPOINTS_PER_DOMAIN)
                    .collect()
            };
            checkpoints.dedup();
            let mut next_checkpoint = 0usize;

            for (i, fact) in walk.iter().enumerate() {
                online_ladder.step(fact.entity);
                online_ladder.step(fact.relation);
                bank.observe_fact(
                    fact.entity,
                    fact.relation,
                    online_ladder.normalized_trace(),
                    fact.target,
                    eta,
                );
                online_ladder.step(fact.target);
                tokens_since_visit += 3;
                if d == 0 && i + 1 == walk.len() {
                    ctx0 = Some(online_ladder.snapshot_state());
                    nodes0 = bank.topo_snapshot();
                }

                if next_checkpoint < checkpoints.len() && i + 1 == checkpoints[next_checkpoint] {
                    next_checkpoint += 1;
                    let delay = if d == 0 { 0 } else { tokens_since_visit };
                    // Probing domain 0 requires domain 0's context, but the
                    // bank is shared with training, and this restore was never
                    // undone. With four checkpoints a visit the context was
                    // repeatedly dragged back to domain 0, so every domain
                    // ended up writing into domain 0's class and the class
                    // mechanism measured as completely inert. The probe must
                    // observe the run, not steer it.
                    let displaced = nodes0.clone().map(|snap| bank.topo_borrow(&snap));
                    match &ctx0 {
                        Some(snap) => probe_ladder.restore_state(snap),
                        None => {
                            probe_ladder.reset();
                            for &tok in &stream.prefixes[0] {
                                probe_ladder.step(tok);
                            }
                        }
                    }
                    let probe =
                        FrozenProbeSlice::evaluate(&stream.facts[0], &probe_ladder, &mut bank);
                    points.push(EbbinghausPoint {
                        round: r,
                        tokens_since_visit: delay,
                        accuracy: probe.accuracy,
                        bits: probe.mean_loss * std::f64::consts::LOG2_E,
                        bits_domain_specific: probe.mean_loss_domain_specific
                            * std::f64::consts::LOG2_E,
                    });
                    if let Some(d) = displaced {
                        bank.topo_return(d);
                    }
                }
            }
            if d == 0 {
                tokens_since_visit = 0;
            }
        }
    }
    if let Some(m) = bank.domain_class_matrix(cfg.domains) {
        println!("  write share by (domain, class):");
        for (d, row) in m.iter().enumerate() {
            let cells: Vec<String> = row.iter().map(|x| format!("{:5.2}", x)).collect();
            println!("    domain {d}: {}", cells.join(" "));
        }
    }
    if let Some((live, distinct, shared)) = bank.collision_stats() {
        println!(
            "  classes: {live} live, {distinct} distinct home classes for {} domains; \
             {:.1}% of writes land in a class shared by more than one domain",
            cfg.domains,
            100.0 * shared
        );
    }
    if let Some((rate, wshare, by_bucket)) = bank.intrusion_stats() {
        println!(
            "  intrusion: {:.1}% of writes land outside the live domain's home \
             class, carrying {:.1}% of the write magnitude",
            100.0 * rate,
            100.0 * wshare
        );
        println!(
            "    by observations since the domain changed: \
             <100 {:.1}%  100-499 {:.1}%  500-1999 {:.1}%  2000+ {:.1}%",
            100.0 * by_bucket[0],
            100.0 * by_bucket[1],
            100.0 * by_bucket[2],
            100.0 * by_bucket[3]
        );
    }
    points
}

pub fn run_sdr_experiment(cfg: &SdrConfig) -> Vec<ArmSummary> {
    let stream = RelationalFactStream::new(
        cfg.mode,
        cfg.domains,
        cfg.facts_per_domain,
        cfg.span_tokens,
        cfg.rounds,
        cfg.zipf_s,
        cfg.hub_ratio,
        cfg.vocab,
        cfg.seed,
        cfg.target_overlap,
        cfg.target_zipf,
    );

    let eta_grid = if let Some(e) = cfg.eta {
        vec![e]
    } else if let Some(g) = cfg.etas.clone() {
        g
    } else {
        vec![0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    };

    // Taken from the grid actually swept, not from the defaults. These were
    // hardcoded to 0.01/10.0, so every --etas run reported the warning
    // against a grid it did not use: an interior optimum could be flagged as
    // pinned, and -- the dangerous direction -- a genuinely pinned one could
    // report OK.
    let min_grid_eta = eta_grid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_grid_eta = eta_grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let base_arms = [
        ArmKind::Ladder8,
        ArmKind::Ladder4,
        ArmKind::Ladder2,
        ArmKind::Plain,
        ArmKind::Linear,
        ArmKind::LinearFast,
    ];

    // Flatten (arm, eta) tasks for full multi-core CPU utilization
    let mut base_tasks = Vec::with_capacity(base_arms.len() * eta_grid.len());
    for &arm in &base_arms {
        for &eta in &eta_grid {
            base_tasks.push((arm, eta));
        }
    }

    // Progress, because a silent three-hour run is indistinguishable from a
    // stalled one. Nothing was printed between the source checks and the final
    // table, so the only way to tell a slow bench from a dead one was to read
    // process CPU counters over ssh.
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total_base = base_tasks.len();
    let started = std::time::Instant::now();
    let base_results: Vec<TrialResult> = base_tasks
        .into_par_iter()
        .map(|(arm, eta)| {
            let out = run_arm_trial(arm, eta, cfg, &stream);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let secs = started.elapsed().as_secs_f64();
            println!(
                "  [{n}/{total_base}] {} eta={eta} done, {secs:.0}s elapsed, ~{:.0}s left",
                arm.short_name(),
                secs / n as f64 * (total_base - n) as f64
            );
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            let (records, round_accs, round_losses) = out;
            (arm, eta, records, round_accs, round_losses)
        })
        .collect();

    let mut summaries: Vec<ArmSummary> = Vec::with_capacity(base_arms.len() + 1);

    for &arm in &base_arms {
        let arm_trials: Vec<_> = base_results
            .iter()
            .filter(|(a, _, _, _, _)| *a == arm)
            .collect();

        let mut best_acc = -1.0;
        let mut best_few20 = -1.0;
        let mut best_loss = f64::INFINITY;
        let mut best_post_acc = -1.0;
        let mut best_eta = eta_grid[0];
        let mut best_records = Vec::new();
        let mut best_round_accs = Vec::new();
        let mut best_round_losses = Vec::new();

        for (_, eta, records, round_accs, round_losses) in arm_trials {
            let final_acc = *round_accs.last().unwrap_or(&0.0);
            let final_loss = *round_losses.last().unwrap_or(&f64::INFINITY);
            let final_r: Vec<_> = records.iter().filter(|r| r.round == cfg.rounds).collect();
            let n_d = cfg.domains.max(1) as f64;
            let few20_acc = final_r.iter().map(|r| r.acc_few20).sum::<f64>() / n_d;
            let post_acc = final_r.iter().map(|r| r.post_train_acc).sum::<f64>() / n_d;

            let is_better = if (final_loss - best_loss).abs() > 1e-3 {
                final_loss < best_loss
            } else if (final_acc - best_acc).abs() > 1e-4 {
                final_acc > best_acc
            } else if (post_acc - best_post_acc).abs() > 1e-4 {
                post_acc > best_post_acc
            } else {
                few20_acc > best_few20
            };

            if is_better {
                best_acc = final_acc;
                best_few20 = few20_acc;
                best_loss = final_loss;
                best_post_acc = post_acc;
                best_eta = *eta;
                best_records = records.clone();
                best_round_accs = round_accs.clone();
                best_round_losses = round_losses.clone();
            }
        }

        let eta_boundary_hit =
            (best_eta - min_grid_eta).abs() < 1e-6 || (best_eta - max_grid_eta).abs() < 1e-6;
        let final_loss = *best_round_losses.last().unwrap_or(&0.0);
        let mean_gap = best_records
            .iter()
            .filter(|r| r.round > 1)
            .map(|r| r.retention_gap)
            .sum::<f64>()
            / best_records.len().max(1) as f64;

        // Extract Few-Shot Recovery Spectrum in final round
        let final_r_records: Vec<_> = best_records
            .iter()
            .filter(|r| r.round == cfg.rounds)
            .collect();
        let n_domains = cfg.domains.max(1) as f64;

        let acc_0 = final_r_records
            .iter()
            .map(|r| r.pre_revisit_acc)
            .sum::<f64>()
            / n_domains;
        let acc_5 = final_r_records.iter().map(|r| r.acc_few5).sum::<f64>() / n_domains;
        let acc_10 = final_r_records.iter().map(|r| r.acc_few10).sum::<f64>() / n_domains;
        let acc_20 = final_r_records.iter().map(|r| r.acc_few20).sum::<f64>() / n_domains;
        let acc_50 = final_r_records.iter().map(|r| r.acc_few50).sum::<f64>() / n_domains;
        let acc_200 = final_r_records.iter().map(|r| r.acc_few200).sum::<f64>() / n_domains;

        let final_few_spectrum = [acc_0, acc_5, acc_10, acc_20, acc_50, acc_200];

        // Compute stratified spectrums
        let mut hub_spec = [0.0; 6];
        let mut mid_spec = [0.0; 6];
        let mut tail_spec = [0.0; 6];
        for r in &final_r_records {
            for i in 0..6 {
                hub_spec[i] += r.spec_hub[i] / n_domains;
                mid_spec[i] += r.spec_mid[i] / n_domains;
                tail_spec[i] += r.spec_tail[i] / n_domains;
            }
        }

        // Measure Forward Plasticity: Round 1 post-train acquisition vs Final Round post-train acquisition
        let r1_records: Vec<_> = best_records.iter().filter(|r| r.round == 1).collect();
        let r1_post_acc = r1_records.iter().map(|r| r.post_train_acc).sum::<f64>() / n_domains;
        let final_post_acc = final_r_records
            .iter()
            .map(|r| r.post_train_acc)
            .sum::<f64>()
            / n_domains;
        let plasticity_ratio = final_post_acc / r1_post_acc.max(0.01);

        summaries.push(ArmSummary {
            arm,
            best_eta,
            best_ewc_lambda: 0.0,
            eta_boundary_hit,
            ewc_zero_lambda_won: false,
            final_retention_acc: best_acc,
            final_retention_loss: final_loss,
            final_few_spectrum,
            final_hub_spectrum: hub_spec,
            final_mid_spectrum: mid_spec,
            final_tail_spectrum: tail_spec,
            r1_post_acc,
            final_post_acc,
            plasticity_ratio,
            mean_retention_gap: mean_gap,
            round_retention_accs: best_round_accs,
            round_retention_losses: best_round_losses,
            trajectories: best_records,
        });
    }

    // Evaluate Online Proximal EWC arm across lambda and eta grids in parallel
    // Trimmed from five points to three. The first cut (0, 10, 100) was
    // wrong: those two nonzero points already sit on the same saturated
    // trade-off curve in the fixed-eta check, which says nothing about
    // whether the real optimum sits somewhere in the untested (0, 10) gap.
    // Bracketing the low end instead -- where the transition from "off" to
    // "engaged" actually happens -- is where a smaller-than-10 optimum would
    // show up if one exists.
    if cfg.no_ewc {
        return summaries;
    }

    let lambda_grid = if let Some(l) = cfg.ewc_lambda {
        vec![l]
    } else {
        vec![0.0, 1.0, 10.0]
    };

    let mut ewc_tasks = Vec::with_capacity(lambda_grid.len() * eta_grid.len());
    for &lam in &lambda_grid {
        for &eta in &eta_grid {
            ewc_tasks.push((lam, eta));
        }
    }

    // EWC updates the whole `vocab x d_sdr` matrix every step while the ladder
    // arms touch only the k active columns, so these trials cost roughly twenty
    // times more each and dominate the wall clock.
    let ewc_done = std::sync::atomic::AtomicUsize::new(0);
    let total_ewc = ewc_tasks.len();
    let ewc_started = std::time::Instant::now();
    let ewc_trials: Vec<EwcTrialResult> = ewc_tasks
        .into_par_iter()
        .map(|(lam, eta)| {
            let arm = ArmKind::ProximalEwc { lambda: lam };
            let out = run_arm_trial(arm, eta, cfg, &stream);
            let n = ewc_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let secs = ewc_started.elapsed().as_secs_f64();
            println!(
                "  [ewc {n}/{total_ewc}] lambda={lam} eta={eta} done, {secs:.0}s elapsed, ~{:.0}s left",
                secs / n as f64 * (total_ewc - n) as f64
            );
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            let (records, round_accs, round_losses) = out;
            (lam, eta, records, round_accs, round_losses)
        })
        .collect();

    let mut best_ewc_acc = -1.0;
    let mut best_ewc_few20 = -1.0;
    let mut best_ewc_loss = f64::INFINITY;
    let mut best_ewc_post_acc = -1.0;
    let mut best_ewc_lam = 0.0;
    let mut best_ewc_eta = eta_grid[0];
    let mut best_ewc_records = Vec::new();
    let mut best_ewc_round_accs = Vec::new();
    let mut best_ewc_round_losses = Vec::new();

    for (lam, eta, records, round_accs, round_losses) in ewc_trials {
        let final_acc = *round_accs.last().unwrap_or(&0.0);
        let final_loss = *round_losses.last().unwrap_or(&f64::INFINITY);
        let final_r: Vec<_> = records.iter().filter(|r| r.round == cfg.rounds).collect();
        let n_d = cfg.domains.max(1) as f64;
        let few20_acc = final_r.iter().map(|r| r.acc_few20).sum::<f64>() / n_d;
        let post_acc = final_r.iter().map(|r| r.post_train_acc).sum::<f64>() / n_d;

        let is_better = if (final_loss - best_ewc_loss).abs() > 1e-3 {
            final_loss < best_ewc_loss
        } else if (final_acc - best_ewc_acc).abs() > 1e-4 {
            final_acc > best_ewc_acc
        } else if (post_acc - best_ewc_post_acc).abs() > 1e-4 {
            post_acc > best_ewc_post_acc
        } else {
            few20_acc > best_ewc_few20
        };

        if is_better {
            best_ewc_acc = final_acc;
            best_ewc_few20 = few20_acc;
            best_ewc_loss = final_loss;
            best_ewc_post_acc = post_acc;
            best_ewc_lam = lam;
            best_ewc_eta = eta;
            best_ewc_records = records;
            best_ewc_round_accs = round_accs;
            best_ewc_round_losses = round_losses;
        }
    }

    let ewc_eta_boundary_hit =
        (best_ewc_eta - min_grid_eta).abs() < 1e-6 || (best_ewc_eta - max_grid_eta).abs() < 1e-6;
    let ewc_zero_lambda_won = best_ewc_lam == 0.0;
    let final_ewc_loss = *best_ewc_round_losses.last().unwrap_or(&0.0);
    let mean_ewc_gap = best_ewc_records
        .iter()
        .filter(|r| r.round > 1)
        .map(|r| r.retention_gap)
        .sum::<f64>()
        / best_ewc_records.len().max(1) as f64;

    let final_r_ewc: Vec<_> = best_ewc_records
        .iter()
        .filter(|r| r.round == cfg.rounds)
        .collect();
    let n_domains = cfg.domains.max(1) as f64;

    let ewc_0 = final_r_ewc.iter().map(|r| r.pre_revisit_acc).sum::<f64>() / n_domains;
    let ewc_5 = final_r_ewc.iter().map(|r| r.acc_few5).sum::<f64>() / n_domains;
    let ewc_10 = final_r_ewc.iter().map(|r| r.acc_few10).sum::<f64>() / n_domains;
    let ewc_20 = final_r_ewc.iter().map(|r| r.acc_few20).sum::<f64>() / n_domains;
    let ewc_50 = final_r_ewc.iter().map(|r| r.acc_few50).sum::<f64>() / n_domains;
    let ewc_200 = final_r_ewc.iter().map(|r| r.acc_few200).sum::<f64>() / n_domains;

    let ewc_few_spectrum = [ewc_0, ewc_5, ewc_10, ewc_20, ewc_50, ewc_200];

    let mut ewc_hub_spec = [0.0; 6];
    let mut ewc_mid_spec = [0.0; 6];
    let mut ewc_tail_spec = [0.0; 6];
    for r in &final_r_ewc {
        for i in 0..6 {
            ewc_hub_spec[i] += r.spec_hub[i] / n_domains;
            ewc_mid_spec[i] += r.spec_mid[i] / n_domains;
            ewc_tail_spec[i] += r.spec_tail[i] / n_domains;
        }
    }

    let r1_ewc: Vec<_> = best_ewc_records.iter().filter(|r| r.round == 1).collect();
    let r1_ewc_post = r1_ewc.iter().map(|r| r.post_train_acc).sum::<f64>() / n_domains;
    let final_ewc_post = final_r_ewc.iter().map(|r| r.post_train_acc).sum::<f64>() / n_domains;
    let ewc_plasticity_ratio = final_ewc_post / r1_ewc_post.max(0.01);

    summaries.push(ArmSummary {
        arm: ArmKind::ProximalEwc {
            lambda: best_ewc_lam,
        },
        best_eta: best_ewc_eta,
        best_ewc_lambda: best_ewc_lam,
        eta_boundary_hit: ewc_eta_boundary_hit,
        ewc_zero_lambda_won,
        final_retention_acc: best_ewc_acc,
        final_retention_loss: final_ewc_loss,
        final_few_spectrum: ewc_few_spectrum,
        final_hub_spectrum: ewc_hub_spec,
        final_mid_spectrum: ewc_mid_spec,
        final_tail_spectrum: ewc_tail_spec,
        r1_post_acc: r1_ewc_post,
        final_post_acc: final_ewc_post,
        plasticity_ratio: ewc_plasticity_ratio,
        mean_retention_gap: mean_ewc_gap,
        round_retention_accs: best_ewc_round_accs,
        round_retention_losses: best_ewc_round_losses,
        trajectories: best_ewc_records,
    });

    summaries
}

/// Exports trajectory records to a CSV file.
pub fn export_trajectories_csv(
    path: &Path,
    summaries: &[ArmSummary],
) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "arm_short,arm_name,eta,ewc_lambda,round,domain,pre_revisit_acc,pre_revisit_loss,acc_few5,acc_few10,acc_few20,acc_few50,acc_few200,post_train_acc,post_train_loss,retention_gap"
    )?;

    for s in summaries {
        for t in &s.trajectories {
            writeln!(
                writer,
                "{},\"{}\",{:.4},{:.4},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                t.arm_short,
                t.arm_name,
                t.eta,
                t.ewc_lambda,
                t.round,
                t.domain,
                t.pre_revisit_acc,
                t.pre_revisit_loss,
                t.acc_few5,
                t.acc_few10,
                t.acc_few20,
                t.acc_few50,
                t.acc_few200,
                t.post_train_acc,
                t.post_train_loss,
                t.retention_gap,
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Exports experiment summary to a formatted JSON file.
pub fn export_summary_json(
    path: &Path,
    cfg: &SdrConfig,
    summaries: &[ArmSummary],
) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{{")?;
    writeln!(writer, "  \"config\": {{")?;
    writeln!(writer, "    \"mode\": \"{}\",", cfg.mode.name())?;
    writeln!(writer, "    \"domains\": {},", cfg.domains)?;
    writeln!(
        writer,
        "    \"facts_per_domain\": {},",
        cfg.facts_per_domain
    )?;
    writeln!(writer, "    \"span_tokens\": {},", cfg.span_tokens)?;
    writeln!(writer, "    \"rounds\": {},", cfg.rounds)?;
    writeln!(writer, "    \"zipf_s\": {:.4},", cfg.zipf_s)?;
    writeln!(writer, "    \"hub_ratio\": {:.4},", cfg.hub_ratio)?;
    writeln!(writer, "    \"d_sdr\": {},", cfg.d_sdr)?;
    writeln!(writer, "    \"k_active\": {},", cfg.k_active)?;
    writeln!(writer, "    \"m_in\": {}", cfg.m_in)?;
    writeln!(writer, "  }},")?;
    writeln!(writer, "  \"arms\": [")?;

    for (i, s) in summaries.iter().enumerate() {
        let is_last = i + 1 == summaries.len();
        writeln!(writer, "    {{")?;
        writeln!(writer, "      \"name\": \"{}\",", s.arm.name())?;
        writeln!(writer, "      \"short_name\": \"{}\",", s.arm.short_name())?;
        writeln!(writer, "      \"best_eta\": {:.4},", s.best_eta)?;
        writeln!(
            writer,
            "      \"best_ewc_lambda\": {:.4},",
            s.best_ewc_lambda
        )?;
        writeln!(
            writer,
            "      \"eta_boundary_hit\": {},",
            s.eta_boundary_hit
        )?;
        writeln!(
            writer,
            "      \"ewc_zero_lambda_won\": {},",
            s.ewc_zero_lambda_won
        )?;
        writeln!(
            writer,
            "      \"final_0shot_acc\": {:.4},",
            s.final_few_spectrum[0]
        )?;
        writeln!(
            writer,
            "      \"final_few5_acc\": {:.4},",
            s.final_few_spectrum[1]
        )?;
        writeln!(
            writer,
            "      \"final_few10_acc\": {:.4},",
            s.final_few_spectrum[2]
        )?;
        writeln!(
            writer,
            "      \"final_few20_acc\": {:.4},",
            s.final_few_spectrum[3]
        )?;
        writeln!(
            writer,
            "      \"final_few50_acc\": {:.4},",
            s.final_few_spectrum[4]
        )?;
        writeln!(
            writer,
            "      \"final_few200_acc\": {:.4},",
            s.final_few_spectrum[5]
        )?;
        writeln!(
            writer,
            "      \"hub_0shot\": {:.4},",
            s.final_hub_spectrum[0]
        )?;
        writeln!(
            writer,
            "      \"hub_few20\": {:.4},",
            s.final_hub_spectrum[3]
        )?;
        writeln!(
            writer,
            "      \"tail_0shot\": {:.4},",
            s.final_tail_spectrum[0]
        )?;
        writeln!(
            writer,
            "      \"tail_few20\": {:.4},",
            s.final_tail_spectrum[3]
        )?;
        writeln!(
            writer,
            "      \"plasticity_ratio\": {:.4},",
            s.plasticity_ratio
        )?;
        writeln!(
            writer,
            "      \"final_retention_loss\": {:.4},",
            s.final_retention_loss
        )?;
        writeln!(
            writer,
            "      \"mean_retention_gap\": {:.4}",
            s.mean_retention_gap
        )?;
        write!(writer, "    }}")?;
        if is_last {
            writeln!(writer)?;
        } else {
            writeln!(writer, ",")?;
        }
    }

    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;
    writer.flush()?;
    Ok(())
}

/// Prints comprehensive human-readable ASCII summary tables of benchmark results.
pub fn print_ascii_summary(cfg: &SdrConfig, summaries: &[ArmSummary]) {
    println!(
        "\n========================================================================================="
    );
    println!(
        "                     SDR Continual Learning Benchmark Results                             "
    );
    println!(
        "========================================================================================="
    );
    println!("Stream Mode: {}", cfg.mode.name());
    println!(
        "Zipf-s: {:.2} | Hub-Ratio: {:.2} | Domains: {} | Facts/Domain: {} | Span: {} tokens | Rounds: {}",
        cfg.zipf_s, cfg.hub_ratio, cfg.domains, cfg.facts_per_domain, cfg.span_tokens, cfg.rounds
    );
    println!(
        "D_sdr: {} | k_active: {} | D_input: {} | M_in: {} | Vocab: {}",
        cfg.d_sdr, cfg.k_active, cfg.d_input, cfg.m_in, cfg.vocab
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!(
        "{:<24} | {:<8} | {:<10} | {:<10} | {:<10} | {:<12} | Status",
        "Arm Name", "Best Eta", "EWC Lambda", "0-Shot Acc", "Few-20 Acc", "Plasticity"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );

    for s in summaries {
        let status = if s.eta_boundary_hit {
            "WARN: eta boundary"
        } else if s.ewc_zero_lambda_won && matches!(s.arm, ArmKind::ProximalEwc { .. }) {
            "WARN: lambda=0 won"
        } else if s.plasticity_ratio < 0.5 {
            "ALERT: Intransigence"
        } else {
            "OK"
        };

        println!(
            "{:<24} | {:<8.4} | {:<10.2} | {:>5.1}    % | {:>5.1}    % | {:>5.1}      % | {}",
            s.arm.name(),
            s.best_eta,
            s.best_ewc_lambda,
            s.final_few_spectrum[0] * 100.0,
            s.final_few_spectrum[3] * 100.0,
            s.plasticity_ratio * 100.0,
            status
        );
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    if summaries
        .iter()
        .any(|s| s.trajectories.iter().any(|r| r.path_entropy_max > 0.0))
    {
        println!("Routing spread (entropy / max) and consistency (probe returns to training node):");
        for s in summaries {
            let last: Vec<_> = s
                .trajectories
                .iter()
                .filter(|r| r.round == cfg.rounds)
                .collect();
            let h = last.iter().map(|r| r.path_entropy_bits).sum::<f64>() / last.len().max(1) as f64;
            let m = last.first().map(|r| r.path_entropy_max).unwrap_or(0.0);
            let c = last.iter().map(|r| r.routing_consistency).sum::<f64>()
                / last.len().max(1) as f64;
            println!(
                "  {:<24} {:>6.2} / {:>5.2} bits   consistency {:>5.3}   class-switch/token {:>7.5}   sharing {:>4.2}   classes-live {:>4.1}   growth-steal {:>5.3}",
                s.arm.short_name(),
                h,
                m,
                c,
                last.iter().map(|r| r.class_switch_rate).sum::<f64>()
                    / last.len().max(1) as f64,
                last.iter().map(|r| r.class_collision).sum::<f64>() / last.len().max(1) as f64,
                last.iter().map(|r| r.classes_live).sum::<f64>() / last.len().max(1) as f64,
                last.iter().map(|r| r.growth_steal).sum::<f64>() / last.len().max(1) as f64
            );
        }
        println!(
            "-----------------------------------------------------------------------------------------"
        );
    }
    println!("Distinct targets predicted over the probe set (collapse check; {} facts probed per domain):", cfg.facts_per_domain);
    for s in summaries {
        let final_r: Vec<_> = s
            .trajectories
            .iter()
            .filter(|r| r.round == cfg.rounds)
            .collect();
        let mean = final_r
            .iter()
            .map(|r| r.post_distinct_predictions as f64)
            .sum::<f64>()
            / final_r.len().max(1) as f64;
        println!("  {:<24} {:>8.1}", s.arm.short_name(), mean);
    }
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    if cfg.retire_after > 0 {
        let half = cfg.domains / 2;
        println!(
            "Retirement (domains 0..{half} stop training after round {}):",
            cfg.retire_after
        );
        println!("  {:<24} {:>10} {:>10}", "", "retired", "active");
        for s in summaries {
            let f: Vec<_> = s
                .trajectories
                .iter()
                .filter(|r| r.round == cfg.rounds)
                .collect();
            let m = |ret: bool| -> f64 {
                let v: Vec<f64> = f
                    .iter()
                    .filter(|r| (r.domain < half) == ret)
                    .map(|r| r.pre_revisit_acc)
                    .collect();
                100.0 * v.iter().sum::<f64>() / v.len().max(1) as f64
            };
            println!(
                "  {:<24} {:>9.1}% {:>9.1}%",
                s.arm.short_name(),
                m(true),
                m(false)
            );
        }
        println!(
            "-----------------------------------------------------------------------------------------"
        );
    }
    println!("Table 1: Evolution of 0-Shot Retention Accuracy over Revisit Rounds:");
    print!("{:<6} | ", "Round");
    for s in summaries {
        print!("{:<14} | ", s.arm.short_name());
    }
    println!();
    println!(
        "-----------------------------------------------------------------------------------------"
    );

    for r in 0..cfg.rounds {
        print!("R{:<5} | ", r + 1);
        for s in summaries {
            let acc = s.round_retention_accs.get(r).copied().unwrap_or(0.0);
            let loss = s.round_retention_losses.get(r).copied().unwrap_or(0.0);
            print!("{:>5.1}% ({:>4.2}) | ", acc * 100.0, loss);
        }
        println!();
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!("Table 2a: Top-20% Hub Facts (High Revisit / Deep Consolidation):");
    println!(
        "{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Arm Short", "0-Shot", "5-Shot", "10-Shot", "20-Shot", "50-Shot", "200-Shot"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for s in summaries {
        let spec = s.final_hub_spectrum;
        println!(
            "{:<15} | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  %",
            s.arm.short_name(),
            spec[0] * 100.0,
            spec[1] * 100.0,
            spec[2] * 100.0,
            spec[3] * 100.0,
            spec[4] * 100.0,
            spec[5] * 100.0,
        );
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!("Table 2b: Mid-30% Domain Facts (Medium Revisit / Intermediate Consolidation):");
    println!(
        "{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Arm Short", "0-Shot", "5-Shot", "10-Shot", "20-Shot", "50-Shot", "200-Shot"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for s in summaries {
        let spec = s.final_mid_spectrum;
        println!(
            "{:<15} | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  %",
            s.arm.short_name(),
            spec[0] * 100.0,
            spec[1] * 100.0,
            spec[2] * 100.0,
            spec[3] * 100.0,
            spec[4] * 100.0,
            spec[5] * 100.0,
        );
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!("Table 2c: Tail-50% Leaf Facts (Low Revisit / Rapid Few-Shot Retrieval):");
    println!(
        "{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Arm Short", "0-Shot", "5-Shot", "10-Shot", "20-Shot", "50-Shot", "200-Shot"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for s in summaries {
        let spec = s.final_tail_spectrum;
        println!(
            "{:<15} | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  %",
            s.arm.short_name(),
            spec[0] * 100.0,
            spec[1] * 100.0,
            spec[2] * 100.0,
            spec[3] * 100.0,
            spec[4] * 100.0,
            spec[5] * 100.0,
        );
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!("Table 2d: Overall Blended Multi-Scale Few-Shot Recovery Spectrum:");
    println!(
        "{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}",
        "Arm Short", "0-Shot", "5-Shot", "10-Shot", "20-Shot", "50-Shot", "200-Shot"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for s in summaries {
        let spec = s.final_few_spectrum;
        println!(
            "{:<15} | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  % | {:>5.1}  %",
            s.arm.short_name(),
            spec[0] * 100.0,
            spec[1] * 100.0,
            spec[2] * 100.0,
            spec[3] * 100.0,
            spec[4] * 100.0,
            spec[5] * 100.0,
        );
    }

    println!(
        "-----------------------------------------------------------------------------------------"
    );
    println!(
        "Table 3: Forward Plasticity & Intransigence Evaluation (New Domain Learning Efficiency):"
    );
    println!(
        "{:<15} | {:<16} | {:<16} | {:<16} | {:<12}",
        "Arm Short", "R1 Acquisition", "R_End Acquisition", "Plasticity Ratio", "Intransigence"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for s in summaries {
        let intransigent = if s.plasticity_ratio < 0.5 {
            "YES (Locked)"
        } else {
            "NO (Healthy)"
        };
        println!(
            "{:<15} | {:>5.1}            % | {:>5.1}            % | {:>5.1}            % | {:<12}",
            s.arm.short_name(),
            s.r1_post_acc * 100.0,
            s.final_post_acc * 100.0,
            s.plasticity_ratio * 100.0,
            intransigent
        );
    }
    println!(
        "========================================================================================="
    );
}

/// Formats and exports all experiment results to disk, and prints the summary.
pub fn export_and_print_results(
    cfg: &SdrConfig,
    summaries: &[ArmSummary],
    out: &Path,
) -> Result<(), std::io::Error> {
    fs::create_dir_all(out)?;
    let csv_path = out.join("probe_trajectories.csv");
    let json_path = out.join("summary.json");

    export_trajectories_csv(&csv_path, summaries)?;
    export_summary_json(&json_path, cfg, summaries)?;

    print_ascii_summary(cfg, summaries);

    println!("Results saved to: {}", out.display());
    println!("  - Trajectories: {}", csv_path.display());
    println!("  - Summary JSON: {}", json_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zipf_graph_stream_generation_mode_a() {
        let stream =
            RelationalFactStream::new(StreamMode::ModeA, 4, 20, 200, 3, 1.0, 0.10, 512, 20260817, 0.0, false);
        assert_eq!(stream.facts.len(), 4);
        assert_eq!(stream.facts[0].len(), 20);
        assert_eq!(stream.walks.len(), 4);
        assert_eq!(stream.walks[0].len(), 200 / 3);
    }

    #[test]
    fn zipf_graph_stream_generation_mode_b() {
        let stream =
            RelationalFactStream::new(StreamMode::ModeB, 4, 32, 200, 3, 1.0, 0.20, 512, 20260817, 0.0, false);
        assert_eq!(stream.facts.len(), 4);
        assert_eq!(stream.facts[0].len(), 32);
        assert_eq!(stream.walks.len(), 4);
        assert_eq!(stream.prefixes.len(), 4);
        assert_eq!(stream.prefixes[0].len(), 16);
    }

    fn probe_test_cfg() -> SdrConfig {
        SdrConfig {
            mode: StreamMode::ModeA,
            domains: 2,
            facts_per_domain: 5,
            span_tokens: 30,
            rounds: 2,
            vocab: 512,
            d_input: 16,
            m_in: 4,
            d_sdr: 32,
            k_active: 4,
            ladder_r: 2.0,
            zipf_s: 1.0,
            hub_ratio: 0.10,
            eta: Some(0.1),
            ewc_lambda: Some(1.0),
            seed: 20260817,
            experts: 1,
            target_overlap: 0.0,
            target_zipf: false,
            long_range_gap: 0,
            long_range_mix: false,
            no_ewc: false,
            etas: None,
            ladder_g1: 0.1,
            tensor_d2: 0,
            tensor_k2: 2,
            tensor_split: 0,
            rotate: false,
            rotate_gain: 1.0,
            topo_nodes: 0,
            topo_shortcuts: 2,
            topo_hops: 3,
            topo_payload: 32,
            topo_forget: 0.0,
            topo_expect: 0.01,
            topo_crowd: 1.0,
            topo_keep: 4,
            edge_nodes: 0,
            edge_shortcuts: 2,
            edge_hops: 3,
            edge_classes: 8,
            edge_dim: 32,
            edge_forget: 0.0,
            edge_hash_class: false,
            edge_class_readout: false,
            edge_expand: 0,
            edge_ctx_gain: 0.0,
            edge_grow_orth: true,
            edge_route_compose: false,
            edge_share: false,
            edge_share_readout: false,
            edge_posterior: false,
            edge_neg_samples: 0,
            edge_grow_hold: 1,
            edge_gate: 0.0,
            edge_clip: 0.0,
            edge_rungs: 1,
            edge_ladder_visits: 1.0,
            edge_init_classes: 8,
            edge_grow_k: 3.0,
            arrival: 0,
            retire_after: 0,
            out: PathBuf::from("results/test_sdr_probe"),
        }
    }

    #[test]
    fn frozen_probe_slice_does_not_modify_memory() {
        let stream =
            RelationalFactStream::new(StreamMode::ModeA, 2, 10, 100, 2, 1.0, 0.10, 512, 20260817, 0.0, false);
        let mut rng = Rng::new(20260817);
        let ladder = InputContextLadder::new(16, 512, 4, 2.0, &mut rng);
        let cfg = SdrConfig {
            d_input: 16,
            vocab: 512,
            m_in: 4,
            d_sdr: 64,
            k_active: 8,
            experts: 1,
            ..probe_test_cfg()
        };
        let mut bank = ExpertBank::new(&cfg, ArmKind::Plain, &mut rng);

        let w_before = bank.fast_weights_snapshot();
        let _ = FrozenProbeSlice::evaluate(&stream.facts[0], &ladder, &mut bank);
        let w_after = bank.fast_weights_snapshot();

        assert_eq!(
            w_before, w_after,
            "frozen probe must not modify memory state"
        );
    }

    #[test]
    fn small_sdr_experiment_smoke_test() {
        let cfg = SdrConfig {
            mode: StreamMode::ModeA,
            domains: 2,
            facts_per_domain: 5,
            span_tokens: 30,
            rounds: 2,
            vocab: 512,
            d_input: 16,
            m_in: 4,
            d_sdr: 32,
            k_active: 4,
            ladder_r: 2.0,
            zipf_s: 1.0,
            hub_ratio: 0.10,
            eta: Some(0.1),
            ewc_lambda: Some(1.0),
            seed: 20260817,
            experts: 1,
            target_overlap: 0.0,
            target_zipf: false,
            long_range_gap: 0,
            long_range_mix: false,
            no_ewc: false,
            etas: None,
            ladder_g1: 0.1,
            tensor_d2: 0,
            tensor_k2: 2,
            tensor_split: 0,
            rotate: false,
            rotate_gain: 1.0,
            topo_nodes: 0,
            topo_shortcuts: 2,
            topo_hops: 3,
            topo_payload: 32,
            topo_forget: 0.0,
            topo_expect: 0.01,
            topo_crowd: 1.0,
            topo_keep: 4,
            edge_nodes: 0,
            edge_shortcuts: 2,
            edge_hops: 3,
            edge_classes: 8,
            edge_dim: 32,
            edge_forget: 0.0,
            edge_hash_class: false,
            edge_class_readout: false,
            edge_expand: 0,
            edge_ctx_gain: 0.0,
            edge_grow_orth: true,
            edge_route_compose: false,
            edge_share: false,
            edge_share_readout: false,
            edge_posterior: false,
            edge_neg_samples: 0,
            edge_grow_hold: 1,
            edge_gate: 0.0,
            edge_clip: 0.0,
            edge_rungs: 1,
            edge_ladder_visits: 1.0,
            edge_init_classes: 8,
            edge_grow_k: 3.0,
            arrival: 0,
            retire_after: 0,
            out: PathBuf::from("results/test_sdr_smoke"),
        };

        let summaries = run_sdr_experiment(&cfg);
        assert_eq!(summaries.len(), 7); // 6 base arms + 1 EWC arm

        for s in &summaries {
            assert_eq!(s.round_retention_accs.len(), 2);
            assert!(s.final_retention_acc >= 0.0 && s.final_retention_acc <= 1.0);
            assert!(s.final_few_spectrum[0] >= 0.0);
        }
    }
}
