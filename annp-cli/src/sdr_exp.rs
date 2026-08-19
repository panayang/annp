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
    ) -> Self {
        assert!(domains > 0, "domains must be positive");
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
            let mut tail_edges = Vec::new();
            for (leaf_idx, &u) in domain_leaf_entities.iter().enumerate() {
                let mut pick_val = rng.next_f64() * sum_leaf_w;
                let mut chosen_v = domain_leaf_entities[0];
                for (&v, &w) in domain_leaf_entities.iter().zip(&d_leaf_weights) {
                    if pick_val <= w {
                        chosen_v = v;
                        break;
                    }
                    pick_val -= w;
                }
                if chosen_v == u && domain_leaf_entities.len() > 1 {
                    chosen_v = domain_leaf_entities[(leaf_idx + 1) % domain_leaf_entities.len()];
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
    ) -> Self {
        let shared_entities_count = facts_per_domain.max(32);
        let shared_rel_count = 4;
        let rel_base = shared_entities_count;
        let targets_base = rel_base + shared_rel_count;
        let total_required = targets_base + domains * shared_entities_count;

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
                // Domain-specific target mapping (100% collision on [entity, rel] across domains)
                let target = targets_base
                    + d * shared_entities_count
                    + ((e * 7 + 13 + d * 31) % shared_entities_count);

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
    pub fn evaluate(
        facts: &[RelationalFact],
        prefix: &[usize],
        ladder_template: &InputContextLadder,
        proj: &RandomProjection,
        memory: &mut SdrMemory,
    ) -> ProbeResult {
        let mut u_buf = vec![0.0; proj.d_sdr()];
        let mut pairs_buf = Vec::with_capacity(proj.d_sdr());
        let mut active = Vec::with_capacity(proj.k());
        let mut alphas = Vec::with_capacity(proj.k());

        let mut correct = 0;
        let mut correct_hub = 0;
        let mut count_hub = 0;
        let mut correct_mid = 0;
        let mut count_mid = 0;
        let mut correct_tail = 0;
        let mut count_tail = 0;

        let mut total_loss = 0.0;
        let mut online_ladder = ladder_template.clone();

        for fact in facts {
            online_ladder.reset();
            for &tok in prefix {
                online_ladder.step(tok);
            }
            online_ladder.step(fact.entity);
            online_ladder.step(fact.relation);

            let z = online_ladder.normalized_trace();
            proj.project_and_select(z, &mut u_buf, &mut pairs_buf, &mut active, &mut alphas);

            let (loss, is_correct) = memory.predict(&active, &alphas, fact.target);
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
        let domain = facts.first().map(|f| f.domain).unwrap_or(0);

        ProbeResult {
            domain,
            accuracy,
            acc_hub,
            acc_mid,
            acc_tail,
            mean_loss,
            count,
        }
    }

    /// Evaluates multi-timescale few-shot recovery curve (0, 5, 10, 20, 50, 200 steps)
    /// across stratified Zipf rank tiers on isolated clones of memory state.
    pub fn evaluate_few_shot(
        facts: &[RelationalFact],
        prefix: &[usize],
        ladder_template: &InputContextLadder,
        proj: &RandomProjection,
        memory: &SdrMemory,
        eta: f64,
        seed: u64,
    ) -> FewShotResult {
        let mut sim_rng = Rng::new(seed);
        let base_probe = Self::evaluate(facts, prefix, ladder_template, proj, &mut memory.clone());

        let budgets = [5, 10, 20, 50, 200];
        let mut few_accs = [base_probe.accuracy; 5];
        let mut few_hub = [base_probe.acc_hub; 5];
        let mut few_mid = [base_probe.acc_mid; 5];
        let mut few_tail = [base_probe.acc_tail; 5];

        let mut u_buf = vec![0.0; proj.d_sdr()];
        let mut pairs_buf = Vec::with_capacity(proj.d_sdr());
        let mut active = Vec::with_capacity(proj.k());
        let mut alphas = Vec::with_capacity(proj.k());

        for (idx, &budget) in budgets.iter().enumerate() {
            let mut temp_mem = memory.clone();
            let mut online_ladder = ladder_template.clone();

            for _ in 0..budget {
                let fact_idx = sim_rng.next_below(facts.len() as u64) as usize;
                let fact = facts[fact_idx];

                online_ladder.reset();
                for &tok in prefix {
                    online_ladder.step(tok);
                }
                online_ladder.step(fact.entity);
                online_ladder.step(fact.relation);

                let z = online_ladder.normalized_trace();
                proj.project_and_select(z, &mut u_buf, &mut pairs_buf, &mut active, &mut alphas);

                temp_mem.observe(&active, &alphas, fact.target, eta);
            }

            let probe = Self::evaluate(facts, prefix, ladder_template, proj, &mut temp_mem);
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
    ProximalEwc { lambda: f64 },
}

impl ArmKind {
    pub fn name(&self) -> String {
        match self {
            ArmKind::Ladder8 => "Ladder-8 (m=8)".to_string(),
            ArmKind::Ladder4 => "Ladder-4 (m=4)".to_string(),
            ArmKind::Ladder2 => "Ladder-2 (m=2)".to_string(),
            ArmKind::Plain => "Plain (m=1)".to_string(),
            ArmKind::ProximalEwc { lambda } => format!("Proximal EWC (lam={:.1})", lambda),
        }
    }

    pub fn short_name(&self) -> &'static str {
        match self {
            ArmKind::Ladder8 => "ladder_8",
            ArmKind::Ladder4 => "ladder_4",
            ArmKind::Ladder2 => "ladder_2",
            ArmKind::Plain => "plain_m1",
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
    pub out: PathBuf,
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
    pub exposure_by_tier: [(f64, f64, f64); 3],
    pub active_fraction: f64,
    /// Fraction of the active set retained from one step to the next, and from
    /// a hundred steps earlier. A code that does not turn over is not encoding
    /// the input, and then a high domain overlap and a small set of neurons
    /// ever used are both consequences of that rather than separate problems.
    pub retention_1: f64,
    pub retention_100: f64,
}

pub fn measure_source(cfg: &SdrConfig, stream: &RelationalFactStream) -> SourceChecks {
    let mut rng = Rng::new(cfg.seed);
    let ladder = InputContextLadder::new(cfg.d_input, cfg.vocab, cfg.m_in, cfg.ladder_r, &mut rng);
    let proj = RandomProjection::new(ladder.total_dim(), cfg.d_sdr, cfg.k_active, &mut rng);

    let mut activity = vec![vec![0.0f64; cfg.d_sdr]; cfg.domains];
    let mut u_buf = vec![0.0; cfg.d_sdr];
    let mut pairs = Vec::with_capacity(cfg.d_sdr);
    let mut active = Vec::with_capacity(cfg.k_active);
    let mut alphas = Vec::with_capacity(cfg.k_active);
    let mut walk_ladder = ladder.clone();
    let mut history: std::collections::VecDeque<Vec<usize>> = std::collections::VecDeque::new();
    let (mut keep1, mut keep100, mut n1, mut n100) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    // One full pass of the cycling stream, exactly as training sees it.
    for _ in 0..cfg.rounds {
        for (act, walk) in activity.iter_mut().zip(&stream.walks) {
            for fact in walk {
                walk_ladder.step(fact.entity);
                walk_ladder.step(fact.relation);
                proj.project_and_select(
                    walk_ladder.normalized_trace(),
                    &mut u_buf,
                    &mut pairs,
                    &mut active,
                    &mut alphas,
                );
                for &i in &active {
                    act[i] += 1.0;
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
                walk_ladder.step(fact.target);
            }
        }
    }

    // Shared activation mass between each pair of domains.
    let norm: Vec<Vec<f64>> = activity
        .iter()
        .map(|a| {
            let s: f64 = a.iter().sum::<f64>().max(f64::MIN_POSITIVE);
            a.iter().map(|v| v / s).collect()
        })
        .collect();
    let mut overlap = vec![vec![0.0; cfg.domains]; cfg.domains];
    let (mut off_sum, mut off_n) = (0.0, 0.0);
    for a in 0..cfg.domains {
        for b in 0..cfg.domains {
            let o: f64 = norm[a].iter().zip(&norm[b]).map(|(x, y)| x.min(*y)).sum();
            overlap[a][b] = o;
            if a != b {
                off_sum += o;
                off_n += 1.0;
            }
        }
    }
    let used: f64 = norm
        .iter()
        .map(|p| p.iter().filter(|v| **v > 0.0).count() as f64)
        .sum::<f64>()
        / (cfg.domains * cfg.d_sdr) as f64;

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

    SourceChecks {
        overlap,
        mean_overlap: if off_n > 0.0 { off_sum / off_n } else { 0.0 },
        exposure_by_tier,
        active_fraction: used,
        retention_1: if n1 > 0.0 { keep1 / n1 } else { f64::NAN },
        retention_100: if n100 > 0.0 { keep100 / n100 } else { f64::NAN },
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
    let proj = RandomProjection::new(
        ladder_template.total_dim(),
        cfg.d_sdr,
        cfg.k_active,
        &mut rng,
    );

    let g1 = 0.1;
    let schedule = Schedule::Geometric {
        r: cfg.ladder_r,
        g1,
    };

    let mut memory = match arm {
        ArmKind::Ladder8 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 8, schedule),
        ArmKind::Ladder4 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 4, schedule),
        ArmKind::Ladder2 => SdrMemory::new_ladder(cfg.vocab, cfg.d_sdr, 2, schedule),
        ArmKind::Plain => SdrMemory::new_plain(cfg.vocab, cfg.d_sdr),
        // Anchor/Fisher follow rate. Its inverse is EWC's memory timescale, and
        // it has to span what the stream asks it to hold. The old value 0.05
        // gives tau=20 steps against a domain visit of span_tokens/3 (~6666)
        // and a full cycle of domains*span_tokens/3 (~26664): by five time
        // constants (~100 steps) into a new domain the anchor has already
        // converged onto it and forgotten the one before, so the penalty pulls
        // toward a moving target rather than toward old-domain knowledge --
        // indistinguishable from no anchor at all. Same failure as the
        // hardcoded ladder g1 fixed earlier and the grow.rs EWC trail fixed
        // days ago; this call site was never updated to match.
        ArmKind::ProximalEwc { lambda } => {
            let cycle = (cfg.domains * (cfg.span_tokens / 3)).max(1) as f64;
            SdrMemory::new_ewc(cfg.vocab, cfg.d_sdr, lambda, 1.0 / cycle)
        }
    };

    let mut records = Vec::new();
    let mut round_accs = Vec::with_capacity(cfg.rounds);
    let mut round_losses = Vec::with_capacity(cfg.rounds);

    let mut u_buf = vec![0.0; cfg.d_sdr];
    let mut pairs_buf = Vec::with_capacity(cfg.d_sdr);
    let mut active = Vec::with_capacity(cfg.k_active);
    let mut alphas = Vec::with_capacity(cfg.k_active);

    let mut online_ladder = ladder_template.clone();

    for r in 1..=cfg.rounds {
        let mut round_pre_acc_sum = 0.0;
        let mut round_pre_loss_sum = 0.0;

        for d in 0..cfg.domains {
            let domain_facts = &stream.facts[d];
            let domain_prefix = &stream.prefixes[d];

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
                    domain_prefix,
                    &ladder_template,
                    &proj,
                    &memory,
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
                let p = FrozenProbeSlice::evaluate(
                    domain_facts,
                    domain_prefix,
                    &ladder_template,
                    &proj,
                    &mut memory.clone(),
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

            // 2. Stream online learning in current domain via continuous random walk
            let domain_walk = &stream.walks[d];
            for fact in domain_walk {
                online_ladder.step(fact.entity);
                online_ladder.step(fact.relation);

                let z = online_ladder.normalized_trace();
                proj.project_and_select(z, &mut u_buf, &mut pairs_buf, &mut active, &mut alphas);

                memory.observe(&active, &alphas, fact.target, eta);
                online_ladder.step(fact.target);
            }

            // 3. Frozen probe immediately after training
            let post_probe = FrozenProbeSlice::evaluate(
                domain_facts,
                domain_prefix,
                &ladder_template,
                &proj,
                &mut memory,
            );

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
                post_train_loss: post_probe.mean_loss,
                retention_gap,
            });
        }

        let mean_pre_acc = round_pre_acc_sum / cfg.domains as f64;
        let mean_pre_loss = round_pre_loss_sum / cfg.domains as f64;
        round_accs.push(mean_pre_acc);
        round_losses.push(mean_pre_loss);
    }

    (records, round_accs, round_losses)
}

/// One trial's outputs. Named because the tuple was wide enough to be unreadable.
type TrialResult = (ArmKind, f64, Vec<TrajectoryRecord>, Vec<f64>, Vec<f64>);

/// The same for an EWC trial, keyed by lambda and eta rather than by arm.
type EwcTrialResult = (f64, f64, Vec<TrajectoryRecord>, Vec<f64>, Vec<f64>);

/// Runs the complete SDR benchmark suite across all 5 arms.
pub fn print_source_checks(c: &SourceChecks) {
    println!();
    println!("SOURCE CHECKS  (properties of the stream and projection, arm-independent)");
    println!(
        "  domain activation overlap: mean {:.3}   (1.000 = same neurons, 0.000 = disjoint)",
        c.mean_overlap
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
    );

    let eta_grid = if let Some(e) = cfg.eta {
        vec![e]
    } else {
        vec![0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    };

    let min_grid_eta = 0.01;
    let max_grid_eta = 10.0;

    let base_arms = [
        ArmKind::Ladder8,
        ArmKind::Ladder4,
        ArmKind::Ladder2,
        ArmKind::Plain,
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
            RelationalFactStream::new(StreamMode::ModeA, 4, 20, 200, 3, 1.0, 0.10, 512, 20260817);
        assert_eq!(stream.facts.len(), 4);
        assert_eq!(stream.facts[0].len(), 20);
        assert_eq!(stream.walks.len(), 4);
        assert_eq!(stream.walks[0].len(), 200 / 3);
    }

    #[test]
    fn zipf_graph_stream_generation_mode_b() {
        let stream =
            RelationalFactStream::new(StreamMode::ModeB, 4, 32, 200, 3, 1.0, 0.20, 512, 20260817);
        assert_eq!(stream.facts.len(), 4);
        assert_eq!(stream.facts[0].len(), 32);
        assert_eq!(stream.walks.len(), 4);
        assert_eq!(stream.prefixes.len(), 4);
        assert_eq!(stream.prefixes[0].len(), 16);
    }

    #[test]
    fn frozen_probe_slice_does_not_modify_memory() {
        let stream =
            RelationalFactStream::new(StreamMode::ModeA, 2, 10, 100, 2, 1.0, 0.10, 512, 20260817);
        let mut rng = Rng::new(20260817);
        let ladder = InputContextLadder::new(16, 512, 4, 2.0, &mut rng);
        let proj = RandomProjection::new(ladder.total_dim(), 64, 8, &mut rng);
        let mut mem = SdrMemory::new_plain(512, 64);

        let w_before = mem.read_fast_weights().as_slice().to_vec();
        let _ = FrozenProbeSlice::evaluate(
            &stream.facts[0],
            &stream.prefixes[0],
            &ladder,
            &proj,
            &mut mem,
        );
        let w_after = mem.read_fast_weights().as_slice().to_vec();

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
            out: PathBuf::from("results/test_sdr_smoke"),
        };

        let summaries = run_sdr_experiment(&cfg);
        assert_eq!(summaries.len(), 5); // 4 base arms + 1 EWC arm

        for s in &summaries {
            assert_eq!(s.round_retention_accs.len(), 2);
            assert!(s.final_retention_acc >= 0.0 && s.final_retention_acc <= 1.0);
            assert!(s.final_few_spectrum[0] >= 0.0);
        }
    }
}
