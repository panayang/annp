//! ANNP command line.
//!
//! Every subcommand is an experiment. Each writes CSVs plus a `manifest.json`
//! recording the git revision and every parameter that went into the numbers,
//! so any result in the paper can be regenerated from one line.

mod baseline;
mod corpus;
mod e0;
mod grow;
mod head;
mod next;

/// Git revision, or `unknown` outside a checkout.
pub fn git_revision() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=12"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Record what produced a result directory: the revision, the exact argv, and
/// the resolved configuration.
///
/// Argv alone would be the "reproduce from one line" claim, but it is not
/// enough on its own, because a run that relied on a default is only
/// reproducible while that default holds — and the defaults in this file have
/// already drifted away from the configuration the design document calls
/// current (they were three versions stale when this was written). So the
/// resolved config goes in too, via `Debug`, which means a field added later is
/// recorded without anyone remembering to update this function. The hand-written
/// per-field manifest in `e0.rs` is exactly what rots.
pub fn write_manifest(out_dir: &Path, experiment: &str, resolved: &impl std::fmt::Debug) {
    let argv: Vec<String> = std::env::args().collect();
    let body = format!(
        "{{\n  \"experiment\": {experiment:?},\n  \"git\": {:?},\n  \"argv\": {:?},\n  \"resolved\": {:?}\n}}\n",
        git_revision(),
        argv,
        format!("{resolved:#?}"),
    );
    if let Err(e) = std::fs::create_dir_all(out_dir)
        .and_then(|_| std::fs::write(out_dir.join("manifest.json"), body))
    {
        eprintln!("could not write manifest to {}: {e}", out_dir.display());
    }
}

mod run;
mod topology;

use std::path::{Path, PathBuf};

use annp_core::model::IngressMode;
use annp_core::node::AbsorbRule;
use clap::{Parser, Subcommand, ValueEnum};

/// CLI spelling of `AbsorbRule`, kept separate so the core crate does not
/// depend on clap.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum AbsorbArg {
    /// Constant logit of zero. Reproduces the measurements that condemned it.
    Fixed,
    /// Move on only if a neighbour expects this better than the current node.
    Relative,
    /// As `relative`, less the node's own surprise.
    Surprise,
}

/// CLI spelling of `IngressMode`.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum IngressArg {
    /// Phase of the embedding in two fixed random planes.
    Content,
    /// One fixed anchor for every token.
    Constant,
    /// A cursor advancing one node per token.
    Cursor,
    /// Where the token's own mass last came to rest.
    Readout,
}

impl From<IngressArg> for IngressMode {
    fn from(i: IngressArg) -> Self {
        match i {
            IngressArg::Content => IngressMode::Content,
            IngressArg::Constant => IngressMode::Constant,
            IngressArg::Cursor => IngressMode::Cursor,
            IngressArg::Readout => IngressMode::Readout,
        }
    }
}

impl From<AbsorbArg> for AbsorbRule {
    fn from(a: AbsorbArg) -> Self {
        match a {
            AbsorbArg::Fixed => AbsorbRule::FixedReference,
            AbsorbArg::Relative => AbsorbRule::Relative,
            AbsorbArg::Surprise => AbsorbRule::RelativeSurprise,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "annp", about = "ANNP experiments", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// E0 — consolidation ladder bench, with no network attached.
    ///
    /// Decides whether the multi-timescale ladder of DESIGN.md §1.8 earns its
    /// place before any network code is written.
    E0 {
        /// Matrix side length of the associative memory.
        #[arg(long, default_value_t = 64)]
        d: usize,
        /// Interfering writes after the probe; also the retention horizon that
        /// sizes every ladder.
        #[arg(long, default_value_t = 30_000)]
        horizon: u64,
        /// Writes before the probe, to reach the interference steady state.
        #[arg(long, default_value_t = 4_000)]
        warmup: u64,
        /// Independent paired trials averaged per scheme.
        #[arg(long, default_value_t = 8)]
        trials: u64,
        /// Log-spaced ages at which retention is sampled. Resolving the
        /// predicted ripple of period ln r needs several points per period.
        #[arg(long, default_value_t = 48)]
        age_samples: usize,
        /// Events to follow the bare impulse response for in E0-d.
        #[arg(long, default_value_t = 1_000_000)]
        impulse_horizon: u64,
        /// Never-written keys used to estimate the noise distribution.
        #[arg(long, default_value_t = 128)]
        decoys: usize,
        /// Delta-rule step size. At 1.0 with unit keys a write is exact.
        #[arg(long, default_value_t = 1.0)]
        eta: f64,
        /// Pattern bank size for the Zipf capacity run.
        #[arg(long, default_value_t = 2_000)]
        patterns: usize,
        /// Zipf exponent for revisit frequencies.
        #[arg(long, default_value_t = 1.0)]
        zipf_exponent: f64,
        /// Uniform-ladder conductance. Must stay below 0.5 for stability.
        #[arg(long, default_value_t = 0.25)]
        g_uniform: f64,
        /// Geometric-ladder base conductance. Must stay below 1.0.
        #[arg(long, default_value_t = 0.5)]
        g1_geometric: f64,
        #[arg(long, default_value_t = 20_260_806)]
        seed: u64,
        #[arg(long, default_value = "results/e0")]
        out: PathBuf,
    },
    /// The growing tree of nodes on a synthetic Zipf source, with an online
    /// EWC control. See DESIGN-TREE.md.
    Grow {
        #[arg(long, default_value_t = 200_000)]
        tokens: usize,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
        /// Kept small on purpose: slicing was dropped in favour of lowering the
        /// source dimension, so one path carries the whole input.
        #[arg(long, default_value_t = 32)]
        d_model: usize,
        #[arg(long, default_value_t = 8)]
        domains: usize,
        #[arg(long, default_value_t = 3000)]
        domain_span: usize,
        /// Window width in strides. 1 is disjoint, `domains` is fully shared;
        /// between the two the domains overlap, which is what gives both
        /// interference and a content signature.
        #[arg(long, default_value_t = 2.0)]
        domain_width: f64,
        #[arg(long, default_value_t = 1.0)]
        zipf_s: f64,
        /// State-dependent tilt on the Zipf marginal; zero makes the current
        /// symbol uninformative.
        #[arg(long, default_value_t = 1.0)]
        tilt: f64,
        #[arg(long, default_value_t = 2)]
        fanout: usize,
        /// Depth bounds compute per token and, with fanout, the node count.
        #[arg(long, default_value_t = 6)]
        depth: usize,
        /// Ladder rungs per node. Swept on the whole tree, not on one node.
        #[arg(long, default_value_t = 4)]
        rungs: usize,
        #[arg(long, default_value_t = 2.0)]
        ladder_r: f64,
        #[arg(long, default_value_t = 0.3)]
        eta: f64,
        /// Run the online EWC control instead of the tree.
        #[arg(long)]
        ewc: bool,
        #[arg(long, default_value_t = 20_260_816)]
        seed: u64,
        #[arg(long, default_value = "results/grow")]
        out: PathBuf,
    },
    /// A learned baseline on the same stream and the same protocol, so the
    /// architecture has something to be compared against that is not an
    /// ablation of itself.
    Baseline {
        #[arg(long, default_value_t = 200_000)]
        tokens: usize,
        #[arg(long, default_value_t = 4096)]
        vocab: usize,
        /// How many previous tokens the model sees. Three is the architecture's
        /// own measured window (§40.6), so it is the matched comparison.
        #[arg(long, default_value_t = 3)]
        window: usize,
        #[arg(long, default_value_t = 128)]
        d_model: usize,
        #[arg(long, default_value_t = 256)]
        hidden: usize,
        #[arg(long, default_value_t = 2)]
        order: usize,
        #[arg(long, default_value_t = 3)]
        fanout: usize,
        #[arg(long, default_value_t = 20_260_807)]
        seed: u64,
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "results/baseline")]
        out: PathBuf,
    },
    /// Candidate A: rotation-addressed context with ladder persistence, and
    /// the 2x2 that separates addressing from persistence.
    Next {
        #[arg(long, default_value_t = 200_000)]
        tokens: usize,
        #[arg(long, default_value_t = 4096)]
        vocab: usize,
        /// Accumulator width. Must be even: it is d/2 rotation planes.
        #[arg(long, default_value_t = 256)]
        d_model: usize,
        /// 363 matches the window MLP's trained parameter count at vocab 4096
        /// and d 256 to within 1,643 of 1,675,520. The input is 2d wide: the
        /// context state plus a lossless copy of the current token.
        #[arg(long, default_value_t = 363)]
        hidden: usize,
        /// Sizes both halves at once: the slowest rotation period and, through
        /// `rungs_for_horizon`, the number of ladder rungs.
        #[arg(long, default_value_t = 1024.0)]
        horizon: f64,
        /// Replace the ladder with a single exponential decay of the same
        /// nominal horizon. This is candidate C's arm.
        #[arg(long)]
        no_ladder: bool,
        /// Remove the rotation. This is what DESIGN.md's architecture was:
        /// persistence with no way to address it.
        #[arg(long)]
        no_addressing: bool,
        /// Frequencies uniform on (0, pi] instead of geometric in period.
        #[arg(long)]
        linear_spacing: bool,
        /// Drop the memory entirely and predict from the current token alone.
        /// This is the order-1 control, run through the same head.
        #[arg(long)]
        no_memory: bool,
        /// Above 1, cycle the stream through this many independent chains and
        /// report retention per revisit instead of a single loss.
        #[arg(long, default_value_t = 1)]
        domains: usize,
        #[arg(long, default_value_t = 4000)]
        domain_span: usize,
        /// Rungs on the readout head's weights. Absent is plain SGD; this is
        /// the arm that tests Benna-Fusi where Benna-Fusi belongs.
        #[arg(long)]
        consolidate: Option<usize>,
        /// Rung 1's conductance; its inverse is the leak time. Defaults to
        /// 1/domain_span so the fast rung holds one visit's learning.
        #[arg(long)]
        consolidate_g1: Option<f64>,
        /// Split the hidden layer into this many content-routed groups, one
        /// active per token. This is candidate B.
        #[arg(long, default_value_t = 1)]
        experts: usize,
        /// Route on the context state only, so the gate reads something that
        /// varies slowly and carries the domain.
        #[arg(long)]
        gate_on_state: bool,
        /// Width of each domain's alphabet window in strides. 1 is disjoint,
        /// `domains` is fully shared, between the two they overlap.
        #[arg(long, default_value_t = 2.0)]
        domain_width: f64,
        /// Run the single ladder node: one matrix, a nonlinear local write, no
        /// hidden layer, no backpropagation.
        #[arg(long)]
        node: bool,
        /// Deliberately more rungs than the horizon needs -- redundancy against
        /// timescales we have not thought of. Costs memory, not forward compute.
        #[arg(long, default_value_t = 8)]
        node_rungs: usize,
        /// Geometric ratio of the node's ladder. E0-b found r=8 with three
        /// rungs matched r=2 with eight, on three times the memory rather than
        /// eight, and recalled more overall.
        #[arg(long, default_value_t = 2.0)]
        node_r: f64,
        #[arg(long, default_value_t = 2)]
        order: usize,
        #[arg(long, default_value_t = 3)]
        fanout: usize,
        #[arg(long, default_value_t = 20_260_807)]
        seed: u64,
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "results/next")]
        out: PathBuf,
    },
    /// End-to-end run on a synthetic Markov source with a known entropy rate.
    Run {
        #[arg(long, default_value_t = 20_000)]
        tokens: usize,
        #[arg(long, default_value_t = 16)]
        vocab: usize,
        /// Markov order. At 1 the current token is a sufficient statistic and
        /// no context-using model can beat a context-free one; above 1 context
        /// is worth something.
        #[arg(long, default_value_t = 2)]
        order: usize,
        /// Independent sources cycled through. Above 1 runs the
        /// continual-learning test.
        #[arg(long, default_value_t = 1)]
        domains: usize,
        /// Tokens spent in each domain before switching.
        #[arg(long, default_value_t = 5000)]
        domain_span: usize,
        /// Replace one domain with an unseen chain at this fraction through the
        /// run, to separate domain-specific retention from general improvement.
        #[arg(long, default_value_t = 0.0)]
        fresh_domain_at: f64,
        /// Needle-in-a-haystack probes for the lost-in-the-middle test: key and
        /// value pairs taught once at spread-out positions and all queried at
        /// the end.
        #[arg(long, default_value_t = 0)]
        needles: usize,
        /// Times each needle is taught. Raise it as a positive control.
        #[arg(long, default_value_t = 1)]
        needle_repeats: usize,
        /// Shared key-symbol pool size. Two or more makes the needle a pair of
        /// keys, which an order-1 model provably cannot resolve.
        #[arg(long, default_value_t = 0)]
        needle_key_symbols: usize,
        /// Subtract the current token's running mean from the assembled vector
        /// before the readout.
        /// Components the lookup readout evaluates per token. 0 keeps the ones
        /// above a uniform share; a positive value keeps exactly that many.
        #[arg(long, default_value_t = 0)]
        head_top_k: usize,
        /// Past keys each node keeps for the key-echo diagnostic. 0 is off.
        /// Ladder rung the forward pass reads. 0 is the fastest, which is what
        /// the design has always used.
        /// Give each particle an immutable query and a fuel tank. Nodes answer
        /// the query instead of their own context, so every node on a route
        /// answers the same question, and a hop costs fuel linearly instead of
        /// dividing the particle's weight away.
        #[arg(long)]
        carry_query: bool,
        /// Weight a deposit by how much of the arriving payload that node's
        /// memory explained, not only by how much mass stopped there.
        #[arg(long)]
        confidence_weighted: bool,
        /// Deposit the payload and the node's answer into separate halves of
        /// the reassembly buffer instead of summing them.
        #[arg(long)]
        split_deposit: bool,
        /// One child down the strongest option, carrying the whole mass,
        /// instead of one child per edge sharing it out.
        #[arg(long)]
        walk: bool,
        /// Hops after which a walker is absorbed. Placeholder termination.
        #[arg(long, default_value_t = 64)]
        hop_cap: u64,
        #[arg(long, default_value_t = 0)]
        read_rung: usize,
        #[arg(long, default_value_t = 0)]
        key_echo: usize,
        /// Probe how far back the assembled vector still names a token.
        #[arg(long)]
        lag_probe: bool,
        #[arg(long)]
        centre_readout: bool,
        /// JSON corpus of `input_text` records. Requires --tokenizer.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// SentencePiece model used to segment --corpus.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Successors per context. Lower means a more predictable source.
        #[arg(long, default_value_t = 3)]
        fanout: usize,
        #[arg(long, default_value_t = 16)]
        d_head: usize,
        /// Particles per token. d_model = slots * d_head must be a power of two.
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 1024)]
        nodes: usize,
        #[arg(long, default_value_t = 4)]
        long_range: usize,
        /// Long-range contact exponent; 0 makes them distance-independent, so
        /// the ring metric contributes nothing to routing. The default follows
        /// the lattice dimension, 1 on a ring; it was 2 for the torus, which on
        /// a ring is past the critical point.
        #[arg(long, default_value_t = 1.0)]
        exponent: f64,
        #[arg(long, default_value_t = 4)]
        rungs: usize,
        /// Timescales in a node's context key. 1 is the last input alone,
        /// which is exactly the behaviour before this existed.
        #[arg(long, default_value_t = 6)]
        context_scales: usize,
        /// Run linear probes along the path and report where context lives.
        #[arg(long, default_value_t = false)]
        probe: bool,
        /// Rungs behind the output table; 1 means a plain matrix.
        #[arg(long, default_value_t = 1)]
        embed_rungs: usize,
        #[arg(long, default_value_t = 1e-3)]
        mass_floor: f64,
        #[arg(long, default_value_t = 1.0)]
        eta: f64,
        #[arg(long, default_value_t = 1.0)]
        learning_rate: f64,
        /// Ladder ratio r. E0-d measured the usable range as 2..=8.
        #[arg(long, default_value_t = 4.0)]
        ladder_ratio: f64,
        #[arg(long, default_value_t = 20_260_807)]
        seed: u64,
        /// Seed for topology and model initialisation only; the source and the
        /// token stream stay on `--seed`. Vary this alone to measure run-to-run
        /// variance at fixed data.
        #[arg(long, default_value_t = 20_260_807)]
        structure_seed: u64,
        /// Control run: score the input embedding instead of the network's
        /// output, with everything else identical.
        #[arg(long, default_value_t = false)]
        bypass: bool,
        /// Freeze the topology instead of rewiring long-range contacts.
        #[arg(long, default_value_t = false)]
        frozen_topology: bool,
        /// Rewire to the first candidate drawn rather than the least-visited of
        /// several. Separates found structure from mere randomisation.
        #[arg(long, default_value_t = false)]
        blind_turnover: bool,
        /// Tie the output head to the embedding table. Provably cannot express
        /// an asymmetric bigram (§11.2), so it is off by default and this flag
        /// exists to reproduce the tied comparison.
        #[arg(long, default_value_t = false)]
        tied: bool,
        /// Slots in the lookup readout. Zero selects the linear head. This is a
        /// genuine new hyperparameter — the capacity of the readout — with no
        /// derivation behind it yet (DESIGN.md §23.5).
        #[arg(long, default_value_t = 512)]
        head_slots: usize,

        /// Admit one token per tick regardless of what is still in flight.
        /// Leaks future tokens into earlier predictions; the difference against
        /// the default serial protocol is the size of that leak.
        #[arg(long, default_value_t = false)]
        overlapped: bool,
        /// How a token's entry point is chosen.
        #[arg(long, value_enum, default_value_t = IngressArg::Cursor)]
        ingress: IngressArg,
        /// How a node decides whether to forward a particle at all.
        #[arg(long, value_enum, default_value_t = AbsorbArg::Surprise)]
        absorb: AbsorbArg,
        #[arg(long, default_value = "results/run")]
        out: PathBuf,
    },
    /// Topology bench — does the long-range exponent follow the lattice
    /// dimension, as DESIGN.md §1.9 claims, or was 2 just asserted?
    Topology {
        /// Ring lengths to sweep; the scaling fit needs at least three.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "1024,2048,4096,8192,16384"
        )]
        sizes: Vec<usize>,
        /// Long-range exponents to compare against Kleinberg's formulas.
        #[arg(long, value_delimiter = ',', default_value = "0,1,1.5,2,2.5,3,4")]
        exponents: Vec<f64>,
        /// Contact counts to compare at the critical exponent.
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8")]
        contacts: Vec<usize>,
        /// Random source/target pairs routed per configuration.
        #[arg(long, default_value_t = 4_000)]
        pairs: u64,
        #[arg(long, default_value_t = 20_260_807)]
        seed: u64,
        #[arg(long, default_value = "results/topology")]
        out: PathBuf,
    },
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::E0 {
            d,
            horizon,
            warmup,
            trials,
            age_samples,
            impulse_horizon,
            decoys,
            eta,
            patterns,
            zipf_exponent,
            g_uniform,
            g1_geometric,
            seed,
            out,
        } => {
            assert!(
                trials >= 2,
                "need at least two trials to report a standard error"
            );
            let cfg = e0::Config {
                d,
                horizon,
                warmup,
                trials,
                age_samples,
                impulse_horizon,
                decoys,
                eta,
                patterns,
                zipf_exponent,
                seed,
                g_uniform,
                g1_geometric,
            };
            e0::run(&cfg, &out)
        }
        Command::Grow {
            tokens,
            vocab,
            d_model,
            domains,
            domain_span,
            domain_width,
            zipf_s,
            tilt,
            fanout,
            depth,
            rungs,
            ladder_r,
            eta,
            ewc,
            seed,
            out,
        } => grow::run(
            &grow::Config {
                tokens,
                vocab,
                d_model,
                domains,
                domain_span,
                domain_width,
                zipf_s,
                tilt,
                fanout,
                depth,
                rungs,
                ladder_r,
                eta,
                ewc,
                seed,
            },
            &out,
        ),
        Command::Baseline {
            tokens,
            vocab,
            window,
            d_model,
            hidden,
            order,
            fanout,
            seed,
            corpus,
            tokenizer,
            out,
        } => baseline::run(
            &baseline::Config {
                tokens,
                vocab,
                window,
                d_model,
                hidden,
                order,
                fanout,
                seed,
                corpus,
                tokenizer,
            },
            &out,
        ),
        Command::Next {
            tokens,
            vocab,
            d_model,
            hidden,
            horizon,
            no_ladder,
            no_addressing,
            linear_spacing,
            no_memory,
            domains,
            domain_span,
            consolidate,
            consolidate_g1,
            experts,
            gate_on_state,
            domain_width,
            node,
            node_rungs,
            node_r,
            order,
            fanout,
            seed,
            corpus,
            tokenizer,
            out,
        } => next::run(
            &next::Config {
                tokens,
                vocab,
                d_model,
                hidden,
                horizon,
                ladder: !no_ladder,
                addressing: !no_addressing,
                memory: !no_memory,
                domains,
                domain_span,
                consolidate,
                consolidate_g1,
                experts,
                gate_on_state,
                domain_width,
                node,
                node_rungs,
                node_r,
                linear_spacing,
                order,
                fanout,
                seed,
                corpus,
                tokenizer,
            },
            &out,
        ),
        Command::Run {
            tokens,
            vocab,
            order,
            domains,
            domain_span,
            fresh_domain_at,
            needles,
            needle_repeats,
            needle_key_symbols,
            head_top_k,
            carry_query,
            confidence_weighted,
            split_deposit,
            walk,
            hop_cap,
            read_rung,
            key_echo,
            lag_probe,
            centre_readout,
            corpus,
            tokenizer,
            fanout,
            d_head,
            slots,
            nodes,
            long_range,
            exponent,
            rungs,
            context_scales,
            probe,
            embed_rungs,
            mass_floor,
            eta,
            learning_rate,
            ladder_ratio,
            seed,
            structure_seed,
            bypass,
            frozen_topology,
            blind_turnover,
            tied,
            head_slots,
            overlapped,
            ingress,
            absorb,
            out,
        } => {
            let cfg = run::Config {
                tokens,
                vocab,
                order,
                domains,
                domain_span,
                fresh_domain_at,
                needles,
                needle_repeats,
                needle_key_symbols,
                head_top_k,
                carry_query,
                confidence_weighted,
                split_deposit,
                walk,
                hop_cap,
                read_rung,
                key_echo,
                lag_probe,
                centre_readout,
                corpus,
                tokenizer,
                fanout,
                d_head,
                slots,
                nodes,
                long_range,
                exponent,
                rungs,
                context_scales,
                probe,
                embed_rungs,
                mass_floor,
                eta,
                learning_rate,
                ladder_ratio,
                seed,
                structure_seed,
                bypass,
                frozen_topology,
                blind_turnover,
                tied,
                head_kind: if head_slots == 0 {
                    annp_core::model::HeadKind::Linear
                } else {
                    annp_core::model::HeadKind::Lookup { slots: head_slots }
                },
                overlapped,
                ingress: ingress.into(),
                absorb: absorb.into(),
            };
            run::run(&cfg, &out)
        }
        Command::Topology {
            sizes,
            exponents,
            contacts,
            pairs,
            seed,
            out,
        } => {
            assert!(
                sizes.len() >= 3,
                "the hops ~ N^beta fit needs at least three ring sizes"
            );
            let cfg = topology::Config {
                sizes,
                exponents,
                contacts,
                pairs,
                seed,
            };
            topology::run(&cfg, &out)
        }
    }
}
