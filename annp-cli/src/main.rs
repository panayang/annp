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
mod edge;
mod sdr_exp;
mod topo;

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
        /// Anchor follow rate; its inverse is EWC's memory timescale. Defaults
        /// to one full domain cycle.
        #[arg(long)]
        ewc_trail: Option<f64>,
        /// Print the source diagnostic and stop.
        #[arg(long)]
        source_only: bool,
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
    /// Continuous SDR with multi-timescale consolidation ladders and relational facts benchmark.
    /// See DESIGN-SDR.md.
    Sdr {
        /// Number of domains to cycle through.
        #[arg(long, default_value_t = 4)]
        domains: usize,
        /// Number of relational facts per domain.
        #[arg(long, default_value_t = 100)]
        facts_per_domain: usize,
        /// Number of stream tokens presented per domain visit (span).
        #[arg(long, default_value_t = 2000)]
        span_tokens: usize,
        /// Number of domain cycling rounds.
        #[arg(long, default_value_t = 5)]
        rounds: usize,
        /// Vocabulary size.
        #[arg(long, default_value_t = 512)]
        vocab: usize,
        /// Input causal trace embedding dimension.
        #[arg(long, default_value_t = 32)]
        d_input: usize,
        /// High-dimensional SDR feature dimension D.
        #[arg(long, default_value_t = 128)]
        d_sdr: usize,
        /// Active sparsity k (k-WTA).
        #[arg(long, default_value_t = 8)]
        k_active: usize,
        /// Input context ladder rung count (replaces arbitrary single gamma).
        #[arg(long, default_value_t = 8)]
        m_in: usize,
        /// Geometric ladder ratio r.
        #[arg(long, default_value_t = 2.0)]
        ladder_r: f64,
        /// Zipf power-law exponent s.
        #[arg(long, default_value_t = 1.0)]
        zipf_s: f64,
        /// Fraction of entities that are global cross-domain hubs. Clamped to
        /// [0.05, 0.40] in the stream generator.
        #[arg(long, default_value_t = 0.10)]
        hub_ratio: f64,
        /// Learning rate eta. If omitted, sweeps grid [0.01, 0.03, 0.1, 0.3, 1.0, 3.0].
        #[arg(long)]
        eta: Option<f64>,
        /// Online EWC regularisation lambda. If omitted, sweeps [0.0, 0.1, 1.0, 10.0, 100.0].
        #[arg(long)]
        ewc_lambda: Option<f64>,
        /// Stream mode: "a" for orthogonal domains (disjoint entities), "b" for shared-entity semantic collision.
        #[arg(long, default_value = "a")]
        mode: String,
        #[arg(long, default_value_t = 20260817)]
        seed: u64,
        /// Print the source checks and stop. They depend only on the stream
        /// and the fixed projection, so they cost nothing next to the arms.
        #[arg(long)]
        source_only: bool,
        /// Run the Ebbinghaus decay-and-savings probe on domain 0 instead of
        /// the full arm sweep: accuracy at every domain boundary within a
        /// round (the decay shape) across rounds (the savings trend). Uses
        /// --eta and defaults to the Ladder-4 arm.
        #[arg(long)]
        ebbinghaus: bool,
        /// Arm for --ebbinghaus: plain, ladder2, ladder4, ladder8.
        #[arg(long, default_value = "ladder4")]
        ebbinghaus_arm: String,
        /// Split the input ladder's rungs across N product-of-experts groups,
        /// each with its own projection and memory, logits summed before one
        /// shared softmax. 1 is the single-projection architecture exactly.
        /// D and k are divided by N so every setting has the same parameter
        /// and active-unit budget.
        #[arg(long, default_value_t = 1)]
        experts: usize,
        /// Skip the EWC sweep. It is most of the wall clock and answers no
        /// question about how content and context should be addressed.
        #[arg(long)]
        no_ewc: bool,
        /// Comma-separated eta grid, replacing the seven-point default.
        #[arg(long, value_delimiter = ',')]
        etas: Option<Vec<f64>>,
        /// Weight-ladder base conductance; tau_k = ladder_r^(2k) / g1 sets the
        /// whole timescale range. Must be < 1 for the explicit-Euler
        /// integration to stay monotone.
        #[arg(long, default_value_t = 0.1)]
        ladder_g1: f64,
        /// Context-axis width for outer-product addressing (0 = off). The
        /// content axis becomes d_sdr / this, so the budget is unchanged.
        #[arg(long, default_value_t = 0)]
        tensor_d2: usize,
        /// Active columns on the context axis; content gets k_active / this.
        #[arg(long, default_value_t = 2)]
        tensor_k2: usize,
        /// Rung where context begins: content is [0, split), context is
        /// [split, m_in). 0 means m_in / 2.
        #[arg(long, default_value_t = 0)]
        tensor_split: usize,
        /// Rotate the content code by a context-generated orthogonal
        /// transform instead of letting the context in additively. Keeps k/D,
        /// and therefore within-fact code stability, untouched while still
        /// separating domains.
        #[arg(long)]
        rotate: bool,
        /// Rotation gain in radians. 0 is the internal control: content-only
        /// projection with no rotation at all.
        #[arg(long, default_value_t = 1.0)]
        rotate_gain: f64,
        /// Nodes in the topological distributed memory (0 = off). The memory
        /// becomes R[node, vocab, d_payload]; at N*d_payload = d_sdr the
        /// parameter count matches the monolithic baseline exactly.
        #[arg(long, default_value_t = 0)]
        topo_nodes: usize,
        /// Long-range contacts per node, on top of the two ring neighbours.
        #[arg(long, default_value_t = 2)]
        topo_shortcuts: usize,
        /// Routing steps taken before the particle is absorbed.
        #[arg(long, default_value_t = 3)]
        topo_hops: usize,
        /// Payload width. topo_nodes * this should equal d_sdr for a matched
        /// parameter budget against the monolithic arms.
        #[arg(long, default_value_t = 32)]
        topo_payload: usize,
        /// Directed forgetting rate: how fast a node's slice decays per unit
        /// of the traffic share it is failing to earn. 0 disables it, which
        /// is the internal control for whether forgetting is doing anything.
        #[arg(long, default_value_t = 0.0)]
        topo_forget: f64,
        /// EMA rate for node expectations -- how fast nodes specialise.
        #[arg(long, default_value_t = 0.01)]
        topo_expect: f64,
        /// How strongly a node's held mass counts against it when routing.
        /// 0 is the internal control: pure rich-get-richer, which collapses.
        #[arg(long, default_value_t = 1.0)]
        topo_crowd: f64,
        /// Nodes that keep a share of the particle's mass after routing.
        /// 1 is the internal control: the hard single-node routing that
        /// scored 0.0% because a fact could not find its way back.
        #[arg(long, default_value_t = 4)]
        topo_keep: usize,
        /// Nodes for the edge-memory arm (0 = off). Routing is fixed at
        /// construction; memory lives on the edges, sliced by class.
        #[arg(long, default_value_t = 0)]
        edge_nodes: usize,
        #[arg(long, default_value_t = 2)]
        edge_shortcuts: usize,
        #[arg(long, default_value_t = 3)]
        edge_hops: usize,
        /// Class slices per edge. The class is inferred online from a slow
        /// context EMA; this is not the number of domains and the system is
        /// never told it.
        #[arg(long, default_value_t = 8)]
        edge_classes: usize,
        #[arg(long, default_value_t = 32)]
        edge_dim: usize,
        /// Directed forgetting: how fast an (edge, class) slice decays per
        /// unit of the traffic share it is failing to earn. 0 is the control.
        #[arg(long, default_value_t = 0.0)]
        edge_forget: f64,
        /// Assign the class by hashing the fact's own tokens instead of
        /// inferring it from context. Same slice count, no context in it --
        /// the control that says whether the class is carrying the regime or
        /// merely widening the address.
        #[arg(long)]
        edge_hash_class: bool,
        /// Give each class its own readout slice. Without it the readout is
        /// shared and is the one place domains still overwrite each other,
        /// however well the edge slices are allocated.
        #[arg(long)]
        edge_class_readout: bool,
        /// Rounds between successive domains entering the stream (0 = all
        /// present from round 1). Needed for the marginal-cost curve: with
        /// every domain present there is no k-th regime to price.
        #[arg(long, default_value_t = 0)]
        arrival: usize,
        /// How many transition-lengths novelty must persist before a class
        /// is allocated. 0 = the old, undebounced behaviour.
        ///
        /// Expressed in transitions rather than observations on purpose: a
        /// domain switch makes similarity dip exactly as novelty does, so the
        /// debounce has to outlast a switch, and how long a switch lasts is a
        /// measured property of the stream (intrusion decays to zero within
        /// about a third of a visit), not a number to pick.
        #[arg(long, default_value_t = 0)]
        edge_grow_hold: usize,
        /// Ceiling on total class slices when capacity is expanded on demand
        /// (0 = fixed budget). Growth then allocates a slice only when a
        /// genuinely novel regime arrives, and the append leaves every
        /// existing weight untouched -- no retraining, unlike widening a
        /// monolithic readout, which changes the projection every code was
        /// Share one edge memory across all classes, keeping the readout
        /// per-class. A payload is built from entity and relation, which are
        /// byte-identical across Mode B's domains, so the transform is common
        /// to every domain and only the target mapping is domain-specific.
        /// Privatising it too gave each new class zeros at every edge.
        #[arg(long)]
        edge_share: bool,
        /// Add a class-common readout block to the per-class one. The
        /// readout is where the parameters are (1.57M against the encoder's
        /// 65K at 12 classes), so it is where "a new class inherits zeros"
        /// mostly still bites, and the source's hub tier is currently
        /// relearned privately by every domain.
        #[arg(long)]
        edge_share_readout: bool,
        /// Choose each write's class by which head scores the observed
        /// target best, rather than by the context prior. Prediction still
        /// uses the prior -- reading may not consult the label. Targets
        /// intrusion, which is 43-44% of write magnitude and which
        /// attenuating writes did not fix.
        #[arg(long)]
        edge_posterior: bool,
        /// written through.
        #[arg(long, default_value_t = 0)]
        edge_expand: usize,
        /// Scale each write by addressing confidence: eta * m/(m+gate), with m
        /// the top-1/top-2 prototype margin. 0 = off. Targets transition
        /// intrusion, which carries 44% of write magnitude and decays to zero
        /// by 2000 observations after a domain switch.
        #[arg(long, default_value_t = 0.0)]
        edge_gate: f64,
        /// Addressing-blind control for --edge-gate: cap the per-write update
        /// norm. Same quantity acted on, no addressing knowledge, so it tells
        /// magnitude control apart from knowing where you are.
        #[arg(long, default_value_t = 0.0)]
        edge_clip: f64,
        /// Benna-Fusi rungs behind each class readout slice (1 = off). Only
        /// meaningful with --edge-class-readout: on a shared readout the deep
        /// rungs would average across domains, which is the configuration
        /// already falsified.
        #[arg(long, default_value_t = 1)]
        edge_rungs: usize,
        /// How many domain visits the first hidden rung averages over. The
        /// conductance is derived from this and the visit length; it is not
        /// settable directly, so it cannot be given in the wrong units.
        #[arg(long, default_value_t = 1.0)]
        edge_ladder_visits: f64,
        /// Classes live at the start; the rest are grown when a genuinely
        /// novel context appears. Equal to --edge-classes disables growth.
        #[arg(long, default_value_t = 8)]
        edge_init_classes: usize,
        /// How many standard deviations below the running mean best-match
        /// counts as a new regime worth its own class.
        #[arg(long, default_value_t = 3.0)]
        edge_grow_k: f64,
        /// Retire the first half of the domains after this round. Creates the
        /// genuine capacity pressure that directed forgetting needs in order
        /// to be worth anything; without it nothing ever becomes obsolete.
        #[arg(long, default_value_t = 0)]
        retire_after: usize,
        #[arg(long, default_value = "results/sdr")]
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
            ewc_trail,
            source_only,
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
                ewc_trail,
                source_only,
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
        Command::Sdr {
            domains,
            facts_per_domain,
            span_tokens,
            rounds,
            vocab,
            d_input,
            d_sdr,
            k_active,
            m_in,
            ladder_r,
            zipf_s,
            hub_ratio,
            eta,
            ewc_lambda,
            mode,
            source_only,
            ebbinghaus,
            ebbinghaus_arm,
            experts,
            no_ewc,
            etas,
            ladder_g1,
            tensor_d2,
            tensor_k2,
            tensor_split,
            rotate,
            rotate_gain,
            topo_nodes,
            topo_shortcuts,
            topo_hops,
            topo_payload,
            topo_forget,
            topo_expect,
            topo_crowd,
            topo_keep,
            edge_nodes,
            edge_shortcuts,
            edge_hops,
            edge_classes,
            edge_dim,
            edge_forget,
            edge_hash_class,
            edge_class_readout,
            arrival,
            edge_share,
            edge_share_readout,
            edge_posterior,
            edge_grow_hold,
            edge_expand,
            edge_gate,
            edge_clip,
            edge_rungs,
            edge_ladder_visits,
            edge_init_classes,
            edge_grow_k,
            retire_after,
            seed,
            out,
        } => {
            let cfg = sdr_exp::SdrConfig {
                mode: sdr_exp::StreamMode::from_str(&mode),
                domains,
                facts_per_domain,
                span_tokens,
                rounds,
                vocab,
                d_input,
                m_in,
                d_sdr,
                k_active,
                ladder_r,
                zipf_s,
                hub_ratio,
                eta,
                ewc_lambda,
                seed,
                experts,
                no_ewc,
                etas,
                ladder_g1,
                tensor_d2,
                tensor_k2,
                tensor_split,
                rotate,
                rotate_gain,
                topo_nodes,
                topo_shortcuts,
                topo_hops,
                topo_payload,
                topo_forget,
                topo_expect,
                topo_crowd,
                topo_keep,
                edge_nodes,
                edge_shortcuts,
                edge_hops,
                edge_classes,
                edge_dim,
                edge_forget,
                edge_hash_class,
                edge_class_readout,
                arrival,
                edge_share,
                edge_share_readout,
                edge_posterior,
                edge_grow_hold,
                edge_expand,
                edge_gate,
                edge_clip,
                edge_rungs,
                edge_ladder_visits,
                edge_init_classes,
                edge_grow_k,
                retire_after,
                out: out.clone(),
            };
            write_manifest(&out, "sdr", &cfg);
            // Manipulation checks first: a null result cannot be read without
            // knowing whether the isolation the design depends on has formed and
            // whether the facts were seen often enough to learn at all.
            let stream_for_checks = sdr_exp::RelationalFactStream::new(
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
            let checks = sdr_exp::measure_source(&cfg, &stream_for_checks);
            sdr_exp::print_source_checks(&checks);
            if source_only {
                return Ok(());
            }
            if ebbinghaus {
                let arm = match ebbinghaus_arm.as_str() {
                    "plain" => sdr_exp::ArmKind::Plain,
                    "ladder2" => sdr_exp::ArmKind::Ladder2,
                    "ladder8" => sdr_exp::ArmKind::Ladder8,
                    _ => sdr_exp::ArmKind::Ladder4,
                };
                let e = cfg.eta.unwrap_or(0.3);
                let points = sdr_exp::ebbinghaus_probe(arm, e, &cfg, &stream_for_checks);
                println!();
                println!("=== Ebbinghaus decay-and-savings probe: {} ===", arm.name());
                println!(
                    "  tokens since domain 0 was last trained -> accuracy -> bits (all / domain-specific), by round"
                );
                println!("  round  tokens   accuracy   bits-all  bits-specific");
                for p in &points {
                    println!(
                        "  {:>5}  {:>6}   {:>8.4}   {:>8.3}  {:>13.3}",
                        p.round, p.tokens_since_visit, p.accuracy, p.bits, p.bits_domain_specific
                    );
                }
                // v2's confound: Mode A's hub facts are the same edges shared
                // identically across every domain, so while domain 0 is
                // "away" the other domains keep training exactly those hub
                // facts through their own walks. A third of domain 0's probe
                // set was therefore never actually away, and the resulting
                // gap can run negative -- accuracy improving with delay --
                // which is what the first run of this probe showed. Restrict
                // to mid+tail (domain-specific) facts for the real signal.
                //
                // The remaining confound (savings vs overall-progress) is
                // handled the same way as before: the within-round gap
                // cancels whatever is common to delay 0 and max delay in the
                // same round, leaving only what changed *during* the gap.
                println!();
                println!("  within-round forgetting, domain-specific bits (max delay minus delay 0):");
                print!("   ");
                for r in 1..=cfg.rounds {
                    let row: Vec<&sdr_exp::EbbinghausPoint> =
                        points.iter().filter(|p| p.round == r).collect();
                    let Some(first) = row.first() else { continue };
                    let Some(last) = row.iter().max_by_key(|p| p.tokens_since_visit) else {
                        continue;
                    };
                    print!(" {:+.3}", last.bits_domain_specific - first.bits_domain_specific);
                }
                println!();
                println!("  (for comparison) within-round forgetting, all facts including shared hub:");
                print!("   ");
                for r in 1..=cfg.rounds {
                    let row: Vec<&sdr_exp::EbbinghausPoint> =
                        points.iter().filter(|p| p.round == r).collect();
                    let Some(first) = row.first() else { continue };
                    let Some(last) = row.iter().max_by_key(|p| p.tokens_since_visit) else {
                        continue;
                    };
                    print!(" {:+.3}", last.bits - first.bits);
                }
                println!();
                return Ok(());
            }
            let summaries = sdr_exp::run_sdr_experiment(&cfg);
            sdr_exp::export_and_print_results(&cfg, &summaries, &out)
        }
    }
}
