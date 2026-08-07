//! ANNP command line.
//!
//! Every subcommand is an experiment. Each writes CSVs plus a `manifest.json`
//! recording the git revision and every parameter that went into the numbers,
//! so any result in the paper can be regenerated from one line.

mod e0;
mod run;
mod topology;

use std::path::PathBuf;

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
        /// Successors per context. Lower means a more predictable source.
        #[arg(long, default_value_t = 3)]
        fanout: usize,
        #[arg(long, default_value_t = 16)]
        d_head: usize,
        /// Particles per token. d_model = slots * d_head must be a power of two.
        #[arg(long, default_value_t = 8)]
        slots: usize,
        #[arg(long, default_value_t = 24)]
        grid_side: usize,
        #[arg(long, default_value_t = 4)]
        long_range: usize,
        #[arg(long, default_value_t = 4)]
        rungs: usize,
        /// Timescales in a node's context key. 1 is the last input alone,
        /// which is exactly the behaviour before this existed.
        #[arg(long, default_value_t = 1)]
        context_scales: usize,
        /// Rungs behind the tied table; 1 means a plain matrix.
        #[arg(long, default_value_t = 3)]
        embed_rungs: usize,
        #[arg(long, default_value_t = 1e-3)]
        mass_floor: f64,
        #[arg(long, default_value_t = 1.0)]
        eta: f64,
        #[arg(long, default_value_t = 0.05)]
        learning_rate: f64,
        /// Ladder ratio r. E0-d measured the usable range as 2..=8.
        #[arg(long, default_value_t = 4.0)]
        ladder_ratio: f64,
        #[arg(long, default_value_t = 20_260_807)]
        seed: u64,
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
        /// Give the output head its own weights. A tied head's scores are
        /// E E^T, which is symmetric and PSD and cannot express an asymmetric
        /// bigram at all.
        #[arg(long, default_value_t = false)]
        untied: bool,

        /// Admit one token per tick regardless of what is still in flight.
        /// Leaks future tokens into earlier predictions; the difference against
        /// the default serial protocol is the size of that leak.
        #[arg(long, default_value_t = false)]
        overlapped: bool,
        /// How a token's entry point is chosen.
        #[arg(long, value_enum, default_value_t = IngressArg::Content)]
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
        /// Grid sides to sweep; the scaling fit needs at least three.
        #[arg(long, value_delimiter = ',', default_value = "32,48,64,96,128")]
        sides: Vec<usize>,
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
            assert!(trials >= 2, "need at least two trials to report a standard error");
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
        Command::Run {
            tokens,
            vocab,
            order,
            fanout,
            d_head,
            slots,
            grid_side,
            long_range,
            rungs,
            context_scales,
            embed_rungs,
            mass_floor,
            eta,
            learning_rate,
            ladder_ratio,
            seed,
            bypass,
            frozen_topology,
            blind_turnover,
            untied,
            overlapped,
            ingress,
            absorb,
            out,
        } => {
            let cfg = run::Config {
                tokens,
                vocab,
                order,
            fanout,
                d_head,
                slots,
                grid_side,
                long_range,
                rungs,
                context_scales,
                embed_rungs,
                mass_floor,
                eta,
                learning_rate,
                ladder_ratio,
                seed,
                bypass,
                frozen_topology,
                blind_turnover,
                untied,
                    overlapped,
                ingress: ingress.into(),
                absorb: absorb.into(),
            };
            run::run(&cfg, &out)
        }
        Command::Topology { sides, exponents, contacts, pairs, seed, out } => {
            assert!(sides.len() >= 3, "the hops ~ N^beta fit needs at least three grid sizes");
            let cfg = topology::Config { sides, exponents, contacts, pairs, seed };
            topology::run(&cfg, &out)
        }
    }
}
