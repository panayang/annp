//! ANNP command line.
//!
//! Every subcommand is an experiment. Each writes CSVs plus a `manifest.json`
//! recording the git revision and every parameter that went into the numbers,
//! so any result in the paper can be regenerated from one line.

mod e0;
mod run;
mod topology;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
        #[arg(long, default_value_t = 64)]
        vocab: usize,
        /// Successors per state. Lower means a more predictable source.
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
        /// Rungs behind the tied table; 1 means a plain matrix.
        #[arg(long, default_value_t = 3)]
        embed_rungs: usize,
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        #[arg(long, default_value_t = 1e-3)]
        mass_floor: f64,
        #[arg(long, default_value_t = 1.0)]
        eta: f64,
        #[arg(long, default_value_t = 0.05)]
        homeostasis: f64,
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
        /// Admit one token per tick regardless of what is still in flight.
        /// Leaks future tokens into earlier predictions; the difference against
        /// the default serial protocol is the size of that leak.
        #[arg(long, default_value_t = false)]
        overlapped: bool,
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
            fanout,
            d_head,
            slots,
            grid_side,
            long_range,
            rungs,
            embed_rungs,
            top_k,
            mass_floor,
            eta,
            homeostasis,
            learning_rate,
            ladder_ratio,
            seed,
            bypass,
            overlapped,
            out,
        } => {
            let cfg = run::Config {
                tokens,
                vocab,
                fanout,
                d_head,
                slots,
                grid_side,
                long_range,
                rungs,
                embed_rungs,
                top_k,
                mass_floor,
                eta,
                homeostasis,
                learning_rate,
                ladder_ratio,
                seed,
                bypass,
                overlapped,
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
