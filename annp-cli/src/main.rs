//! ANNP command line.
//!
//! Every subcommand is an experiment. Each writes CSVs plus a `manifest.json`
//! recording the git revision and every parameter that went into the numbers,
//! so any result in the paper can be regenerated from one line.

mod e0;

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
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::E0 {
            d,
            horizon,
            warmup,
            trials,
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
    }
}
