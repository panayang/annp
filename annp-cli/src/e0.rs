//! E0 — the consolidation ladder on a test bench, with no network attached.
//!
//! The ladder is a drop-in replacement for the state of any online learner, so
//! it can be judged before a single line of network code exists. Three
//! measurements, all on one `d x d` delta-rule associative memory:
//!
//! * **E0-a retention** — write one probe, bury it under fresh interfering
//!   patterns, and track its ideal-observer SNR against age. Exponential
//!   forgetting commits to one characteristic timescale; the ladder should decay
//!   as a power law instead. Both the initial SNR and the lifetime are reported,
//!   because trading the former for the latter is the whole point and hiding it
//!   would be dishonest.
//! * **E0-b Zipf capacity** — revisit a fixed pattern bank with scale-free
//!   frequencies and ask what is still recallable at the end, resolved by rank.
//!   This is the practically useful metric: do rare patterns survive.
//! * **E0-c power spectrum** — drive the ladder with white noise and measure
//!   `||U_k - U_{k+1}||^2` against `tau_k`. The null model is
//!   `Var(U_k) ~ sigma^2 tau_k / C_k^2`: charge arriving in a rung's band is a
//!   random walk over `tau_k`, spread across capacity `C_k`. For the geometric
//!   schedule those cancel, so the spectrum is *flat*. This validates the
//!   instrument DESIGN.md §1.8.3 uses to place the gradient's injection depth,
//!   before anything is allowed to depend on it.
//!
//! Every number here is produced from an explicit seed, and trials are paired:
//! all consolidation schemes see the identical stream of keys and values, so
//! differences between them are not sampling noise.

use std::fmt::Write as _;
use std::path::Path;

use annp_core::ladder::{AssocMemory, Schedule};
use annp_core::linalg::{least_squares, linear_fit, mean_std};
use annp_core::rng::Rng;
use rayon::prelude::*;

/// Separates the generator that produces probes and decoys from the one that
/// produces the interference stream, so the two can never shift each other.
const PROBE_SALT: u64 = 0xA076_1D64_78BD_642F;

/// A consolidation scheme under test.
#[derive(Clone, Copy, Debug)]
pub enum MemSpec {
    /// Single matrix, no forgetting. The palimpsest baseline.
    Plain,
    /// Single matrix with per-event decay, the incumbent this has to beat.
    Exponential {
        decay: f64,
    },
    Ladder {
        schedule: Schedule,
        rungs: usize,
    },
}

impl MemSpec {
    fn label(&self) -> String {
        match *self {
            MemSpec::Plain => "plain".to_string(),
            MemSpec::Exponential { decay } => format!("exp/tau={:.0}", 1.0 / (1.0 - decay)),
            MemSpec::Ladder {
                schedule: Schedule::Uniform { g },
                rungs,
            } => {
                format!("uniform/g={g}/m={rungs}")
            }
            MemSpec::Ladder {
                schedule: Schedule::Geometric { r, .. },
                rungs,
            } => {
                format!("geo/r={r:.0}/m={rungs}")
            }
        }
    }

    fn build(&self, d: usize) -> AssocMemory {
        match *self {
            MemSpec::Plain => AssocMemory::plain(d),
            MemSpec::Exponential { decay } => AssocMemory::single(d, decay),
            MemSpec::Ladder { schedule, rungs } => AssocMemory::ladder(d, schedule, rungs),
        }
    }

    fn is_ladder(&self) -> bool {
        matches!(self, MemSpec::Ladder { .. })
    }
}

/// Everything that determines the numbers. Recorded verbatim into the manifest.
#[derive(Clone, Debug)]
pub struct Config {
    pub d: usize,
    pub horizon: u64,
    pub warmup: u64,
    pub trials: u64,
    /// Log-spaced ages at which retention is measured. Must be dense enough to
    /// resolve a ripple of period `ln r`, which needs several points per period.
    pub age_samples: usize,
    /// Events to follow the bare impulse response for in E0-d.
    pub impulse_horizon: u64,
    pub decoys: usize,
    pub eta: f64,
    pub patterns: usize,
    pub zipf_exponent: f64,
    pub seed: u64,
    pub g_uniform: f64,
    pub g1_geometric: f64,
}

impl Config {
    /// The schemes compared, all sized from the same retention horizon.
    ///
    /// The uniform ladder is deliberately given the *same rung count* as the
    /// `r = 2` geometric one rather than the `sqrt(horizon)` rungs it would need
    /// for full coverage. That is the contrast worth showing: at equal memory
    /// cost, uniform coupling reaches `~m^2` events and geometric reaches
    /// `~r^(2m)`.
    pub fn specs(&self) -> Vec<MemSpec> {
        let horizon = self.horizon as f64;
        let geo = |r: f64| Schedule::Geometric {
            r,
            g1: self.g1_geometric,
        };
        let mut specs = vec![MemSpec::Plain];
        for tau in [30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0] {
            specs.push(MemSpec::Exponential {
                decay: 1.0 - 1.0 / tau,
            });
        }
        let matched_rungs = geo(2.0).rungs_for_horizon(horizon);
        specs.push(MemSpec::Ladder {
            schedule: Schedule::Uniform { g: self.g_uniform },
            rungs: matched_rungs,
        });
        for r in [2.0, 4.0, 8.0] {
            specs.push(MemSpec::Ladder {
                schedule: geo(r),
                rungs: geo(r).rungs_for_horizon(horizon),
            });
        }
        specs
    }
}

// ---------------------------------------------------------------------------
// Shared measurement
// ---------------------------------------------------------------------------

/// Ideal-observer SNR of one stored pattern.
///
/// `(signal - mean noise) / sd noise`, where the noise distribution comes from
/// probing the same value vector with keys that were never written. SNR ~ 1 is
/// the detectability threshold.
fn snr(mem: &mut AssocMemory, key: &[f64], value: &[f64], decoys: &[Vec<f64>]) -> f64 {
    let signal = mem.recall(key, value);
    let noise: Vec<f64> = decoys.iter().map(|k| mem.recall(k, value)).collect();
    let (mean, sd) = mean_std(&noise);
    if sd <= 0.0 { 0.0 } else { (signal - mean) / sd }
}

/// Independent, scheme-agnostic source of interfering `(key, value)` pairs.
///
/// Pairing the trials depends on every consolidation scheme seeing the exact
/// same stream, so the stream owns its own generator and is advanced exactly
/// once per event by all of them. Probe and decoy vectors are drawn from a
/// *separate* generator, so changing `--decoys` cannot shift the interference.
struct PatternStream {
    rng: Rng,
    key: Vec<f64>,
    value: Vec<f64>,
}

impl PatternStream {
    fn new(seed: u64, d: usize) -> Self {
        Self {
            rng: Rng::new(seed),
            key: vec![0.0; d],
            value: vec![0.0; d],
        }
    }

    fn advance(&mut self) {
        self.rng.fill_unit_vector(&mut self.key);
        self.rng.fill_unit_vector(&mut self.value);
    }
}

fn unit_vectors(rng: &mut Rng, count: usize, d: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|_| {
            let mut v = vec![0.0; d];
            rng.fill_unit_vector(&mut v);
            v
        })
        .collect()
}

/// Log-spaced integers in `1..=max`, deduplicated and ascending.
fn log_spaced(max: u64, count: usize) -> Vec<u64> {
    assert!(max >= 1 && count >= 2);
    let mut v: Vec<u64> = (0..count)
        .map(|i| {
            let f = i as f64 / (count - 1) as f64;
            ((max as f64).powf(f).round() as u64).clamp(1, max)
        })
        .collect();
    v.dedup();
    v
}

/// First age at which the mean SNR is no longer separable from zero at 3
/// standard errors. Everything beyond this is sampling noise and must not be
/// read as signal — the tail of these curves oscillates around zero with an
/// amplitude set purely by the trial count.
fn measurement_floor(ages: &[u64], mean: &[f64], stderr: &[f64]) -> u64 {
    for i in 0..ages.len() {
        if mean[i] < 3.0 * stderr[i].max(1e-300) {
            return ages[i];
        }
    }
    *ages.last().expect("ages is never empty")
}

/// A log-periodic component detected in the retention curve.
struct Ripple {
    amplitude: f64,
    stderr: f64,
}

/// Fits `y = a0 + a1 x + a2 x^2 + a3 cos(w x) + a4 sin(w x)` and returns the
/// amplitude of the oscillatory part.
///
/// The quadratic terms matter: these curves have real curvature in log-log
/// (a plateau while the fresh trace is still intact, then a steepening), and
/// without absorbing it into the smooth part any single arc masquerades as half
/// an oscillation. DESIGN.md §1.8.2 predicts a ripple of period `ln r` for a
/// geometric ladder, so the honest test is this amplitude at `w = 2 pi / ln r`
/// against the same amplitude at a period where nothing is predicted.
fn ripple_at(x: &[f64], y: &[f64], omega: f64) -> Option<Ripple> {
    const K: usize = 5;
    if x.len() <= K + 2 {
        return None;
    }
    let design: Vec<Vec<f64>> = x
        .iter()
        .map(|&t| vec![1.0, t, t * t, (omega * t).cos(), (omega * t).sin()])
        .collect();
    let c = least_squares(&design, y)?;
    let rss: f64 = design
        .iter()
        .zip(y)
        .map(|(row, &yi)| {
            let fit: f64 = row.iter().zip(&c).map(|(a, b)| a * b).sum();
            (yi - fit) * (yi - fit)
        })
        .sum();
    let s2 = rss / (x.len() - K) as f64;
    // Across a window spanning several periods sum(cos^2) ~ n/2, so each
    // sinusoid coefficient carries variance ~2 s2/n. Good enough for a
    // detection threshold, not for a published error bar.
    let stderr = (2.0 * s2 / x.len() as f64).sqrt();
    Some(Ripple {
        amplitude: (c[3] * c[3] + c[4] * c[4]).sqrt(),
        stderr,
    })
}

/// Age at which SNR first falls through `threshold`, interpolated in log-log.
/// `None` means it never did within the horizon.
fn lifetime(ages: &[u64], snr: &[f64], threshold: f64) -> Option<f64> {
    for i in 1..snr.len() {
        if snr[i] < threshold && snr[i - 1] >= threshold {
            let (x0, x1) = ((ages[i - 1] as f64).ln(), (ages[i] as f64).ln());
            let (y0, y1) = (snr[i - 1].max(1e-12).ln(), snr[i].max(1e-12).ln());
            if (y1 - y0).abs() < 1e-12 {
                return Some(x0.exp());
            }
            let t = (threshold.ln() - y0) / (y1 - y0);
            return Some((x0 + t * (x1 - x0)).exp());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// E0-a: retention under interference
// ---------------------------------------------------------------------------

/// One paired trial. The key/value stream depends only on `seed`, never on
/// `spec`, so every scheme is buried under the identical interference.
fn retention_trial(spec: MemSpec, cfg: &Config, ages: &[u64], seed: u64) -> Vec<f64> {
    let d = cfg.d;
    let mut probe_rng = Rng::new(seed ^ PROBE_SALT);
    let probe_key = unit_vectors(&mut probe_rng, 1, d).pop().unwrap();
    let probe_value = unit_vectors(&mut probe_rng, 1, d).pop().unwrap();
    let decoys = unit_vectors(&mut probe_rng, cfg.decoys, d);

    let mut mem = spec.build(d);
    let mut stream = PatternStream::new(seed, d);

    // Reach the steady state of the interference process before planting the
    // probe, so we measure retention rather than the transient from an empty
    // matrix.
    for _ in 0..cfg.warmup {
        stream.advance();
        mem.write(&stream.key, &stream.value, cfg.eta);
        mem.relax();
    }

    mem.write(&probe_key, &probe_value, cfg.eta);
    mem.relax();

    let mut out = Vec::with_capacity(ages.len());
    let mut next = 0usize;
    for age in 1..=cfg.horizon {
        stream.advance();
        mem.write(&stream.key, &stream.value, cfg.eta);
        mem.relax();
        while next < ages.len() && ages[next] == age {
            out.push(snr(&mut mem, &probe_key, &probe_value, &decoys));
            next += 1;
        }
    }
    assert_eq!(
        out.len(),
        ages.len(),
        "every requested age must be measured"
    );
    out
}

struct RetentionResult {
    spec: MemSpec,
    label: String,
    ages: Vec<u64>,
    snr_mean: Vec<f64>,
    snr_stderr: Vec<f64>,
}

fn run_retention(cfg: &Config, specs: &[MemSpec]) -> Vec<RetentionResult> {
    let ages = log_spaced(cfg.horizon, cfg.age_samples);
    specs
        .par_iter()
        .map(|spec| {
            let trials: Vec<Vec<f64>> = (0..cfg.trials)
                .into_par_iter()
                // Seeded by trial only: the stream is shared across specs.
                .map(|t| {
                    retention_trial(*spec, cfg, &ages, cfg.seed ^ (t.wrapping_mul(0x9E37_79B9)))
                })
                .collect();
            let (mut snr_mean, mut snr_stderr) = (Vec::new(), Vec::new());
            for i in 0..ages.len() {
                let column: Vec<f64> = trials.iter().map(|t| t[i]).collect();
                let (m, s) = mean_std(&column);
                snr_mean.push(m);
                snr_stderr.push(s / (cfg.trials as f64).sqrt());
            }
            RetentionResult {
                spec: *spec,
                label: spec.label(),
                ages: ages.clone(),
                snr_mean,
                snr_stderr,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// E0-b: capacity under Zipf revisits
// ---------------------------------------------------------------------------

/// Inverse-CDF sampler for a Zipf distribution over `n` ranks.
struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    fn new(n: usize, exponent: f64) -> Self {
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for rank in 1..=n {
            acc += (rank as f64).powf(-exponent);
            cdf.push(acc);
        }
        for c in cdf.iter_mut() {
            *c /= acc;
        }
        Self { cdf }
    }

    fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.next_f64();
        // First index with cdf >= u. `partition_point` is a binary search, so
        // this stays O(log n) for large banks.
        self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1)
    }
}

/// Returns per-pattern SNR and visit count, indexed by rank.
fn zipf_trial(spec: MemSpec, cfg: &Config, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let d = cfg.d;
    let mut pattern_rng = Rng::new(seed ^ PROBE_SALT);
    let keys = unit_vectors(&mut pattern_rng, cfg.patterns, d);
    let values = unit_vectors(&mut pattern_rng, cfg.patterns, d);
    let decoys = unit_vectors(&mut pattern_rng, cfg.decoys, d);

    let mut mem = spec.build(d);
    let mut rng = Rng::new(seed);
    let zipf = Zipf::new(cfg.patterns, cfg.zipf_exponent);

    let mut visits = vec![0.0; cfg.patterns];
    for _ in 0..cfg.horizon {
        let i = zipf.sample(&mut rng);
        visits[i] += 1.0;
        mem.write(&keys[i], &values[i], cfg.eta);
        mem.relax();
    }

    let snrs = (0..cfg.patterns)
        .map(|i| snr(&mut mem, &keys[i], &values[i], &decoys))
        .collect();
    (snrs, visits)
}

struct ZipfResult {
    label: String,
    /// Geometric rank buckets: (lo, hi] in 1-based rank.
    buckets: Vec<(usize, usize)>,
    visits_mean: Vec<f64>,
    snr_mean: Vec<f64>,
    recalled_fraction: Vec<f64>,
}

fn rank_buckets(n: usize, count: usize) -> Vec<(usize, usize)> {
    let mut edges: Vec<usize> = (0..=count)
        .map(|i| {
            let f = i as f64 / count as f64;
            ((n as f64).powf(f).round() as usize).clamp(1, n)
        })
        .collect();
    edges[0] = 0;
    edges.dedup();
    edges
        .windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|(lo, hi)| hi > lo)
        .collect()
}

fn run_zipf(cfg: &Config, specs: &[MemSpec]) -> Vec<ZipfResult> {
    let buckets = rank_buckets(cfg.patterns, 10);
    specs
        .par_iter()
        .map(|spec| {
            let trials: Vec<(Vec<f64>, Vec<f64>)> = (0..cfg.trials)
                .into_par_iter()
                .map(|t| zipf_trial(*spec, cfg, cfg.seed ^ (t.wrapping_mul(0x85EB_CA6B))))
                .collect();

            let (mut visits_mean, mut snr_mean, mut recalled) =
                (Vec::new(), Vec::new(), Vec::new());
            for &(lo, hi) in &buckets {
                let mut v = 0.0;
                let mut s = 0.0;
                let mut hit = 0.0;
                let mut n = 0.0;
                for (snrs, visits) in &trials {
                    for i in lo..hi {
                        v += visits[i];
                        s += snrs[i];
                        hit += if snrs[i] >= 1.0 { 1.0 } else { 0.0 };
                        n += 1.0;
                    }
                }
                visits_mean.push(v / n);
                snr_mean.push(s / n);
                recalled.push(hit / n);
            }
            ZipfResult {
                label: spec.label(),
                buckets: buckets.clone(),
                visits_mean,
                snr_mean,
                recalled_fraction: recalled,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// E0-c: white-noise power spectrum
// ---------------------------------------------------------------------------

struct SpectrumResult {
    label: String,
    tau: Vec<f64>,
    power: Vec<f64>,
    slope: f64,
}

/// Drives the ladder with white noise — raw random rank-one injections, no
/// delta rule — and reads off the fluctuation power per timescale band.
fn run_spectrum(cfg: &Config, specs: &[MemSpec]) -> Vec<SpectrumResult> {
    specs
        .par_iter()
        .filter(|s| s.is_ladder())
        .map(|spec| {
            let d = cfg.d;
            let mut rng = Rng::new(cfg.seed ^ 0xC2B2_AE3D);
            let mut mem = spec.build(d);
            let (mut a, mut b) = (vec![0.0; d], vec![0.0; d]);
            for _ in 0..cfg.horizon {
                rng.fill_unit_vector(&mut a);
                rng.fill_unit_vector(&mut b);
                mem.inject(&a, &b, cfg.eta);
                mem.relax();
            }

            let l = mem.ladder_state().expect("filtered to ladders");
            let n = (d * d) as f64;
            let mut tau = Vec::new();
            let mut power = Vec::new();
            for k in 0..l.num_rungs() - 1 {
                tau.push(l.relaxation_time(k));
                power.push(l.rung(k).dist_sq(l.rung(k + 1)) / n);
            }
            let logs: (Vec<f64>, Vec<f64>) = tau
                .iter()
                .zip(&power)
                .map(|(t, p)| (t.ln(), p.max(1e-300).ln()))
                .unzip();
            let slope = if logs.0.len() >= 2 {
                linear_fit(&logs.0, &logs.1).0
            } else {
                f64::NAN
            };
            SpectrumResult {
                label: spec.label(),
                tau,
                power,
                slope,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// E0-d: bare impulse response
// ---------------------------------------------------------------------------

/// The Green's function of the ladder itself, with no memory task attached.
///
/// E0-a measures the ripple through a composite observable — the SNR of a
/// delta-rule memory under a continuously refreshed interference process — and
/// that process can smear a discrete scale invariance that is nonetheless
/// present in the ladder. §1.8.2 predicts the ripple for the ladder's *response*,
/// so this measures exactly that: inject one unit impulse into rung 1 and watch
/// it decay.
///
/// The dynamics are linear and elementwise decoupled, so a 1x1 matrix carries
/// the whole answer, and there is no sampling noise at all — the only
/// uncertainty is fit residual. Anything not visible here is not in the ladder.
struct ImpulseResult {
    label: String,
    times: Vec<u64>,
    value: Vec<f64>,
    /// Window actually fitted: the decade range where the chain is still
    /// relaxing rather than equilibrated.
    window: (u64, u64),
    slope: f64,
    ripple_pred: Option<Ripple>,
    ripple_decoy: Option<Ripple>,
}

fn run_impulse(cfg: &Config) -> Vec<ImpulseResult> {
    let horizon = cfg.impulse_horizon;
    // Size each chain to cover ten times the horizon so the measurement never
    // runs into the plateau where the finite chain has equilibrated.
    // Ratios well past anything we would deploy: if the discrete scale
    // invariance leaves a log-periodic signature at all, its amplitude must
    // grow with the coarseness of the ladder, so a null that persists out to
    // r = 32 is a null everywhere.
    let mut schedules: Vec<Schedule> = vec![Schedule::Uniform { g: cfg.g_uniform }];
    for r in [2.0, 4.0, 8.0, 16.0, 32.0] {
        schedules.push(Schedule::Geometric {
            r,
            g1: cfg.g1_geometric,
        });
    }

    schedules
        .par_iter()
        .map(|&schedule| {
            let rungs = schedule.rungs_for_horizon(10.0 * horizon as f64);
            let spec = MemSpec::Ladder { schedule, rungs };
            let mut l = annp_core::ladder::Ladder::new(schedule, rungs, 1, 1);
            l.inject(&[1.0], &[1.0], 1.0);

            let times = log_spaced(horizon, cfg.age_samples);
            let mut value = Vec::with_capacity(times.len());
            let mut next = 0usize;
            for t in 1..=horizon {
                l.relax();
                while next < times.len() && times[next] == t {
                    value.push(l.rung(0).get(0, 0));
                    next += 1;
                }
            }

            // Fit where the response is a decaying power law: past the first
            // rung's own relaxation, before the chain equilibrates.
            let lo = (l.relaxation_time(0) * 10.0).ceil() as u64;
            let hi = horizon;
            let (x, y): (Vec<f64>, Vec<f64>) = times
                .iter()
                .zip(&value)
                .filter(|(t, v)| **t >= lo && **t <= hi && **v > 0.0)
                .map(|(t, v)| ((*t as f64).ln(), v.ln()))
                .unzip();
            let slope = if x.len() >= 5 {
                linear_fit(&x, &y).0
            } else {
                f64::NAN
            };
            let (pred, decoy) = match schedule {
                Schedule::Geometric { r, .. } => (
                    ripple_at(&x, &y, std::f64::consts::TAU / r.ln()),
                    ripple_at(
                        &x,
                        &y,
                        std::f64::consts::TAU / (r.ln() * std::f64::consts::SQRT_2),
                    ),
                ),
                Schedule::Uniform { .. } => (None, None),
            };
            ImpulseResult {
                label: spec.label(),
                times,
                value,
                window: (lo, hi),
                slope,
                ripple_pred: pred,
                ripple_decoy: decoy,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Driver and output
// ---------------------------------------------------------------------------

fn write_csv(dir: &Path, name: &str, body: String) -> std::io::Result<()> {
    let path = dir.join(name);
    std::fs::write(&path, body)?;
    println!("  wrote {}", path.display());
    Ok(())
}

fn git_revision() -> String {
    std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=12"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn run(cfg: &Config, out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let specs = cfg.specs();

    println!("E0 — consolidation ladder bench");
    println!(
        "  d={} horizon={} warmup={} trials={}",
        cfg.d, cfg.horizon, cfg.warmup, cfg.trials
    );
    println!("  seed={} schemes={}", cfg.seed, specs.len());
    println!();

    let impulse = run_impulse(cfg);
    let retention = run_retention(cfg, &specs);
    let zipf = run_zipf(cfg, &specs);
    let spectrum = run_spectrum(cfg, &specs);

    // --- E0-a summary -----------------------------------------------------
    println!("E0-a  retention under interference");
    println!(
        "  {:<20} {:>10} {:>12} {:>8} {:>9} {:>7}",
        "scheme", "SNR(age=1)", "SNR>1 until", "floor", "slope", "pts"
    );
    for r in &retention {
        let floor = measurement_floor(&r.ages, &r.snr_mean, &r.snr_stderr);
        // Fit only where the signal is real: past the initial plateau, above a
        // usable SNR, and below the age where sampling noise takes over.
        let pts: Vec<(f64, f64)> = r
            .ages
            .iter()
            .zip(&r.snr_mean)
            .filter(|(a, s)| **a >= 10 && **a <= floor && **s > 0.3)
            .map(|(a, s)| ((*a as f64).ln(), s.ln()))
            .collect();
        let n = pts.len();
        let slope = if n >= 5 {
            let (x, y): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
            linear_fit(&x, &y).0
        } else {
            f64::NAN
        };
        let life = lifetime(&r.ages, &r.snr_mean, 1.0);
        println!(
            "  {:<20} {:>10.2} {:>12} {:>8} {:>9.3} {:>7}",
            r.label,
            r.snr_mean[0],
            life.map(|l| format!("{l:.0}"))
                .unwrap_or_else(|| format!(">{}", cfg.horizon)),
            floor,
            slope,
            n,
        );
    }
    println!(
        "  floor: first age where the mean is within 3 stderr of zero. Nothing past it is signal."
    );
    println!("  slope: local log-log slope inside [10, floor]. A power-law ladder sits near -0.5.");
    println!();

    // --- E0-a, log-periodic ripple ---------------------------------------
    // DESIGN.md §1.8.2: a geometric ladder is only discretely scale invariant,
    // so its t^-1/2 decay should carry a ripple of period ln r. The decoy
    // column fits the same model at an unpredicted period; without a clear
    // separation between the two there is no detection, only curvature.
    println!("E0-a  log-periodic ripple (predicted period ln r)");
    println!(
        "  {:<20} {:>10} {:>18} {:>18}",
        "scheme", "pts/period", "amp @ ln r", "amp @ decoy"
    );
    for r in &retention {
        let MemSpec::Ladder {
            schedule: Schedule::Geometric { r: ratio, .. },
            ..
        } = r.spec
        else {
            continue;
        };
        let floor = measurement_floor(&r.ages, &r.snr_mean, &r.snr_stderr);
        let (x, y): (Vec<f64>, Vec<f64>) = r
            .ages
            .iter()
            .zip(&r.snr_mean)
            .filter(|(a, s)| **a >= 10 && **a <= floor && **s > 0.3)
            .map(|(a, s)| ((*a as f64).ln(), s.ln()))
            .unzip();
        if x.len() < 8 {
            println!("  {:<20} {:>10}", r.label, "too few pts");
            continue;
        }
        let span = x.last().unwrap() - x.first().unwrap();
        let per_period = x.len() as f64 * ratio.ln() / span;
        // sqrt(2) is irrational, so the decoy period cannot alias onto the
        // predicted one or any of its harmonics.
        let fits = [
            ripple_at(&x, &y, std::f64::consts::TAU / ratio.ln()),
            ripple_at(
                &x,
                &y,
                std::f64::consts::TAU / (ratio.ln() * std::f64::consts::SQRT_2),
            ),
        ];
        let show = |f: &Option<Ripple>| {
            f.as_ref()
                .map(|r| format!("{:.4} +/- {:.4}", r.amplitude, r.stderr))
                .unwrap_or_else(|| "n/a".to_string())
        };
        println!(
            "  {:<20} {per_period:>10.1} {:>18} {:>18}",
            r.label,
            show(&fits[0]),
            show(&fits[1])
        );
    }
    println!("  a ripple needs >=4 pts/period to be resolvable at all, and must beat the decoy.");
    println!();

    // --- E0-d summary -----------------------------------------------------
    // No sampling noise here at all: the chain is linear, deterministic, and
    // measured on a 1x1 matrix. If the ripple is not in this table it is not in
    // the ladder, and E0-a's null result is not an artefact of the memory task.
    println!("E0-d  bare impulse response (no memory task, no sampling noise)");
    println!(
        "  {:<20} {:>9} {:>18} {:>18} {:>16}",
        "schedule", "slope", "amp @ ln r", "amp @ decoy", "window"
    );
    for r in &impulse {
        let show = |f: &Option<Ripple>| {
            f.as_ref()
                .map(|r| format!("{:.5} +/- {:.5}", r.amplitude, r.stderr))
                .unwrap_or_else(|| "n/a".to_string())
        };
        println!(
            "  {:<20} {:>9.4} {:>18} {:>18} {:>16}",
            r.label,
            r.slope,
            show(&r.ripple_pred),
            show(&r.ripple_decoy),
            format!("{}..{}", r.window.0, r.window.1)
        );
    }
    println!();

    // --- E0-c summary -----------------------------------------------------
    // Null model, corrected. The variance a white-noise drive leaves in rung k
    // is ~ sigma^2 * tau_k / C_k^2: charge arriving in that rung's band is a
    // random walk over tau_k, spread over capacity C_k. The geometric schedule
    // sets tau_k = r^2k/g1 against C_k = r^k, so the ratio is constant and the
    // spectrum is FLAT — the ladder allocates equal fluctuation power to every
    // timescale decade. (An earlier -1 prediction ignored the C_k^2 term.)
    // A flat baseline is the better instrument anyway: signal is a bump, not a
    // change of slope.
    println!("E0-c  white-noise power spectrum (null model: flat for geometric)");
    for s in &spectrum {
        let levels: Vec<String> = s.power.iter().map(|p| format!("{p:.1e}")).collect();
        println!(
            "  {:<20} slope = {:>7.3}   rungs: {}",
            s.label,
            s.slope,
            levels.join(" ")
        );
    }
    println!();

    // --- E0-b summary -----------------------------------------------------
    // The practical question: after a scale-free stream of revisits, how deep
    // into the frequency tail is anything still recallable?
    println!(
        "E0-b  capacity under Zipf revisits ({} patterns)",
        cfg.patterns
    );
    println!(
        "  {:<20} {:>14} {:>16}",
        "scheme", "usable ranks", "recalled overall"
    );
    for z in &zipf {
        // Highest rank whose bucket still averages SNR >= 1.
        let usable = z
            .buckets
            .iter()
            .zip(&z.snr_mean)
            .filter(|(_, s)| **s >= 1.0)
            .map(|((_, hi), _)| *hi)
            .max()
            .unwrap_or(0);
        let overall: f64 = z
            .recalled_fraction
            .iter()
            .zip(&z.buckets)
            .map(|(f, (lo, hi))| f * (hi - lo) as f64)
            .sum::<f64>()
            / cfg.patterns as f64;
        println!("  {:<20} {usable:>14} {:>15.1}%", z.label, 100.0 * overall);
    }
    println!();

    // --- files ------------------------------------------------------------
    let mut csv = String::from("scheme,age,snr_mean,snr_stderr\n");
    for r in &retention {
        for i in 0..r.ages.len() {
            let _ = writeln!(
                csv,
                "{},{},{:.6e},{:.6e}",
                r.label, r.ages[i], r.snr_mean[i], r.snr_stderr[i]
            );
        }
    }
    write_csv(out_dir, "e0a_retention.csv", csv)?;

    let mut csv = String::from("scheme,rank_lo,rank_hi,visits_mean,snr_mean,recalled_fraction\n");
    for r in &zipf {
        for i in 0..r.buckets.len() {
            let (lo, hi) = r.buckets[i];
            let _ = writeln!(
                csv,
                "{},{},{},{:.6e},{:.6e},{:.6e}",
                r.label,
                lo + 1,
                hi,
                r.visits_mean[i],
                r.snr_mean[i],
                r.recalled_fraction[i]
            );
        }
    }
    write_csv(out_dir, "e0b_zipf.csv", csv)?;

    let mut csv = String::from("schedule,t,value\n");
    for r in &impulse {
        for i in 0..r.times.len() {
            let _ = writeln!(csv, "{},{},{:.12e}", r.label, r.times[i], r.value[i]);
        }
    }
    write_csv(out_dir, "e0d_impulse.csv", csv)?;

    let mut csv = String::from("scheme,rung,tau,power\n");
    for s in &spectrum {
        for k in 0..s.tau.len() {
            let _ = writeln!(
                csv,
                "{},{},{:.6e},{:.6e}",
                s.label,
                k + 1,
                s.tau[k],
                s.power[k]
            );
        }
    }
    write_csv(out_dir, "e0c_spectrum.csv", csv)?;

    let manifest = format!(
        concat!(
            "{{\n",
            "  \"experiment\": \"E0\",\n",
            "  \"git\": \"{git}\",\n",
            "  \"d\": {d},\n",
            "  \"horizon\": {horizon},\n",
            "  \"warmup\": {warmup},\n",
            "  \"trials\": {trials},\n",
            "  \"age_samples\": {age_samples},\n",
            "  \"impulse_horizon\": {impulse_horizon},\n",
            "  \"decoys\": {decoys},\n",
            "  \"eta\": {eta},\n",
            "  \"patterns\": {patterns},\n",
            "  \"zipf_exponent\": {zipf},\n",
            "  \"g_uniform\": {gu},\n",
            "  \"g1_geometric\": {g1},\n",
            "  \"seed\": {seed},\n",
            "  \"schemes\": [{schemes}]\n",
            "}}\n"
        ),
        git = git_revision(),
        d = cfg.d,
        horizon = cfg.horizon,
        warmup = cfg.warmup,
        trials = cfg.trials,
        age_samples = cfg.age_samples,
        impulse_horizon = cfg.impulse_horizon,
        decoys = cfg.decoys,
        eta = cfg.eta,
        patterns = cfg.patterns,
        zipf = cfg.zipf_exponent,
        gu = cfg.g_uniform,
        g1 = cfg.g1_geometric,
        seed = cfg.seed,
        schemes = specs
            .iter()
            .map(|s| format!("\"{}\"", s.label()))
            .collect::<Vec<_>>()
            .join(", "),
    );
    write_csv(out_dir, "e0_manifest.json", manifest)?;
    crate::write_manifest(out_dir, "E0", cfg);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            d: 16,
            horizon: 400,
            warmup: 100,
            trials: 2,
            age_samples: 16,
            impulse_horizon: 2_000,
            decoys: 32,
            eta: 1.0,
            patterns: 64,
            zipf_exponent: 1.0,
            seed: 1,
            g_uniform: 0.25,
            g1_geometric: 0.5,
        }
    }

    #[test]
    fn log_spaced_is_ascending_and_bounded() {
        let v = log_spaced(30_000, 48);
        assert_eq!(v[0], 1);
        assert_eq!(*v.last().unwrap(), 30_000);
        assert!(v.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn rank_buckets_tile_the_range_exactly() {
        let b = rank_buckets(2000, 10);
        assert_eq!(b[0].0, 0);
        assert_eq!(b.last().unwrap().1, 2000);
        assert!(
            b.windows(2).all(|w| w[0].1 == w[1].0),
            "buckets must be contiguous"
        );
    }

    #[test]
    fn zipf_ranks_are_ordered_by_frequency() {
        let z = Zipf::new(100, 1.0);
        let mut rng = Rng::new(5);
        let mut counts = vec![0u32; 100];
        for _ in 0..200_000 {
            counts[z.sample(&mut rng)] += 1;
        }
        assert!(counts[0] > counts[1], "rank 1 must be the most frequent");
        assert!(
            counts[0] > 4 * counts[9],
            "1/k should make rank 1 ~10x rank 10"
        );
        assert!(counts.iter().sum::<u32>() == 200_000);
        assert!(counts[99] > 0, "the tail must still be reachable");
    }

    #[test]
    fn measurement_floor_finds_where_the_signal_dies() {
        let ages = [1u64, 10, 100, 1000, 10000];
        let mean = [5.0, 3.0, 1.0, 0.05, -0.2];
        let stderr = [0.1, 0.1, 0.1, 0.1, 0.1];
        // 0.05 < 3 * 0.1 is the first age that fails the separability test.
        assert_eq!(measurement_floor(&ages, &mean, &stderr), 1000);
        // A curve that never dies reports the last age rather than panicking.
        assert_eq!(measurement_floor(&ages, &[5.0; 5], &stderr), 10000);
    }

    #[test]
    fn lifetime_interpolates_a_known_crossing() {
        // SNR halves each doubling of age: SNR = 4 at 1, 2 at 2, 1 at 4.
        let ages = [1u64, 2, 4, 8];
        let snr = [4.0, 2.0, 1.0, 0.5];
        let l = lifetime(&ages, &snr, 1.0).unwrap();
        // Crossing happens exactly at the sample age 8 boundary: the first
        // point strictly below 1.0 is at age 8, interpolating back to 4.
        assert!((l - 4.0).abs() < 1e-9, "got {l}");
    }

    #[test]
    fn interference_stream_depends_only_on_its_seed() {
        // The whole comparison is paired, which is only meaningful if the
        // stream is byte-identical across schemes. It is spec-agnostic by
        // construction; this pins that it is also seed-deterministic and that
        // different seeds really do diverge.
        let (mut a, mut b, mut c) = (
            PatternStream::new(9, 8),
            PatternStream::new(9, 8),
            PatternStream::new(10, 8),
        );
        for _ in 0..64 {
            a.advance();
            b.advance();
            c.advance();
            assert_eq!(a.key, b.key);
            assert_eq!(a.value, b.value);
        }
        assert_ne!(a.key, c.key);
    }

    #[test]
    fn the_ladder_holds_less_interference_in_the_matrix_it_reads() {
        // Not an assumption, a measured property worth pinning: rung 1 sheds
        // old writes into the deeper rungs, so at equal age it carries a lower
        // noise floor than a plain matrix and the same probe reads out at
        // higher SNR. This is why the ladder's initial SNR is not simply
        // traded away against its lifetime.
        let c = cfg();
        let ages = log_spaced(c.horizon, 8);
        let plain = retention_trial(MemSpec::Plain, &c, &ages, 42);
        let ladder = retention_trial(
            MemSpec::Ladder {
                schedule: Schedule::Geometric { r: 4.0, g1: 0.5 },
                rungs: 4,
            },
            &c,
            &ages,
            42,
        );
        assert!(
            ladder[0] > plain[0],
            "ladder {} vs plain {}",
            ladder[0],
            plain[0]
        );
    }

    #[test]
    fn a_freshly_written_probe_is_strongly_detectable() {
        let c = cfg();
        let ages = log_spaced(c.horizon, 8);
        let snr = retention_trial(MemSpec::Plain, &c, &ages, 7);
        // Detectability threshold is SNR = 1. The margin above it grows with d;
        // these tests use d = 16, where a single interfering write already
        // overlaps the probe key by ~1/sqrt(d) = 0.25.
        assert!(
            snr[0] > 2.0,
            "one-shot storage should clear threshold: {}",
            snr[0]
        );
        assert!(
            snr[0] > *snr.last().unwrap(),
            "SNR must decay with interference"
        );
    }
}
