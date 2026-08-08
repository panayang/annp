//! Deterministic RNG, vendored on purpose.
//!
//! Experiments must reproduce bit-for-bit years from now, including from a
//! `Cargo.lock` that someone regenerated. Depending on an external RNG crate
//! makes reproducibility hostage to that crate's version history, so we carry
//! xoshiro256++ (seeded by SplitMix64) here. Both are fixed, published
//! algorithms; the constants below are their specified values.

/// xoshiro256++ state.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    /// Second variate from the last Box-Muller pair, if one is pending.
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seeds the generator. Every seed produces a distinct, well-mixed stream,
    /// including `0` (SplitMix64 has no zero-state defect).
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)`, using the top 53 bits so every output is an exactly
    /// representable multiple of 2^-53.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on `[0, n)`. Rejection-sampled, so it is exactly uniform rather
    /// than modulo-biased.
    #[inline]
    pub fn next_below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "next_below requires a positive bound");
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }

    /// Standard normal via Box-Muller. Generates two variates at a time and
    /// caches the second.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // next_f64 can return exactly 0.0, whose log is -inf; shift off zero.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fills `out` with a uniformly random direction on the unit sphere.
    pub fn fill_unit_vector(&mut self, out: &mut [f64]) {
        loop {
            for x in out.iter_mut() {
                *x = self.next_normal();
            }
            let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Probability ~0; retry rather than divide by a denormal.
            if norm > 1e-12 {
                for x in out.iter_mut() {
                    *x /= norm;
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let a: Vec<u64> = (0..64).map(|_| Rng::new(12345).next_u64()).collect();
        let mut b = Rng::new(12345);
        assert!(a.iter().all(|&x| x == a[0]));
        assert_eq!(a[0], b.next_u64());
    }

    #[test]
    fn distinct_seeds_diverge_immediately() {
        // A weak seeder would leave nearby seeds correlated in the first draws.
        let first: Vec<u64> = (0..16).map(|s| Rng::new(s).next_u64()).collect();
        for i in 0..first.len() {
            for j in (i + 1)..first.len() {
                assert_ne!(first[i], first[j]);
            }
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..100_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn next_below_is_unbiased_enough() {
        let mut rng = Rng::new(99);
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[rng.next_below(7) as usize] += 1;
        }
        // Expect 10_000 each; 5-sigma of a binomial(70k, 1/7) is ~490.
        for c in counts {
            assert!(
                (9_000..11_000).contains(&c),
                "bucket counts skewed: {counts:?}"
            );
        }
    }

    #[test]
    fn normal_moments() {
        let mut rng = Rng::new(2024);
        let n = 400_000;
        let (mut sum, mut sum_sq) = (0.0, 0.0);
        for _ in 0..n {
            let z = rng.next_normal();
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn unit_vectors_are_unit_and_isotropic() {
        let mut rng = Rng::new(31337);
        let d = 32;
        let mut v = vec![0.0; d];
        let mut mean = vec![0.0; d];
        let trials = 20_000;
        for _ in 0..trials {
            rng.fill_unit_vector(&mut v);
            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
            assert!((norm_sq - 1.0).abs() < 1e-12, "not unit: {norm_sq}");
            for (m, x) in mean.iter_mut().zip(&v) {
                *m += x / trials as f64;
            }
        }
        // Isotropy: each coordinate averages to 0 with sd ~ 1/sqrt(d*trials).
        let tol = 8.0 / ((d * trials) as f64).sqrt();
        assert!(mean.iter().all(|m| m.abs() < tol), "anisotropic: {mean:?}");
    }
}
