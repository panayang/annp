//! Dense `f64` matrices and vector helpers.
//!
//! `f64`, not `f32`, and this is not negotiable for the consolidation ladder:
//! the deepest rung of a geometric ladder advances by a factor of
//! `g1 * r^-(2m-3)` per step, which for `r = 4, m = 8` is `1.5e-8` — below the
//! `f32` epsilon of `1.2e-7`. In `f32` the deep rungs would silently stop
//! integrating and the ladder would look like it simply does not work.
//! See `ladder::tests::deep_rungs_actually_integrate`.

/// Row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn fill_zero(&mut self) {
        self.data.fill(0.0);
    }

    /// `out <- self * x`.
    pub fn mul_vec(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), self.cols, "mul_vec: input length must equal cols");
        assert_eq!(out.len(), self.rows, "mul_vec: output length must equal rows");
        for (i, o) in out.iter_mut().enumerate() {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }

    /// `self += s * a * b^T`.
    ///
    /// Every entry `(i, j)` is touched: this is a full rank-one update, not a
    /// blockwise or diagonal one. `outer_update_is_dense` guards that.
    pub fn add_outer(&mut self, a: &[f64], b: &[f64], s: f64) {
        assert_eq!(a.len(), self.rows, "add_outer: a must have length rows");
        assert_eq!(b.len(), self.cols, "add_outer: b must have length cols");
        for (i, &ai) in a.iter().enumerate() {
            let sai = s * ai;
            let row = &mut self.data[i * self.cols..(i + 1) * self.cols];
            for (dst, &bj) in row.iter_mut().zip(b) {
                *dst += sai * bj;
            }
        }
    }

    /// `self += s * other`.
    pub fn axpy(&mut self, s: f64, other: &Mat) {
        assert_eq!(self.shape(), other.shape(), "axpy: shape mismatch");
        for (dst, &src) in self.data.iter_mut().zip(&other.data) {
            *dst += s * src;
        }
    }

    /// `self *= s`.
    pub fn scale(&mut self, s: f64) {
        for dst in self.data.iter_mut() {
            *dst *= s;
        }
    }

    pub fn frob_sq(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }

    /// `||self - other||_F^2`.
    pub fn dist_sq(&self, other: &Mat) -> f64 {
        assert_eq!(self.shape(), other.shape(), "dist_sq: shape mismatch");
        self.data.iter().zip(&other.data).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `out <- a - b`.
#[inline]
pub fn sub_into(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len(), "sub_into: length mismatch");
    assert_eq!(a.len(), out.len(), "sub_into: output length mismatch");
    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
        *o = x - y;
    }
}

/// Sample mean and standard deviation (Bessel-corrected).
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    assert!(xs.len() >= 2, "mean_std needs at least two samples");
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

/// Least-squares slope and intercept of `y = slope * x + intercept`.
pub fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "linear_fit: length mismatch");
    assert!(x.len() >= 2, "linear_fit needs at least two points");
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    assert!(sxx > 0.0, "linear_fit: x values are degenerate");
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// Solves `min_c ||A c - y||^2` via the normal equations with partial pivoting.
///
/// Sized for the diagnostics in this crate: a handful of basis columns over a
/// few dozen observations, where conditioning is not a concern. Returns `None`
/// if the design is rank deficient rather than emitting a garbage fit.
pub fn least_squares(design: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(design.len(), y.len(), "least_squares: row count mismatch");
    let k = design.first()?.len();
    assert!(design.iter().all(|r| r.len() == k), "least_squares: ragged design");
    if design.len() < k {
        return None;
    }

    // Augmented normal equations: [A^T A | A^T y].
    let mut m = vec![vec![0.0; k + 1]; k];
    for (row, &yi) in design.iter().zip(y) {
        for i in 0..k {
            for j in 0..k {
                m[i][j] += row[i] * row[j];
            }
            m[i][k] += row[i] * yi;
        }
    }

    for col in 0..k {
        let pivot = (col..k).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        for v in m[col].iter_mut() {
            *v /= d;
        }
        let pivot_row: Vec<f64> = m[col][col..=k].to_vec();
        for (row, r) in m.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let f = r[col];
            if f != 0.0 {
                for (dst, &p) in r[col..=k].iter_mut().zip(&pivot_row) {
                    *dst -= f * p;
                }
            }
        }
    }
    Some((0..k).map(|i| m[i][k]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_rows(rows: &[&[f64]]) -> Mat {
        let mut m = Mat::zeros(rows.len(), rows[0].len());
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                m.set(i, j, v);
            }
        }
        m
    }

    #[test]
    fn mul_vec_matches_hand_computation() {
        // Deliberately non-square so a transposed indexing bug cannot pass.
        let m = from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let mut out = vec![0.0; 2];
        m.mul_vec(&[1.0, 0.5, -2.0], &mut out);
        assert_eq!(out, vec![1.0 + 1.0 - 6.0, 4.0 + 2.5 - 12.0]);
    }

    #[test]
    fn outer_update_is_dense() {
        // The bug class this guards: a rank-one update written so that it only
        // fills a diagonal or a block, turning an all-to-all coupling into a
        // partitioned one. Every single entry must move.
        let (rows, cols) = (7, 5);
        let mut m = Mat::zeros(rows, cols);
        let a: Vec<f64> = (0..rows).map(|i| 1.0 + i as f64).collect();
        let b: Vec<f64> = (0..cols).map(|j| 2.0 + j as f64).collect();
        m.add_outer(&a, &b, 0.5);
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let want = 0.5 * ai * bj;
                assert!(want != 0.0);
                assert_eq!(m.get(i, j), want, "entry ({i},{j}) wrong");
            }
        }
    }

    #[test]
    fn outer_then_apply_is_a_projection() {
        // (s * a b^T) x == s * a * (b . x), independent of dimensions.
        let mut rng = crate::rng::Rng::new(5);
        let (rows, cols) = (9, 6);
        let mut m = Mat::zeros(rows, cols);
        let mut a = vec![0.0; rows];
        let mut b = vec![0.0; cols];
        let mut x = vec![0.0; cols];
        rng.fill_unit_vector(&mut a);
        rng.fill_unit_vector(&mut b);
        rng.fill_unit_vector(&mut x);
        m.add_outer(&a, &b, 1.7);
        let mut out = vec![0.0; rows];
        m.mul_vec(&x, &mut out);
        let bx = dot(&b, &x);
        for i in 0..rows {
            assert!((out[i] - 1.7 * a[i] * bx).abs() < 1e-12);
        }
    }

    #[test]
    fn delta_rule_stores_exactly_in_one_shot() {
        // With a unit-norm key and eta = 1, W += (v - Wk)k^T must give Wk == v.
        // This pins the delta rule's orientation: the key indexes columns.
        let mut rng = crate::rng::Rng::new(11);
        let d = 16;
        let mut w = Mat::zeros(d, d);
        let (mut k, mut v) = (vec![0.0; d], vec![0.0; d]);
        rng.fill_unit_vector(&mut k);
        rng.fill_unit_vector(&mut v);

        let mut pred = vec![0.0; d];
        w.mul_vec(&k, &mut pred);
        let mut resid = vec![0.0; d];
        sub_into(&v, &pred, &mut resid);
        w.add_outer(&resid, &k, 1.0);

        w.mul_vec(&k, &mut pred);
        for i in 0..d {
            assert!((pred[i] - v[i]).abs() < 1e-12, "coordinate {i} not stored");
        }
    }

    #[test]
    fn linear_fit_recovers_a_known_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
        let y: Vec<f64> = x.iter().map(|v| -1.5 * v + 4.0).collect();
        let (slope, intercept) = linear_fit(&x, &y);
        assert!((slope + 1.5).abs() < 1e-12);
        assert!((intercept - 4.0).abs() < 1e-12);
    }

    #[test]
    fn least_squares_recovers_exact_coefficients() {
        // y = 2 - 3x + 0.5x^2 + 4cos(x) - 1.5sin(x), sampled off the grid so no
        // basis column is accidentally orthogonalised into agreement.
        let want = [2.0, -3.0, 0.5, 4.0, -1.5];
        let xs: Vec<f64> = (0..40).map(|i| 0.13 * i as f64).collect();
        let design: Vec<Vec<f64>> =
            xs.iter().map(|&x| vec![1.0, x, x * x, x.cos(), x.sin()]).collect();
        let y: Vec<f64> =
            design.iter().map(|r| r.iter().zip(want).map(|(a, b)| a * b).sum()).collect();
        let got = least_squares(&design, &y).expect("well-conditioned design");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-8, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn least_squares_rejects_a_rank_deficient_design() {
        // Second column is a multiple of the first.
        let design: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 2.0 * i as f64]).collect();
        assert!(least_squares(&design, &[1.0; 10]).is_none());
        // Fewer observations than unknowns.
        assert!(least_squares(&[vec![1.0, 2.0]], &[1.0]).is_none());
    }

    #[test]
    fn mean_std_matches_hand_computation() {
        let (m, s) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((m - 5.0).abs() < 1e-12);
        // Bessel-corrected variance is 32/7.
        assert!((s - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
    }
}
