# Multi-timescale consolidation as a continual-learning mechanism: a measured negative, with one narrow positive

Working notes, 2026-08-16. Synthetic sources only. Every number here comes from
a run whose configuration and self-checks are printed alongside it.

---

## 1. What is under test

A **node** is the only computational unit: one weight matrix, a nonlinear local
write, and a Benna–Fusi consolidation ladder underneath it.

$$p = \mathrm{softmax}(Wk), \qquad W \mathrel{+}= \eta\,(\mathbf{1}_y - p)\,k^{\top}$$

The write is simultaneously a nonlinear local rule and the exact cross-entropy
gradient of a linear softmax layer, so "no backpropagation" costs nothing here —
there is no hidden layer for a gradient to traverse. $W$ sits on a ladder with
$m$ rungs; rung 1 is live and read, the rest are state.

Nodes are arranged either **flat** (one router, $n$ siblings) or as a **growing
tree** (each node up to $n$ children, instantiated on demand, every path bounded
at depth $D$). Routing is by prototype mismatch, computed from the input alone —
routing on prediction error would require the target, and the protocol is
predict, charge, then write. Growth never stops; the shape is bounded instead,
so the node count is pinned at $(n^{D+1}-1)/(n-1)$ without a second parameter.

**Controls.** Online EWC with a proximal penalty, task-free, sharing the same
write. Earlier work in this project compared against feedforward networks; those
are memoryless learners, and judging a retention mechanism against them on a
stationary stream is their home ground rather than the question.

---

## 2. Source

Order-1 chains with a **Zipf marginal** and a state-dependent tilt:

$$P(j \mid c) \;\propto\; \pi(j)\,e^{\pm\varepsilon}, \qquad \pi(j) \propto j^{-s}$$

Non-stationarity comes from cycling through domains whose alphabet windows
overlap: the overlap gives interference, so there is something to forget, and
the non-overlap gives a content signature, so routing has something to separate.

Every run verifies the realised shape rather than assuming it. At $s=1$ the
measured exponent is $-1.005$ with head and tail sub-fits 0.18 apart, and the
tilt buys about 0.44 bits of conditional information above the estimator's floor.

Two calibration facts worth carrying:

- **Mutual information estimated from counts is positively biased.** At $256^2$
  cells over 200k samples the floor is 0.211 bits, and at $\varepsilon = 0$ the
  true value is zero. MI is only readable relative to $\varepsilon = 0$.
- **Mixing domains flattens the combined marginal.** Eight overlapping windows
  measure $-0.77$ where a single domain measures $-1.005$. Per-domain and
  combined exponents are different quantities and both must be quoted.

---

## 3. The tree divides labour, and depth buys nothing over width

Every token traverses every level and each level is charged its own prediction,
so the levels are scored on identical data.

| depth | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| bits | 6.740 | 6.394 | 6.289 | 6.227 | 6.132 |
| decrement | — | **0.346** | 0.105 | 0.062 | 0.078 |

Level 1 improves on level 0, so the routing does split the problem. The
decrements then collapse, which reads off the useful depth as three or four
rather than fitting it.

But hierarchy is not what pays. At **identical node count, parameters and write
rule**, a flat mixture of 30 experts beats a depth-4 binary tree:

| topology | nodes | retention gap | MACs/token |
|---|---|---|---|
| flat, 30 experts | 31 | **0.578** | **32,768** |
| tree, depth 4 | 31 | 0.644 | 81,920 |

An earlier comparison had the tree beating a depth-1 arm 0.644 to 1.127; that
arm held three nodes against thirty-one. Depth has not won a single comparison
once node count is held fixed.

---

## 4. The ladder's effect reverses across the tail — and again across the revisit interval

### 4.1 Stratification was necessary

An aggregate bits-per-token over a Zipf marginal is very nearly a measurement of
the head. The ladder's mechanism — protecting rare items from being overwritten
by frequent ones — is a tail effect by construction, and the result it descends
from was reported as a tail metric. Six earlier verdicts in this project were
taken on an average that does not contain the thing being judged.

### 4.2 Ladder effect by rank band and revisit interval

Values are $m{=}4$ minus $m{=}1$ in bits; **positive means the ladder costs**.
Flat topology, 31 nodes, vocabulary 2048, $d = 32$ so no arm can hold the
alphabet, 384k tokens, visit length fixed at 3000.

| revisit interval | rank 1-7 | rank 8-45 | rank 46-304 | **rank 305+** | tail mass |
|---|---|---|---|---|---|
| 12k (4 domains) | +0.605 | +0.729 | +0.382 | **−0.146** | 33.8% |
| 24k (8 domains) | +0.681 | +0.768 | +0.492 | **−0.082** | 36.5% |
| 48k (16 domains) | +0.703 | +0.879 | +0.622 | **+0.008** | 40.9% |
| 96k (32 domains) | +0.645 | +0.801 | +0.657 | **+0.103** | 46.2% |
| 192k (64 domains) | +0.393 | +0.479 | +0.499 | **+0.133** | 51.8% |

Two orthogonal reversals:

**Across rank.** At short intervals the ladder's cost falls monotonically with
rarity and crosses into benefit in the tail. This is the mechanism doing what it
claims, and it appears only in the bounded regime — with a representation wide
enough to hold the alphabet, the tail is not contested and nothing crosses zero.
It also explains a prior failure on BPE-tokenised text, where the tail held
about 1% of the mass: a benefit times 1% cannot pay a head cost.

**Across revisit interval.** The tail benefit shrinks and reverses as gaps
lengthen: $-0.146 \to -0.082 \to +0.008 \to +0.103 \to +0.133$.

This was contrary to prediction. Longer gaps produce far more forgetting — the
head gap grows from +0.61 to +5.05 across the same sweep — so consolidation
should have become *more* valuable, not less.

### 4.3 Why the prediction was wrong

The ladder's operating principle is that repetition is the filter: a one-off
write is pulled back by the next rung and vanishes, a repeated write pushes that
rung the same way each time and survives. Depth of timescale is not the
constraint here — at $m=4$, $r=2$, $g_1 = 1/3000$ the rungs span
$\{3\text{k}, 12\text{k}, 48\text{k}, 192\text{k}\}$, covering the longest gap
tested.

What changes is a domain's **share of traffic**. At four domains a domain owns a
quarter of the stream and its material genuinely repeats; at sixty-four it owns
1.6%, and the deep rungs accumulate the average over all domains, which is the
mixture rather than any part of it. Consolidation preserves what the domains
share and suppresses what distinguishes them, and that suppression gets worse
exactly as the domains get further apart.

> **Consolidation protects material that repeats relative to competing traffic,
> not material that is old. Rarity in time is not the same as rarity in rank,
> and the ladder only helps with the second.**

---

## 5. EWC earns nothing at any interval

Online EWC, proximal form, Fisher normalised by its own mean:

$$w \leftarrow \frac{w + \eta g + \eta\lambda\hat F w^*}{1 + \eta\lambda\hat F}$$

| $\lambda$ | 0 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| total bits | **1,906,864** | 2,107,575 | 2,110,340 | 2,111,603 | 2,111,950 |
| $\lvert$penalty$\rvert / \lvert$gradient$\rvert$ | 0 | 0.54 | 0.62 | 0.72 | 0.83 |

$\lambda = 0$ wins, at every revisit interval from 12k to 192k, and the anchor
timescale makes no difference across a 128-fold sweep (0.18% spread). The
penalty is demonstrably active — it reaches 0.83 of the gradient magnitude — and
still costs about 10%.

So a second consolidation mechanism, of an entirely different family, fails on
the same source. Both are paying for protection against a cost that plain
tracking absorbs more cheaply.

---

## 6. What this establishes and what it does not

**Established.**

1. Prototype routing splits the problem: level 1 improves on level 0 by 0.346
   bits on identical data.
2. Hierarchical depth buys nothing over a flat mixture at matched node count,
   parameters and write rule, at 2.5× the compute.
3. Multi-timescale consolidation reduces tail loss by 0.08–0.15 bits when
   revisits are frequent, and costs 0.4–1.0 bits on the head.
4. That tail benefit vanishes and reverses as the revisit interval grows, for a
   reason internal to the mechanism: consolidation tracks share of traffic, not
   age.
5. EWC's penalty is a net cost at every interval and every anchor timescale
   tested.

**Not established.**

- Any regime in which consolidation is a net win by mass. The tail benefit never
  outweighs the head cost in these experiments.
- Whether a mechanism keyed to *age* rather than *repetition* would behave
  differently. That is the natural reading of §4.3 and it has not been built.
- Anything about real text. The corpus previously used was BPE-tokenised, which
  flattens the head and removes the tail; that line was abandoned rather than
  fixed.

---

## 7. Methodological notes

Recorded because they cost more time than the results did, and because each was
a silent failure — a run that completed and produced ordinary-looking numbers.

**A closed ladder integrates; it does not average.** Injecting a scalar every
step makes the routing threshold outgrow the bounded surprise within a few
hundred tokens, after which novelty never fires and the tree grows as a chain of
width one. Reading value over count from two chains fed in lockstep restores a
running mean and keeps the timescales.

**A node with no prototype scores maximum mismatch by definition.** Charging
that to its own threshold sets the bar at the ceiling, and it then accepts
everything forever. This was masked by the integrator bug — early thresholds
were small enough for a second child to appear — so the test was green with both
bugs present and failed only after one was fixed.

**A Zipf conditional is not a Zipf marginal.** An independent successor
permutation per state gives every context a heavy tail and leaves the marginal
nearly uniform, because the mass lands on a different symbol from each state.
The mechanism under test acts on the marginal.

**A regulariser has to span its own scale.** EWC's gradient carries $O(1)$ on
the target row and $-p \sim 1/V$ elsewhere, so the Fisher sits near $10^{-6}$
for almost every weight and a penalty at $\lambda = 100$ is four orders of
magnitude below what it opposes. Raising the grid diverges instead. No single
$\lambda$ is both effective and stable in an explicit step; the proximal form
removes the problem.

**Assert magnitudes, not directions.** The test guarding that penalty asserted
only that restrained weights were smaller than free ones, which a 0.1% effect
satisfies — and 0.1% was what was happening.

**Match on more than one currency.** Live parameters, total state held and
multiply-accumulates per token size the same comparison differently, and
quoting whichever one equalises the arms is not a control. All three are printed
on every run.

**Check that the binary is the one you think.** A remote build failed with
"access is denied" because a stale job held the executable, and the batch ran to
completion on the previous binary. The failure is silent by construction.
