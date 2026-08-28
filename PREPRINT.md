# Content-Addressed Continual Memory with On-Demand Capacity

*Draft v1. Numbers marked [PENDING] await runs in flight; every other figure is
produced by `mainacc.py` from the logs in `/home/pana/annp-runs`.*

## Abstract

We describe a continual-learning memory in which capacity is addressed by
content, allocated on demand from the stream without task identifiers, and
expanded without disturbing anything already stored. On a relational stream
whose domains are distinguishable, it reaches 49.5% zero-shot retention where a
parameter-matched monolithic model reaches 15.0%. Sharing the transform across
regimes while keeping per-regime readouts adds 16.4 points over a fully private
architecture (six seeds, p < 0.001). Allocating readout rows only for targets a
regime actually emits reduces occupied capacity 7.6x at no cost in accuracy,
and adding capacity leaves existing weights byte-identical and existing
predictions bit-identical.

We also report what the architecture does not do. It has no mechanism for
associating items across a gap in the stream, and we measure the resulting
degradation. Multi-hop routing, which could in principle buy combinatorial
addressing from linear memory, does not: it occupies the same capacity for 25
points less accuracy. Retention declines as regimes accumulate, and we show the
decline is not capacity starvation, not worsening misrouting, and not failing
separation, but retrieval difficulty scaling with what is stored.

## 1. Mechanism

A fact is a pair (entity, relation) with a target. The payload is
`p = normalise(emb[entity] + emb[relation])` over fixed random embeddings that
are never learned.

**Addressing** has two independent factors:

- *Content to edge.* Each node holds fixed random keys; the walk takes
  `argmax_slot <key, p>`. Nothing here is learned, so a fact always reaches the
  same edge and routing consistency is 1.000 by construction.
- *Regime to slice.* A slow context EMA is matched against prototypes;
  `argmax_c <proto[c], ctx>` picks the readout slice. No task identifier is
  used at any point, and the class is never told which domain is live.

**Memory** is likewise split. The edge transform `E[edge]` applies
`p <- normalise(p + tanh(E p))` and is shared across regimes. The readout
`R[class]` maps the transformed payload to logits and is private per regime.
The split follows the data rather than preference: a payload is built from
entity and relation, which are common across regimes, while targets are what
regimes disagree about.

**Learning** is online, single-pass and prequential -- predict, take the loss,
then write. There is no replay buffer, no task boundary and no second pass.
The readout takes a delta-rule step on the target row plus sampled negatives
drawn from targets that regime has emitted; gradients flow back along the
walked path into the shared transform. Routers receive no gradient.

**Capacity** is allocated when a class's similarity dispersion shows it is
serving heterogeneous content, and a new prototype is placed at the current
context. Allocation appends: existing slices keep their offsets and values.

## 2. Results

### 2.1 Where regimes are distinguishable

Mode A, eight regimes, three seeds [PENDING: seeds 2-3 of the control]:

| | zero-shot |
|---|---|
| this work | **49.53%** |
| monolithic, parameter-matched | **15.00%** |
| linear, no sparse coding | 15.30% |

### 2.2 Sharing the transform

Mode B, twelve regimes, six seeds:

| | zero-shot | paired difference |
|---|---|---|
| shared transform, private readouts | **29.42%** | +16.43pp, 6/6 seeds, **p < 0.001** |
| fully private | 12.98% | -- |
| monolithic | 21.28% | +8.13pp, 5/6, p = 0.119 (not established) |

Sharing the *readout* as well is harmful: -27.50pp. A common readout block
accumulates every regime's target mapping, and Mode B is defined by the same
(entity, relation) pointing at different targets per regime, so it averages
contradictions.

### 2.3 Capacity follows content

The dense delta rule touches every vocabulary row on every write, so a regime's
readout fills although it can only need rows for targets it emits. Drawing
negatives from emitted targets instead:

| | occupied rows | rows per class | accuracy |
|---|---|---|---|
| dense | 45056 / 45056 (100%) | 4096 | 32.6% |
| sampled | 5918 / 45056 (**13.1%**) | **538** | 33.8% |

### 2.4 Expansion does not disturb what is stored

`expand_classes` appends a slice. A test asserts that every existing weight is
byte-identical afterwards and that `forward` returns bit-identical logits --
not approximately, exactly. A monolithic model cannot do this: its capacity is
the sparse-coding width, and widening it changes the projection every stored
code was written through.

### 2.5 Cost per token does not grow with context

Context access is `H d^2` per token, independent of sequence length, against
attention's `N d`. The crossover is at `N = d`; at N = 4096 the ratio is 128x.

Stated against ourselves: overall activation is 6.178% of memory parameters,
of which the edge memory is 1024 and the remainder is the readout scoring all
V rows. Sparsity currently holds for the expert part only.

## 3. What it does not do

### 3.1 Association across a gap

The architecture carries nothing between facts except a single-timescale
context EMA, so a cue survives `(1-r)^gap`. With the probe rebuilding its
context from a presented episode rather than a restored per-domain snapshot,
retention falls with the gap:

| gap | cue remaining | accuracy |
|---|---|---|
| 1 | 0.990 | 10.10% |
| 16 | 0.851 | 10.33% |
| 64 | 0.526 | 8.40% |
| 256 | 0.076 | 6.83% |

Paired degradation +3.27pp, 5/6 seeds, p = 0.013, and the cliff falls where
`ctx_rate` predicts before the run.

Exact association under constant memory is information-theoretically
unavailable, so the question is what the lossy code should retain. A
multi-scale average answers "which regime" and cannot answer "which item";
putting the context into the address recovers about a tenth of the gap and
costs lookup accuracy.

### 3.2 Multi-hop

H hops distinguish a fact by a path, which could buy combinatorial addressing
from linear memory. Measured with regimes separating, on identical occupancy:

| H | zero-shot | occupied |
|---|---|---|
| 1 | **55.15%** | 750 |
| 2 | 46.00% | 737 |
| 3 | 29.63% | 750 |

The under-training explanation is excluded by direct measurement: writes per
edge *rise* with hops, 3246 to 9222, because each fact touches more edges.

### 3.3 Retention declines as regimes accumulate

| regimes | zero-shot | occupied | rows/regime | home classes per regime |
|---|---|---|---|---|
| 8 | 49.53% | 530 | 66 | 0.75 |
| 16 | 43.07% | 1142 | 71 | 0.64 |
| 32 | 37.65% | 2412 | 75 | 0.70 |
| 64 | 34.80% | 5198 | 81 | 0.78 |

[PENDING: monolithic baselines beyond eight regimes.]

Three explanations are excluded by the data. Capacity is supplied linearly, so
it is not starvation. Misrouted write magnitude is 57.0, 54.7, 51.0, 55.0 per
cent -- flat, so it is not worsening interference. Home classes per regime is
flat, so separation does not degrade.

What remains is retrieval difficulty. Confusable stored targets grow 9.8x
while accuracy falls to 0.70x, and distinct predictions rise from 47 to 57
rather than collapsing, so the memory keeps discriminating broadly. The
decline is sub-proportional to what is stored.

## 4. Open questions

1. **Does savings exist here?** Later regimes cost less per unit accuracy
   (ratio 0.75-0.88), but the control arm shows it too, so it is not yet
   attributable to sharing. [PENDING: position controls.]
2. **Is the decline a power law?** `acc ~ N^-alpha` with alpha about 0.16 over
   the measured range, and whether stronger addressing rather than more
   capacity moves the exponent.
3. **What should a constant-memory context retain?** Sparse retention of
   salient items, or superposition with approximate unbinding, instead of a
   multi-scale average.
4. **Does the plasticity-stability frontier extend or merely shift?** The two
   frontiers barely overlap, so at present only reachability can be compared.

## 5. Methodological note

Six measurements in this work were invalidated after collection and re-run.
Three probe-side leaks handed the model information it was meant to infer; a
growth criterion compared novelty against a global distribution and collapsed
every regime into one class at scale; log parsing read a downstream table and
reported 54.2% as 100.00%.

Each was caught by a consistency check rather than by inspection: a statistic
that contradicted another statistic, a diagnostic that printed nothing, an
accuracy that could not coexist with its own exposure counts. The instruments
that survived -- write locality by time-since-switch, occupancy against
content, growth timing against boundaries -- are reported here because they
are what made the failures visible.
