# fast_gsbs

A faster implementation of GSBS (Greedy State Boundary Search; Geerligs, van Gerven & Güçlü, 2021).
It has the same arguments and outputs as `statesegmentation.GSBS` 0.0.6. What changes is how the
score of each candidate boundary is computed.

On my laptop one fit of a 278 × 1864 fMRI array with kmax 30 took 0.44 s instead of 577 s. On the
inputs I tested, the output was bitwise identical to statesegmentation except in two edge cases,
ties and zero-mean states (see Limitations).

## Install

From a copy of this repository, in a virtual environment:

```
pip install .
```

It needs Python 3.9 or later, NumPy 1.26 or later and SciPy 1.11 or later.

## Usage

```python
from fast_gsbs import GSBS

g = GSBS(kmax=30, x=X)            # X: timepoints x features (e.g. voxels)
g.fit(showProgressBar=False)
g.nstates                         # number of states with the highest t-distance
g.bounds                          # boundaries for that number of states
g.get_bounds(k)                   # boundaries for k states
g.tdists, g.strengths, g.state_patterns
```

The constructor is
`GSBS(kmax, x, statewise_detection=True, finetune=1, finetune_order=True, y=None, blocksize=50, dmin=1)`.
Every argument, method and property means what it means in statesegmentation. `python example.py`
runs a small example.

What is different from statesegmentation:

- NaN or Inf values and constant timepoints raise `ValueError`. statesegmentation runs on such
  input and returns a meaningless result.
- `fit()` prints a plain counter, not a tqdm bar. Pass `showProgressBar=False` in batch jobs.
- A fitted object keeps its running sums and its score cache, so it takes more memory and pickles
  larger. In parallel jobs, return the arrays you need, not the object.
- Results can differ in the two cases under Limitations.

## How it works

statesegmentation scores a segmentation by the mean correlation between each timepoint and the mean
pattern of its state, and it recomputes this over the whole array for every candidate boundary. For
one state S, the sum of these correlations over its timepoints is

```
(w · s) / (‖w‖ √(V − 1))    with  w = Σ (x_t − mean(x_t))  and  s = Σ z_t  over the timepoints of S
```

where `z_t` is timepoint t z-scored across the V features. Both `w` and `s` are differences of
running sums over time, so a state costs O(V) instead of O(T·V), and a candidate boundary changes
only the state it splits. Interval scores are cached.

This scoring is in `fast_gsbs/core.py`, together with the input checks, the warning and the block
search, which is the statesegmentation code with its two scoring calls redirected. The search loop,
fine-tuning, the two-boundary (statewise) step and the t-distance run from the statesegmentation
code, copied into `fast_gsbs/_reference_base.py` with one changed line (the tqdm import) and a
header comment.

## Validation

I compared every output with statesegmentation 0.0.6 in 142 cases, including 19 real fMRI and
language-model arrays. 130 were bitwise identical, 5 were rejected (constant rows) and 7 differed
(2 ties and 5 zero-mean states, see Limitations). I tested on Windows 11 and Linux. 

## Speed

I timed both on my laptop (Intel i7-1360P, single thread) in October 2026, using an earlier version with the same scoring code. Each time is the median of several runs:

| input | statesegmentation | fast_gsbs | ratio |
|---|---|---|---|
| fMRI, 278 × 1864, kmax 30 | 577 s | 0.44 s | 1320 |
| language model, 277 × 1024, kmax 30 | 389 s | 0.47 s | 836 |
| synthetic, 300 × 500, kmax 30 | 150 s | 0.44 s | 338 |
| synthetic, 100 × 50, kmax 20 | 0.73 s | 0.09 s | 8 |

The ratio depends on the machine and its state. A second session on the same laptop gave 937 for
the fMRI array, and two single runs on a cloud Xeon gave 316 and 327 on synthetic arrays of that
size.

fast_gsbs uses more memory than statesegmentation, because it caches one score per interval,
T(T+1)/2 of them with statewise detection (the default). At 1000 timepoints × 100 features the
tracemalloc peak was 122 MB against 37 MB, and it grows with T². For several thousand timepoints,
check the available RAM first.

## Limitations

**Ties.** The scores agree with statesegmentation to rounding (about 2e-15), not bit for bit, and
boundaries are chosen by argmax. When two candidates have equal scores, or scores within rounding
of each other, the two implementations can pick different boundaries, and the later steps then
differ too. I saw this only on artificial inputs: time-symmetric data, noise-free piecewise-constant
data, data with two features, and data with per-feature offsets of about 1e6 times the signal or
more. For exact ties neither choice is better. Which one you get also depends on the NumPy build.
Z-scoring each feature over time removes the offset case.

fast_gsbs computes in float64. On float32 input the scores agree only to about 6e-8, so boundaries
can differ more often; convert to float64 before comparing with statesegmentation.

**Zero-mean states.** If the mean pattern of a state is exactly constant across features, for
example when `x[1] == -x[0]`, statesegmentation gets a NaN score and its argmax picks the NaN.
fast_gsbs handles the NaN differently, so the boundaries and `nstates` can differ: on one of my
test inputs fast_gsbs found 4 states and statesegmentation 2. fast_gsbs emits a
`DegenerateStateWarning` when a state is this close to constant. The warning is a heuristic: it
also fires on harmless data with a very large common offset (1e8), and it does not detect ties.

## Tests

```
pip install -r requirements-test.txt
python -m unittest discover -s tests -t .
```

The 26 tests run on small inputs. 14 of them use statesegmentation 0.0.6 (outputs, interface, ties,
zero-mean states); the others check the input checks, the warning, the progress counter, the version
and the imports. The four tie tests only check that the result is valid, and print whether it
matched statesegmentation on your machine.

## Citation


- Geerligs L., van Gerven M., Güçlü U. (2021). Detecting neural state transitions underlying event
  segmentation. NeuroImage. https://doi.org/10.1016/j.neuroimage.2021.118085
- Geerligs L., Gözükara D., Oetringer D., Campbell K., van Gerven M., Güçlü U. (2022). A partially
  nested cortical hierarchy of neural states underlies event segmentation in the human brain.
  eLife 11, e77430. https://doi.org/10.7554/eLife.77430 (the statewise step)
- statesegmentation: https://github.com/lgeerligs/statesegmentation

## License

MIT. `fast_gsbs/_reference_base.py` is statesegmentation 0.0.6 (MIT, Copyright (c) 2020 Umut
Güçlü) with one changed line and a header comment, and three methods of `fast_gsbs/core.py` are
adapted from it. Both license texts are in `LICENSE`.


