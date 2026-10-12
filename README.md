# lightgbm-takt

LightGBM 4.7.0 for Python with faster repeated-call analyses. The Python package is the official LightGBM wheel,
unchanged (`import lightgbm`); only the library inside it, `lightgbm/lib/lib_lightgbm.so`, is replaced by a TAKT
build of LightGBM 4.7.0. Every prediction is identical to the official wheel, bit for bit, on the same machine.

**What this free package accelerates, and its limits.** Repeated-call analyses are accelerated on models with up
to 300 trees: permutation importance, partial dependence and ICE curves, MDA feature importance across
cross-validation folds, and what-if scenario sweeps, which call `predict` many times. The accelerated part runs on
one thread; when LightGBM itself runs on several threads, it is used only where it is faster than LightGBM's own
multi-threaded code. Everything else runs LightGBM's own code at its normal speed, with the full LightGBM
functionality available: training, single predictions, batch scoring, SHAP values, models with more than 300
trees, every parameter and input type. A commercial licence covers the full version (see "Full version" below).

Not affiliated with or endorsed by the LightGBM project or Microsoft.

## Install

```sh
pip uninstall -y lightgbm        # both packages provide `import lightgbm`
pip install lightgbm-takt
```

Linux x86-64, glibc 2.27 or newer (the same platforms as the official wheel), Python 3.10 to 3.14.
`import lightgbm`, its API and its dependencies are unchanged.

## What is faster

Factors over the official LightGBM 4.7.0 wheel for the whole Python process (start-up, imports, data loading,
the analysis), with LightGBM on one thread, by CPU cycles (`cycles:u`) with 95% confidence intervals, measured on
AMD Zen 3 (Threadripper PRO 5975WX) and AMD Zen 4 (Ryzen 9 7950X3D). The output was identical to the official
wheel in every run. The rows marked "LightGBM's own code" show that everything outside the free scope runs at the
official speed.

| Analysis | AMD Zen 3 | AMD Zen 4 |
|---|---:|---:|
| Permutation importance (Alpha158 quant features, 300 trees) | **3.67×** (3.60–3.75) | **3.40×** (3.36–3.48) |
| Partial dependence + ICE (the same model, 20 features) | **4.23×** (4.20–4.36) | **3.84×** (3.79–3.90) |
| Permutation importance (OpenML kick, categorical and missing values, 300 trees) | **3.71×** (3.65–3.74) | **3.81×** (3.70–3.93) |
| Permutation importance (synthetic data, 50 features, 300 trees, numpy) | **5.04×** (5.01–5.09) | **4.93×** (4.84–5.01) |
| What-if scenarios (kick model, 30 scenarios) | **2.88×** (2.85–2.89) | **3.08×** (3.02–3.15) |
| MDA importance across 5 folds, fitting included | **2.96×** (2.95–2.98) | **2.98×** (2.95–3.02) |
| Partial dependence + ICE (California housing regressor, 300 trees) | **1.71×** (1.69–1.77) | **1.53×** (1.45–1.62) |
| Batch `predict_proba`, quant features (LightGBM's own code) | 1.00× (0.99–1.01) | 1.01× (0.98–1.03) |
| Batch `predict_proba`, kick (LightGBM's own code) | 0.99× (0.98–1.00) | 1.00× (0.98–1.01) |
| `shap.PermutationExplainer` (LightGBM's own code) | 1.00× (0.99–1.02) | 1.02× (1.00–1.04) |
| Permutation importance on a 500-tree model (over the limit: LightGBM's own code) | 1.02× (1.00–1.03) | 1.00× (1.00–1.02) |
| Permutation importance on a 1,000-tree model (over the limit: LightGBM's own code) | 1.01× (1.00–1.03) | 1.02× (1.01–1.03) |
| C API batch prediction, one thread (LightGBM's own code) | 1.01× (1.00–1.04) | 1.01× (0.98–1.04) |

The analyses are scikit-learn's `permutation_importance` and `partial_dependence` (`kind="both"`, grid of 50)
called on an `LGBMClassifier`/`LGBMRegressor` with `n_jobs=1`, unchanged. Every reading and the scripts:
https://taktcycles.com/showcase/lightgbm/README.html?src=pypi-lightgbm

**With LightGBM on several threads** (its default in the scikit-learn API), the gain is smaller or none, and no
measured analysis is slower than with the official wheel beyond its confidence interval. Elapsed-time factors with LightGBM's default thread
count (7 CPUs on Zen 3, 4 cores on Zen 4), 95% confidence intervals:

| Analysis, LightGBM's default threads | AMD Zen 3 | AMD Zen 4 |
|---|---:|---:|
| Permutation importance (synthetic data, 50 features, numpy) | 1.09× (1.07–1.11) | 1.40× (1.37–1.42) |
| The same with a pandas DataFrame | 1.11× (1.04–1.15) | 1.42× (1.36–1.44) |
| Permutation importance (OpenML kick, 300 trees) | 1.28× (1.27–1.29) | 1.69× (1.66–1.71) |
| MDA importance across 5 folds | 1.10× (1.09–1.11) | 1.36× (1.35–1.39) |
| What-if scenarios (kick model) | 1.03× (1.02–1.03) | 1.09× (1.06–1.10) |
| Permutation importance (Alpha158 quant features, 300 trees) | 1.00× (0.97–1.04) | 1.03× (0.94–1.11) |
| Partial dependence + ICE (the same model) | 1.00× (0.99–1.02) | 1.03× (0.95–1.06) |
| Partial dependence + ICE (California housing) | 1.01× (0.98–1.03) | 1.03× (0.99–1.05) |

Readings with 4 and 8 LightGBM threads, and with LightGBM's own code (batch prediction, explainers, larger
models), are in the showcase; none is slower than the official wheel beyond its confidence interval.

## What changed in 4.7.0.post1

- **Several LightGBM threads.** In 4.7.0 the accelerated part could be slower than LightGBM's own multi-threaded
  prediction, so analyses run with LightGBM's default threads took up to 1.47 times as long as with the official
  wheel (partial dependence on the quant-feature model, Zen 3, 7 threads). 4.7.0.post1 uses the accelerated part
  only where it is faster and LightGBM's own code everywhere else.
- **One thread.** Sweeps that change a different column at each step (permutation importance, what-if scenarios)
  are faster than in 4.7.0, for example permutation importance on the kick model
  3.71× instead of 2.65× on Zen 3.
- Unchanged: the scope (repeated-call analyses on models up to 300 trees), the Python package, and every output,
  identical to the official wheel bit for bit.

## The same output, bit for bit

Every number this package returns matches the official LightGBM wheel on the same machine. We check 4,497
prediction cases (about 1.06 billion values) on 23 public models: regression, binary, multiclass, ranking, DART,
random forest, linear trees, categorical- and missing-heavy data, adversarial inputs (NaN, ±inf, −0, subnormals,
values exactly on split thresholds), every input type and prediction type, on one and several threads.
LightGBM's own C++ test suite (34 of 34) and Python test suite (1,689 passed; the same 4 graphviz plotting tests
fail with the official wheel when the `dot` binary is missing) give the same results as with the official wheel.
Five 30-minute soak runs on the two machines, with LightGBM on 1, 2 and 4 threads and 12 Python threads each,
every accelerated call recomputed by LightGBM's original code and compared, found no difference in 675,078 calls.

Check it yourself. The script `verify.py` (in the GitHub repository, reproduced below) trains a 300-tree model on
synthetic data and prints a permutation importance and a partial dependence curve to 17 significant digits; the time
of the analyses goes to stderr. Run it once in an environment with `lightgbm` and once with `lightgbm-takt`, and
compare:

```sh
python -m venv a && a/bin/pip install -q lightgbm==4.7.0 scikit-learn && a/bin/python verify.py > a.txt
python -m venv c && c/bin/pip install -q lightgbm-takt scikit-learn  && c/bin/python verify.py > c.txt
diff a.txt c.txt && echo "identical"
```

```python
# verify.py
import sys
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.inspection import partial_dependence, permutation_importance

import lightgbm as lgb

X, y = make_classification(n_samples=40000, n_features=40, n_informative=12, random_state=0)
X = X.astype(np.float32)
Xtr, ytr, Xte, yte = X[:30000], y[:30000], X[30000:35000], y[30000:35000]
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, random_state=0, n_jobs=1,
                         verbosity=-1, deterministic=True)
clf.fit(Xtr, ytr)

t0 = time.perf_counter()
r = permutation_importance(clf, Xte, yte, scoring="neg_log_loss", n_repeats=5, random_state=0, n_jobs=1)
pd = partial_dependence(clf, Xte, [0], kind="both", grid_resolution=50)
dt = time.perf_counter() - t0

for row in r.importances:
    print(" ".join("%.17g" % v for v in row))
print(" ".join("%.17g" % v for v in pd["average"].ravel()))
print("ice sum %.17g" % float(np.sum(pd["individual"])))
print("lightgbm %s: analyses took %.2f s" % (lgb.__version__, dt), file=sys.stderr)
```

Optional environment variables: `TAKT_LGBM_VERIFY=1` recomputes every accelerated call with LightGBM's original
code and reports any difference on stderr (slow; for checking). `TAKT_LGBM_DISABLE=1` turns the acceleration off.
The acceleration uses some extra memory, at most 1 GB per process; `TAKT_LGBM_CACHE_MB=N` sets a lower limit (0
turns the acceleration off).

## Full version

The commercial licence covers the full version:

- **Every model size**, and the repeated-call analyses on several threads. Measured on one thread: permutation
  importance and PDP/ICE 16.8–18.4× faster on a 1,000-tree classifier, permutation importance 13.0–14.6× on a
  500-tree classifier with categorical features, PDP/ICE 4.8–5.4× on a 500-tree regressor, MDA across folds
  5.9–6.1×.
- **Faster batch prediction** through the C API, on one or several threads: 4.0–5.4× in CPU time for `predict`
  (2.3–4.4× elapsed), and a faster `shap.PermutationExplainer` (3.2–4.0×).
- **The C/C++ library** (`lib_lightgbm.so`), a drop-in with the same exports for programs and bindings that load it.
- **Builds for other platforms and LightGBM versions** on request.
- **Support** and **source escrow**: the modified source in an encrypted escrow box, released to you on the
  conditions of our Terms of Service.

Prices: **Team**, $399 a month billed yearly ($4,788): the full version on one platform, with every new LightGBM
release rebuilt and checked bit for bit again. **Enterprise**, from $1,500 a month billed yearly: other platforms
and LightGBM versions, an SLA, source escrow and acceptance on your own hardware.

All measurements, free and full: https://taktcycles.com/showcase/lightgbm/README.html?src=pypi-lightgbm

New measurements of other libraries, one a week: https://t.me/taktcycles

Get a licence, a quote, or a free speed check of your own code:
https://taktcycles.com/?src=pypi-lightgbm#start

## Licence

The Python package is LightGBM's own code, the official LightGBM 4.7.0 package, unchanged, under the MIT licence
(`LICENSE`). The file `lightgbm/lib/lib_lightgbm.so` is a modified build of LightGBM 4.7.0: it is the only file
with TAKT contributions, which are licensed under the TAKT Free Binary Licence (`LICENSE-TAKT.txt`); the LightGBM
code in it remains under the MIT licence. See `NOTICE`. The source of the TAKT contributions is not published.
This package is not an unmodified LightGBM release.

Questions, a result that differs, a build you need, a function you want faster:
https://github.com/abx-takt/lightgbm-takt/issues
