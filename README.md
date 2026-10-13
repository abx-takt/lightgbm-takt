# lightgbm-takt

LightGBM 4.7.0 for Python with faster repeated-call analyses, now on one or several LightGBM
threads. The Python package and API are the official package, unchanged (`import lightgbm`);
only its bundled `lib_lightgbm.so` is replaced. Predictions match the official library on the
same system, bit for bit.

The free package accelerates permutation importance, partial dependence and ICE curves,
MDA across folds and what-if sweeps on dense inputs with up to 300 trees per prediction range.
Supported ranges of a larger loaded model are included. Training, single predictions, ordinary
batch scoring, SHAP contributions, larger prediction ranges and other
paths retain LightGBM's own code and functionality. Not affiliated with or endorsed by
LightGBM or Microsoft.

## Install

```sh
pip uninstall -y lightgbm
pip install -U lightgbm-takt
```

Linux x86-64, glibc 2.27 or newer, Python 3.10–3.14. Both distributions provide
`import lightgbm`, so install them in separate environments when comparing.

## What changed in 4.7.0.post2

Repeated-call acceleration now runs on several LightGBM threads within the same free scope.
The estimator's default thread setting can be kept. Earlier post1 measurements and the
first release's multi-thread slowdown remain in the public showcase's previous report.
The Python API, 300-tree free limit and licence are unchanged.

## Default threads

Whole-process elapsed-time factors against the official wheel at the same thread setting,
with 95% intervals from interleaved runs. Default used 7 allocated CPUs on AMD Threadripper
PRO 5975WX (Zen 3) and 4 on Ryzen 9 7950X3D (Zen 4); neither is a full-machine benchmark.

| Workload | Free, Zen 3 | Free, Zen 4 | Full, Zen 3 | Full, Zen 4 |
|---|---:|---:|---:|---:|
| Permutation importance, quant features | 1.96× (0.98–2.04) | 2.17× (2.10–2.28) | 5.66× (5.19–5.70) | 8.61× (7.43–8.85) |
| PDP + ICE, quant features | 2.68× (2.20–2.86) | 2.64× (2.51–2.76) | 6.94× (6.23–8.29) | 10.65× (10.55–10.75) |
| Permutation importance, categorical + missing data | 2.03× (2.02–2.06) | 2.44× (2.24–2.56) | 3.84× (3.77–3.86) | 5.81× (4.78–6.00) |
| MDA across 5 folds, fitting included | 1.57× (1.55–1.59) | 1.81× (1.78–1.86) | 1.87× (1.78–1.89) | 2.35× (2.30–2.40) |
| PDP + ICE, California housing | 1.35× (1.18–1.37) | 1.24× (1.20–1.28) | 2.96× (1.83–3.03) | 2.76× (2.73–2.86) |
| Permutation importance, synthetic numpy data | 3.00× (2.92–3.01) | 3.27× (3.13–3.33) | 4.47× (4.42–4.49) | 5.43× (3.93–5.54) |
| What-if scenarios, 300 trees | 1.38× (1.36–1.45) | 1.50× (1.43–1.52) | 1.65× (1.62–1.74) | 1.94× (1.79–2.02) |

Free quant models: 300 trees; full quant models: 1,000 trees. Free categorical model:
300 trees; full: 500. California models: 300 and 500. MDA, synthetic and what-if rows use
the same 300-tree models. Each factor has its own official baseline; free and full factors
on different model sizes are not direct comparisons between the two products.

## One thread

| Workload | Free, Zen 3 | Free, Zen 4 | Full, Zen 3 | Full, Zen 4 |
|---|---:|---:|---:|---:|
| Permutation importance, quant features | 3.65× (3.61–3.67) | 3.09× (2.98–3.11) | 18.60× (18.34–18.76) | 17.85× (17.48–17.94) |
| PDP + ICE, quant features | 4.21× (4.18–4.34) | 3.50× (3.43–3.54) | 19.14× (19.00–19.33) | 18.13× (17.70–18.76) |
| Permutation importance, categorical + missing data | 3.38× (3.36–3.41) | 3.32× (3.28–3.38) | 12.35× (12.22–12.56) | 12.84× (12.62–12.88) |
| MDA across 5 folds, fitting included | 2.73× (2.68–2.77) | 2.68× (2.63–2.68) | 5.96× (5.94–6.10) | 5.72× (5.64–5.74) |
| PDP + ICE, California housing | 1.63× (1.60–1.67) | 1.51× (1.48–1.58) | 5.21× (5.16–5.27) | 4.45× (4.33–4.63) |
| Permutation importance, synthetic numpy data | 4.91× (4.83–4.95) | 4.50× (4.42–4.55) | 14.58× (14.31–14.67) | 12.73× (12.32–12.86) |
| What-if scenarios, 300 trees | 2.60× (2.58–2.63) | 2.47× (2.44–2.52) | 4.84× (4.74–4.90) | 4.10× (3.86–4.27) |

Some one-thread workloads take longer than with post1: up to 12.7% in the measured kick
permutation-importance case on Zen 4, while remaining faster than the official wheel.
The showcase retains the interleaved post1 readings. For a
workflow fixed to one prediction thread, measure both versions before upgrading.

Measurements use unchanged scikit-learn functions, 3 interleaved rounds for long analyses
and 5 for short runs; short runs have one warm-up, long analyses have none.
Raw readings retain the harness's quality flags.
Two small flagged differences were examined with 9- and 13-run follow-ups; the showcase
retains the original series and the follow-ups. Other gains depend
on the workload and hardware. All 4/8/default-thread readings, longer 50,000-row analyses,
independent-process checks and C API measurements:
https://taktcycles.com/showcase/lightgbm/README.html?src=pypi-lightgbm

## The same output

Both new libraries passed 4,497 prediction cases, 1,060,559,499 values on each machine,
with zero differing values. Coverage includes 23 public models, numeric/categorical/missing
data, adversarial inputs, output modes, input types, mutation between calls and several
threads. C++ tests: 34 pass; Python: 1,689 pass and the same four graphviz plotting failures
as the official wheel when `dot` is missing. Shared-model checks with 8, 12 and 16 Python
threads found zero mismatches. Eight 15-minute soaks with every accelerated call checked
against the original code found zero mismatches or errors in 1,457,085 calls.

Bit identity is per system: compare on the same machine. LightGBM's upstream concurrency
limits still apply to its original paths, including model mutation during prediction and
concurrent shared-booster calls with different iteration ranges or thread counts.

Verified free `lightgbm/lib/lib_lightgbm.so` SHA-256:
`bfa043475c2cf5f330b1fa822025695e89ffaf0b2ecf7503369d44e88e8dab4d`.

## Verify it yourself

Save the following as `verify.py`. It trains the model on one thread, then performs the
same permutation importance and PDP/ICE with the selected prediction thread count.

```sh
python -m venv a
a/bin/pip install lightgbm==4.7.0 scikit-learn
python -m venv b
b/bin/pip install lightgbm-takt==4.7.0.post2 scikit-learn
a/bin/python verify.py --threads default > a.txt
b/bin/python verify.py --threads default > b.txt
diff a.txt b.txt
```

```python
"""verify.py: run it once with the official `lightgbm` and once with `lightgbm-takt` installed, then compare.

    python verify.py > out.txt
    python verify.py --threads default > out-default.txt

stdout: the permutation importances and a partial-dependence curve at full precision (17 significant digits),
so a byte comparison of the two outputs is a bit comparison. stderr: the time of the analyses.
The model (300 trees) is trained in the script on synthetic data, the same way in both environments.
"""
import argparse
import hashlib
import sys
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.inspection import partial_dependence, permutation_importance

import lightgbm as lgb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--threads', default='1', help='LightGBM prediction threads: positive integer or default')
args = parser.parse_args()
if args.threads == 'default':
    threads = None
else:
    try:
        threads = int(args.threads)
        if threads < 1:
            raise ValueError
    except ValueError:
        parser.error('--threads must be a positive integer or default')

X, y = make_classification(n_samples=40000, n_features=40, n_informative=12, random_state=0)
X = X.astype(np.float32)
Xtr, ytr, Xte, yte = X[:30000], y[:30000], X[30000:35000], y[30000:35000]
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, random_state=0, n_jobs=1,
                         verbosity=-1, deterministic=True)
clf.fit(Xtr, ytr)
clf.set_params(n_jobs=threads)

t0 = time.perf_counter()
r = permutation_importance(clf, Xte, yte, scoring="neg_log_loss", n_repeats=5, random_state=0, n_jobs=1)
pd = partial_dependence(clf, Xte, [0], kind="both", grid_resolution=50)
dt = time.perf_counter() - t0

for row in r.importances:
    print(" ".join("%.17g" % v for v in row))
print(" ".join("%.17g" % v for v in pd["average"].ravel()))
for name, values in (("importances", r.importances), ("average", pd["average"]),
                     ("individual", pd["individual"]), ("grid", pd["grid_values"][0])):
    raw = np.ascontiguousarray(values, dtype=np.float64)
    print(name, raw.shape, hashlib.sha256(raw.tobytes()).hexdigest())
print("lightgbm %s, threads=%s: analyses took %.2f s" % (lgb.__version__, args.threads, dt), file=sys.stderr)
```

Optional variables: `TAKT_LGBM_VERIFY=1` checks accelerated calls with LightGBM's original
code (slow); `TAKT_LGBM_DISABLE=1` disables acceleration; `TAKT_LGBM_CACHE_MB=N` sets the
cache budget, default and free maximum 1 GB; 0 disables acceleration.

## Full version

The commercial version accelerates larger models, C API batch prediction and permutation
explainers; it includes the C/C++ library, support and builds for other platforms/versions
on request. Request a measurement of your own full workflow before ordering.

Team: **$399/month billed yearly ($4,788)**, one platform with new LightGBM releases rebuilt
and checked. Enterprise: **from $1,500/month billed yearly**, other platforms and versions,
SLA, source escrow and acceptance on your hardware, under the accepted order and Terms.

Description of your workload or a quote: https://taktcycles.com/?src=pypi-lightgbm#start

Measurements and updates: https://t.me/taktcycles

## Licence

The Python package is LightGBM's own code, the official LightGBM 4.7.0 package, unchanged, under the MIT licence
(`LICENSE`). The file `lightgbm/lib/lib_lightgbm.so` is a modified build of LightGBM 4.7.0: it is the only file
with TAKT contributions, which are licensed under the TAKT Free Binary Licence (`LICENSE-TAKT.txt`); the LightGBM
code in it remains under the MIT licence. See `NOTICE`. The source of the TAKT contributions is not published.
This package is not an unmodified LightGBM release.

Questions, a result that differs, a build you need, a function you want faster:
https://github.com/abx-takt/lightgbm-takt/issues
