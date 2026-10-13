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
