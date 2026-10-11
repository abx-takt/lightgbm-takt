---
name: Output differs or something breaks
about: A prediction that differs from the official LightGBM, a crash, or an install problem
title: "Problem: "
labels: bug
---

**What happened** (function, model type and size, the difference you see):

**How to reproduce** (a few lines of Python; the data shape and dtype):

**Environment** (`pip show lightgbm-takt`, Python version, numpy version, CPU model, Linux distribution, glibc version):

Remember: some results depend on the system maths library, so compare lightgbm-takt against the official
`lightgbm` wheel **on the same machine** (see the README's verify script). `TAKT_LGBM_VERIFY=1` recomputes every
accelerated call with LightGBM's original code and reports any difference on stderr.
