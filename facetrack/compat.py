"""Shims applied before the vendored BoT-SORT / AdaFace imports."""
import sys
import types as _types

import numpy as np

if "pkg_resources" not in sys.modules:
    _pr = _types.ModuleType("pkg_resources")
    _pr.declare_namespace = lambda *a, **kw: None
    sys.modules["pkg_resources"] = _pr

np.float = float
np.int = int
np.bool = bool
