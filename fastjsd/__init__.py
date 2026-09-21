"""fastjsd -- fast exact pairwise Jensen-Shannon distance.

    >>> import numpy as np
    >>> from fastjsd import jsd_matrix
    >>> P = np.random.default_rng(0).dirichlet(np.full(32, 0.1), size=500)
    >>> D = jsd_matrix(P)          # (500, 500), base 2, distances in [0, 1]

Drop-in for ``scipy.spatial.distance.pdist(P, "jensenshannon")`` and
``cdist(P, Q, "jensenshannon")``, with the same values (base ``e``: pass
``base=np.e``). The harnesses under ``bench/`` in the source repository
measure how the two compare on a given machine.

Public API
----------
jsd_matrix    (n, n) self or (n, m) cross distance matrix
jsd_pdist     condensed upper triangle, SciPy ``pdist`` order
jsd_pairwise  row-aligned distances between two stacks
jsd           scalar distance between two distributions
to_similarity ``1 - d``, valid for base 2
"""

from ._api import (
    SPARSE_DENSITY_MAX,
    SVML_DENSITY_MAX,
    available_backends,
    jsd,
    jsd_matrix,
    jsd_pairwise,
    jsd_pdist,
    resolve_backend,
    to_similarity,
)

__all__ = [
    "jsd", "jsd_matrix", "jsd_pdist", "jsd_pairwise", "to_similarity",
    "available_backends", "resolve_backend", "SPARSE_DENSITY_MAX",
    "SVML_DENSITY_MAX",
    "__version__",
]

__version__ = "0.1.0"
