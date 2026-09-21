# Benchmark fixtures

Two real soft-clustering membership matrices, `(n_rows, n_clusters)` float64,
saved as plain `.npy`. Each row is an HDBSCAN `all_points_membership_vectors`
output -- a point's soft assignment over the discovered clusters.

They are here because synthetic Dirichlet draws do not reproduce the shape of
real membership data: real rows are *dense but wildly unequal* (row sums range
over ~0.05-1.0 before normalization, and mass concentrates on a handful of
clusters), which is exactly the regime where kernel choice matters.

| file | shape | nonzero fraction | notes |
|---|---|---|---|
| `soft_clusters_931x25.npy` | (931, 25) | 1.00 | raw memberships, rows not normalized |
| `soft_clusters_1014x40.npy` | (1014, 40) | 1.00 | raw memberships, rows not normalized |

`bench/datasets.py` also derives the **thresholded** variant of each (drop
entries below `1/k`, renormalize -- the standard way these vectors are cleaned
before use), which lands at 12-24 % density and exercises the sparse kernel.

No labels, text, identifiers or any other content is included -- these are
numeric membership matrices only.
