"""Argument handling, validation and the contracts the docstrings promise."""

import numpy as np
import pytest
from conftest import dirichlet, thresholded

import fastjsd
from fastjsd import available_backends, jsd_matrix, jsd_pdist, resolve_backend


def test_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        jsd_matrix(np.array([[0.5, -0.5], [1.0, 0.0]]))


def test_rejects_non_finite():
    with pytest.raises(ValueError, match="NaN or infinite"):
        jsd_matrix(np.array([[np.nan, 1.0], [1.0, 0.0]]))


def test_rejects_all_zero_row():
    with pytest.raises(ValueError, match="cannot normalize"):
        jsd_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_rejects_unnormalized_input_when_normalize_is_off():
    with pytest.raises(ValueError, match="not normalized"):
        jsd_matrix(np.array([[2.0, 1.0], [1.0, 0.0]]), normalize=False)


def test_normalize_off_is_a_fast_path_not_a_different_answer():
    """Skipping the division changes the last ulp of already-normalized rows
    (dividing by 1.0 +- 1 ulp is not the identity), never the answer."""
    P = thresholded(30, 10, 0.2, seed=40)
    assert np.abs(jsd_matrix(P) - jsd_matrix(P, normalize=False)).max() < 1e-15


def test_input_is_never_mutated():
    P = dirichlet(20, 6, 0.4, seed=41) * 3.0
    before = P.copy()
    jsd_matrix(P)
    jsd_pdist(P)
    assert np.array_equal(P, before)


def test_rejects_mismatched_column_counts():
    with pytest.raises(ValueError, match="same number of columns"):
        jsd_matrix(dirichlet(5, 4, 1.0), dirichlet(5, 3, 1.0))


def test_rejects_bad_base():
    for bad in (1.0, 0.5, 0.0, -2.0, np.inf):
        with pytest.raises(ValueError, match="base must be"):
            jsd_matrix(dirichlet(4, 3, 1.0), base=bad)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        jsd_matrix(dirichlet(4, 3, 1.0), backend="cuda")


def test_one_dimensional_input_is_treated_as_a_single_row():
    p = np.array([0.2, 0.8])
    assert jsd_matrix(p).shape == (1, 1)


def test_accepts_integer_and_non_contiguous_input(backend):
    counts = np.array([[3, 1, 0], [0, 2, 2], [1, 1, 1]])
    D = jsd_matrix(counts, backend=backend)
    assert D.dtype == np.float64
    wide = np.repeat(dirichlet(12, 5, 0.5, seed=42), 2, axis=1)[:, ::2]
    assert not wide.flags.c_contiguous
    assert np.isfinite(jsd_matrix(wide, backend=backend)).all()


def test_out_parameter(backend):
    P = thresholded(24, 10, 0.2, seed=43)
    buf = np.empty((24, 24))
    got = jsd_matrix(P, out=buf, backend=backend)
    assert got is buf
    assert np.array_equal(buf, jsd_matrix(P, backend=backend))

    cbuf = np.empty(24 * 23 // 2)
    assert jsd_pdist(P, out=cbuf, backend=backend) is cbuf


@pytest.mark.parametrize("bad,msg", [
    (np.empty((3, 3)), "shape"),
    (np.empty((24, 24), dtype=np.float32), "dtype"),
    (np.asfortranarray(np.empty((24, 24))), "contiguous"),
])
def test_out_parameter_is_validated(bad, msg):
    with pytest.raises((ValueError, TypeError), match=msg):
        jsd_matrix(thresholded(24, 10, 0.2, seed=44), out=bad)


def test_backend_resolution_follows_density():
    """Dense input goes to *a* dense kernel, sparse input to the merge.

    Which dense kernel depends on what is installed: `numba-svml` when Intel's
    libsvml and AVX-512 are both present, `numba` otherwise. Asserting the
    literal `"numba"` here would fail on an SVML-capable machine for a
    perfectly correct dispatch.
    """
    dense = dirichlet(20, 8, 5.0, seed=45)
    sparse = thresholded(20, 40, 0.02, seed=46)
    if "numba" in available_backends():
        assert resolve_backend("auto", dense) in ("numba", "numba-svml")
        assert resolve_backend("auto", sparse) == "numba-sparse"
    else:
        assert resolve_backend("auto", dense) == "numpy"


def test_membership_density_still_dispatches_to_the_merge():
    """The band that thresholded membership data actually lands in.

    The test above uses density 0.02, far below every threshold, so it could
    never catch a threshold set too low. This one sits at ~0.24, the density of
    `bench/fixtures/soft_clusters_931x25` after the standard 1/k threshold. A
    `SVML_DENSITY_MAX` set below this band routes membership-shaped data to the
    SVML dense kernel, which evaluates the full `k` logarithms whatever the
    support and so gives up every coordinate the merge would have skipped.

    Asserting the choice rather than the constant, so retuning stays free as
    long as the intent survives.
    """
    P = thresholded(200, 25, 0.35, seed=47)
    d = float((P > 0).mean())
    assert 0.20 < d < 0.30, f"fixture drifted out of the band under test: {d}"
    expected = "numba-sparse" if "numba" in available_backends() else "numpy-sparse"
    assert resolve_backend("auto", P) == expected


def test_numba_backend_reports_clearly_when_unavailable():
    if "numba" in available_backends():
        pytest.skip("numba is installed")
    with pytest.raises(RuntimeError, match="requires numba"):
        jsd_matrix(dirichlet(4, 3, 1.0), backend="numba")


def test_threads_argument_does_not_change_results():
    P = thresholded(64, 20, 0.1, seed=47)
    assert np.array_equal(jsd_matrix(P), jsd_matrix(P, threads=1))


def test_numpy_tiling_is_transparent():
    """Tile size is a tuning knob, never a correctness parameter."""
    from fastjsd import _numpy
    P = thresholded(70, 24, 0.1, seed=48)
    ref = jsd_matrix(P, backend="numpy")
    for budget in (4096, 1 << 16, 1 << 24):
        got = _numpy.matrix(P / P.sum(1, keepdims=True), None,
                            1.0 / np.log(2.0), False, None, tile_bytes=budget)
        assert np.abs(got - ref).max() < 1e-15


def test_version_and_exports():
    assert isinstance(fastjsd.__version__, str)
    for name in fastjsd.__all__:
        assert hasattr(fastjsd, name)
