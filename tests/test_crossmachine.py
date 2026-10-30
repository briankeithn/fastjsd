"""The cross-machine benchmark: the leg (`bench/portable.py`) and the aggregator
(`bench/aggregate.py`).

That workflow is dispatched by hand, after a release, on boxes nothing else
here tests on -- so a bug in it surfaces exactly when it is most awkward to fix.
It is tested here instead, where CI runs: the aggregator on hand-built legs
whose verdicts are known in advance, and the leg end to end on a tiny case set,
through the aggregator, so that the JSON one writes is the JSON the other reads.
"""

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fastjsd import available_backends, jsd_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))

import aggregate  # noqa: E402  (stdlib only, so this imports on every leg)
import portable  # noqa: E402

needs_f32 = pytest.mark.skipif("numba-f32" not in available_backends(),
                               reason="numba-f32 needs numba")


def _ratio(values):
    values = list(values)
    return {"per_repeat": values, "min": min(values), "max": max(values),
            "median": sorted(values)[len(values) // 2],
            "resolvable": min(values) > 1.0 or max(values) < 1.0}


def _leg(leg_id, *, f32=(3.0, 3.2, 3.4), numba=True, f32_status="ok",
         exact_zero=True, exact_one=True, f32_abs=1e-7):
    """A minimal schema-/2 leg: one case against SciPy, one numba-f32 case.

    Milliseconds are built from the ratios, so every ratio brackets the ratio
    of its own minima, as the leg's are.
    """
    case = {"case": "dense n=8 k=4", "n": 8, "k": 4, "density": 1.0,
            "input_sha256": "same",
            "impls": {"fastjsd auto": {"status": "ok", "ms": [10.0] * 3,
                                       "ms_min": 10.0, "agrees": True},
                      "scipy pdist": {"status": "ok", "ms": [50.0, 60.0, 70.0],
                                      "ms_min": 50.0, "agrees": True}},
            "ratios": {"scipy pdist": _ratio((5.0, 6.0, 7.0))}}
    if f32_status == "ok":
        f32_row = {"case": "dense n=8 k=4", "n": 8, "k": 4, "status": "ok",
                   "impls": {"numba": {"ms": [2.0 * r for r in f32],
                                       "ms_min": 2.0 * min(f32)},
                             "numba-f32": {"ms": [2.0] * 3, "ms_min": 2.0}},
                   "ratio": _ratio(f32), "max_abs_err": 1e-7, "agrees": True,
                   "tol": portable.F32_TOL}
    else:
        # A failed row is skipped on its status, not on a missing ratio, so
        # this one carries a ratio the aggregator must still not count.
        f32_row = {"case": "dense n=8 k=4", "n": 8, "k": 4,
                   "status": "failed", "reason": "LoweringError: boom",
                   "impls": {}, "ratio": _ratio(f32)}
    f32_block = ({"status": "ok", "cases": [f32_row]} if numba else
                 {"status": "absent", "reason": "no numba", "cases": []})
    return {
        "schema": aggregate.SCHEMA, "leg_id": leg_id,
        "machine": {"cpu_usable": 2, "cpu_model": "test cpu"},
        "software": {"numba": "0.60.0" if numba else None},
        "validity": {"verdict": "ok", "reasons": []},
        "cases": [case], "f32": f32_block,
        "accuracy": {
            "duplicate_rows_exact_zero": {"fastjsd auto": exact_zero},
            "disjoint_supports_exact_one": {"fastjsd auto": True,
                                            "fastjsd numba-f32": exact_one},
            "vs_mpmath": {"status": "ok", "per_regime": {"random": {
                "numba-f32": {"abs": f32_abs, "rel": 2 * f32_abs}}}},
        },
        "caveats": [],
    }


def _text(summary):
    out = []
    aggregate.report(summary, out.append)
    return "\n".join(out)


# --- the aggregator ----------------------------------------------------------

def test_f32_section_is_counted_over_boxes():
    s = aggregate.build([_leg("a"), _leg("b", f32=(2.0, 2.1, 2.2))])
    v = s["f32"]["verdict"]
    assert s["f32"]["n_boxes"] == 2
    assert (v["boxes_won"], v["n_cells"], v["cells_lost"]) == (2, 2, 0)
    assert v["range"] == [2.1, 3.2]                  # per-cell medians
    assert ("[numba-f32] numba-f32 was faster than numba on 2/2 boxes"
            in "\n".join(s["verdict_lines"]))
    assert not s["problems"]


def test_f32_loss_is_printed_before_its_section():
    s = aggregate.build([_leg("fast"), _leg("slow", f32=(0.7, 0.8, 0.9))])
    text = _text(s)
    head = text.index("LOSSES: numba-f32 was SLOWER than numba in "
                      "1 case-box cell")
    section = text.index("numba-f32 vs numba   (dense cases")
    assert head < section
    assert "slow" in text[head:section]
    assert "numba-f32 was SLOWER than numba in 1 of 2" in text


def test_leg_without_numba_contributes_no_f32_cell():
    s = aggregate.build([_leg("np", numba=False)])
    assert s["f32"] is None
    assert "no leg produced a comparable cell" in _text(s)


def test_failed_f32_case_is_noted_and_not_counted():
    s = aggregate.build([_leg("broken", f32_status="failed"), _leg("ok")])
    assert s["f32"]["n_boxes"] == 1
    assert any("broken" in p and "did not run" in p for p in s["problems"])


def test_f32_ratio_must_bracket_its_own_milliseconds():
    leg = _leg("a")
    leg["f32"]["cases"][0]["impls"]["numba"]["ms_min"] *= 2
    notes = aggregate.crosscheck(leg)
    assert any("numba-f32 ratio block" in n and "does not bracket" in n
               for n in notes)


def test_exactness_failures_are_reported_from_every_leg():
    # An excluded (too noisy to time) box still reports its exactness: runner
    # noise cannot touch a deterministic check.
    noisy = _leg("noisy", exact_one=False, exact_zero=False)
    noisy["validity"] = {"verdict": "unusable", "reasons": ["steal"]}
    s = aggregate.build([_leg("a"), noisy])
    assert s["disjoint_failures"] == {"noisy": ["fastjsd numba-f32"]}
    assert s["exactness_failures"] == {"noisy": ["fastjsd auto"]}
    text = _text(s)
    assert "disjoint supports did not come back exactly 1.0" in text
    assert "identical rows did not come back exactly 0.0" in text


def test_f32_accuracy_names_the_worst_box():
    s = aggregate.build([_leg("a", f32_abs=1e-7), _leg("b", f32_abs=3e-7)])
    a = s["f32_accuracy"]["random"]
    assert (a["worst_abs"], a["worst_abs_box"], a["n_boxes"]) == (3e-7, "b", 2)


def test_old_schema_is_refused(tmp_path):
    path = tmp_path / "portable-old.json"
    path.write_text(json.dumps({**_leg("old"),
                                "schema": "fastjsd-crossmachine/1"}))
    legs, problems = aggregate.load([str(path)])
    assert legs == []
    assert "refusing" in problems[0]


# --- the leg -------------------------------------------------------------------

def test_disjoint_check_holds_only_fastjsd_and_catches_a_miss():
    def numpy_backend(P):
        return jsd_matrix(P, backend="numpy")

    def one_ulp_low(P):
        return np.nextafter(jsd_matrix(P, backend="numpy"), 0.0)

    def scipy(P):
        raise AssertionError("SciPy is not held to the 1.0 guarantee")

    r = portable.disjoint_exactness([("fastjsd numpy", numpy_backend),
                                     ("fastjsd low", one_ulp_low),
                                     ("scipy pdist", scipy)])
    assert r["trials"] == 40
    assert r["exact_one"] == {"fastjsd numpy": True, "fastjsd low": False}
    assert r["worst_miss"]["fastjsd low"] > 0.0


def test_f32_section_without_the_kernel_is_absent_not_fatal():
    P = np.random.default_rng(0).dirichlet(np.full(8, 0.5), size=10)
    out = portable.f32_section([("dense", P)], SimpleNamespace(reps=1, repeats=1),
                               None, "numba-f32 needs numba")
    assert (out["status"], out["reason"], out["cases"]) == (
        "absent", "numba-f32 needs numba", [])


@needs_f32
def test_f32_case_verifies_then_times_on_the_same_rows():
    P = np.random.default_rng(1).dirichlet(np.full(16, 0.5), size=48)
    row = portable.f32_case("dense", P, SimpleNamespace(reps=1, repeats=2))
    assert row["status"] == "ok"
    assert row["agrees"] and row["max_abs_err"] <= portable.F32_TOL
    base = row["impls"]["numba"]
    f32 = row["impls"]["numba-f32"]
    assert len(base["ms"]) == len(f32["ms"]) == 2
    r = row["ratio"]
    # The bracket the aggregator checks, with its slack for the rounded ms.
    assert r["min"] * (1 - 1e-3) <= base["ms_min"] / f32["ms_min"] \
        <= r["max"] * (1 + 1e-3)


@needs_f32
def test_microsecond_timings_keep_the_bracket(monkeypatch):
    """Times from a run of the test above, in seconds: rounded to 0.1 us, the
    ratio of their minima fell 0.14 % below the ratio block, outside the
    aggregator's 0.1 % slack, and the test failed in about one run in 20."""
    times = iter([[1.87611e-4, 7.24163e-5], [8.38432e-5, 4.96583e-5]])
    monkeypatch.setattr(portable, "interleaved", lambda fns, X, reps: next(times))
    P = np.random.default_rng(1).dirichlet(np.full(16, 0.5), size=48)
    row = portable.f32_case("dense", P, SimpleNamespace(reps=1, repeats=2))
    assert row["status"] == "ok"
    assert aggregate.crosscheck({"leg_id": "t", "f32": {"cases": [row]}}) == []


@needs_f32
def test_f32_section_skips_what_auto_sends_to_a_sparse_kernel():
    rng = np.random.default_rng(2)
    dense = rng.dirichlet(np.full(16, 0.5), size=40)
    sparse = np.eye(40)
    out = portable.f32_section([("dense", dense), ("sparse", sparse)],
                               SimpleNamespace(reps=1, repeats=1),
                               portable._f32_matrix, None)
    assert out["status"] == "ok"
    assert [row["case"] for row in out["cases"]] == ["dense"]


@needs_f32
def test_f32_accuracy_is_against_its_float32_input():
    pytest.importorskip("mpmath")
    acc = portable.accuracy_vs_mpmath(n=6, k=8)
    near = acc["per_regime"]["near-duplicate"]["numba-f32"]
    assert near["reference"] == "its float32-rounded input"
    # Rows 1e-9 apart are (all but) the same row in float32. Against the
    # float64 input that reads as a 100 % error; against what the kernel was
    # given it is no error at all -- or, where one element of a synthetic row
    # rounds the other way on this platform, a tiny one.
    assert near["abs"] <= portable.F32_TOL and near["rel"] <= 1e-3
    assert acc["per_regime"]["random"]["numba-f32"]["abs"] <= portable.F32_TOL
    skipped = portable.accuracy_vs_mpmath(n=6, k=8, f32_ok=False)
    assert all("numba-f32" not in per
               for per in skipped["per_regime"].values())


@needs_f32
def test_leg_round_trips_through_the_aggregator(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(3)
    dense = rng.dirichlet(np.full(12, 0.5), size=40)
    sparse = dense * (rng.random(dense.shape) < 0.2)
    sparse[:, 0] += 0.1
    monkeypatch.setattr(portable, "portable_cases", lambda: [
        ("tiny dense n=40 k=12", dense), ("tiny sparse n=40 k=12", sparse)])
    # The machine probe shells out per OS and is not what this tests.
    monkeypatch.setattr(portable, "machine", lambda leg_id: {
        "leg_id": leg_id, "cpu_usable": 2, "cpu_model": "test cpu"})
    path = tmp_path / "portable-t.json"
    monkeypatch.setattr(sys, "argv", [
        "portable.py", "--leg-id", "t", "--json", str(path),
        "--reps", "1", "--repeats", "2"])
    portable.main()
    printed = capsys.readouterr().out
    assert "numba-f32 against numba, dense cases" in printed

    doc = json.loads(path.read_text())
    assert doc["schema"] == portable.SCHEMA == aggregate.SCHEMA
    assert [row["case"] for row in doc["f32"]["cases"]] == [
        "tiny dense n=40 k=12"]
    assert doc["f32"]["cases"][0]["agrees"]
    acc = doc["accuracy"]
    assert acc["duplicate_rows_exact_zero"]["fastjsd numba-f32"]
    assert acc["disjoint_supports_exact_one"] == {
        "fastjsd auto": True, "fastjsd numpy": True, "fastjsd numba-f32": True}

    legs, problems = aggregate.load([str(path)])
    assert len(legs) == 1 and not problems
    # Two repeats of one rep on a CI runner can grade themselves unusable; the
    # verdict rule is not under test here, the document's shape is.
    legs[0]["validity"]["verdict"] = "ok"
    s = aggregate.build(legs)
    # Only what this change is responsible for: a third-party baseline that
    # disagrees with SciPy on some platform is a fact the report records.
    assert not [p for p in s["problems"]
                if "numba-f32" in p or "does not bracket" in p]
    assert not s["exactness_failures"] and not s["disjoint_failures"]
    assert s["f32"]["n_boxes"] == 1
    assert "numba-f32 vs numba   (dense cases" in _text(s)
