"""Combine cross-machine legs into a claim, and make the disagreement loud.

    python bench/aggregate.py 'artifacts/bench-*/portable-*.json'
    python bench/aggregate.py leg1.json leg2.json --json summary.json
    python bench/aggregate.py 'legs/portable-*.json' --markdown \
        >> "$GITHUB_STEP_SUMMARY"

Match `portable-*.json`, not `*.json`. A leg directory also holds the
`fingerprint-*.json` a leg writes before it times anything, and the
`summary.json` this file writes -- neither is a leg, and `load()` rejects both
by shape rather than trusting the caller's glob.

Input is one `bench/portable.py` document per box. The job here is arithmetic
plus refusal, and the refusals are the point:

Never average milliseconds.
    A time from one runner divided by a time from another is meaningless -- the
    clock rate, core count and neighbours all differ. Only ratios cross a
    machine boundary, because each was measured between two implementations in
    the same interleaved round-robin on the same box. This module therefore
    reads `ratios` and treats `ms` as evidence, not as data.

The box is the block, not the case.
    Five boxes times seven cases is not thirty-five independent observations:
    cases within a box share the hardware. So the headline count is over BOXES,
    the per-case spread is reported as detail, and the per-box summary is the
    GEOMETRIC mean over cases -- ratios compose multiplicatively and the
    arithmetic mean of a ratio set is simply the wrong number.

A loss is the headline, never a footnote.
    If any box shows fastjsd slower than a baseline, that is printed first, in
    its own block, before any range or geomean. A summary that reports a wide
    speedup range while one of its cells sits below 1.0 has told the reader the
    opposite of what happened. The same applies to cells whose across-repeat
    spread straddles 1.0: those are `indistinguishable`, and folding them into
    a range asserts a ratio from a box that cannot resolve it.

Configurations are partitioned before anything is combined.
    A leg without numba runs the NumPy backends, which measure the
    support-intersection identity rather than the kernel, and on dense inputs
    can be slower than SciPy. That is an honest finding and it gets its own
    section; pooling it into the headline range would destroy both claims.

numba-f32 gets a section of its own.
    Its ratio is numba / numba-f32 on the same float32 rows -- a claim about
    the float32 kernel, not about fastjsd against a baseline -- so it is
    counted and printed apart, under the same rules: the box as the unit, the
    geometric mean, and any loss in the headline.

Boxes that graded themselves `unusable` are dropped, and the exclusion is
printed. A silently dropped box is worse than a bad one.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import sys

SCHEMA = "fastjsd-crossmachine/2"

#: The summary this file writes carries its OWN schema, so that re-running the
#: aggregation over a directory it has already written to cannot read its own
#: output back in as though it were a leg -- which is what sharing `SCHEMA`
#: between the two would allow.
SUMMARY_SCHEMA = "fastjsd-crossmachine-summary/2"

#: The subject. Every ratio in a leg is <baseline> / <this>.
SUBJECT = "fastjsd auto"

#: The leg's `f32` section: numba-f32 timed against the float64 dense kernel on
#: the same float32 rows. Same orientation, <baseline> / <subject>, and a claim
#: of its own, so it is combined and reported in a section of its own.
F32_SUBJECT = "numba-f32"
F32_BASELINE = "numba"


def geomean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def expand(patterns):
    """Expand globs here rather than trusting the shell -- one leg of this
    matrix runs on Windows, where the shell does not do it."""
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            out.extend(hits)
        elif pathlib.Path(pat).exists():
            out.append(pat)
        else:
            out.append(pat)                  # let load() report it as missing
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def load(paths):
    """Read the legs, refusing anything this aggregator cannot read correctly.

    A stale harness whose schema has moved must be REJECTED, not silently
    misread into a plausible-looking table -- which is why `schema` is the
    first key a leg writes.

    Matching the schema string is not enough, because `portable.py
    --fingerprint-only` writes the same string on a document that has no
    `cases` at all. Accepted as a leg, such a document contributes zero cells
    while still appearing in the boxes table and counting toward the
    per-configuration box total, so a section headed "3 boxes" would carry a
    verdict counted over 2. A leg is therefore identified by its SHAPE -- a
    `leg_id` and a `cases` list -- and anything else is named and skipped.
    """
    legs, problems = [], []
    for path in paths:
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except Exception as exc:
            problems.append(f"{path}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{path}: top level is not an object; not a leg")
            continue
        if doc.get("schema") == SUMMARY_SCHEMA:
            problems.append(
                f"{path}: this is an aggregate.py SUMMARY, not a leg; skipped "
                f"(match 'portable-*.json' rather than '*.json')")
            continue
        if doc.get("schema") != SCHEMA:
            problems.append(
                f"{path}: schema {doc.get('schema')!r} != {SCHEMA!r}; refusing "
                f"to guess at its layout")
            continue
        missing = [k for k in ("leg_id", "cases") if k not in doc]
        if missing:
            what = ("a --fingerprint-only document" if "cases" in missing
                    and "machine" in doc else "not a leg")
            problems.append(
                f"{path}: carries the leg schema but has no "
                f"{', '.join(missing)} -- {what}; skipped")
            continue
        if not isinstance(doc["cases"], list):
            problems.append(f"{path}: 'cases' is not a list; skipped")
            continue
        doc["_path"] = path
        legs.append(doc)
    return legs, problems


def crosscheck(leg):
    """Recompute what can be recomputed, and complain when it disagrees.

    The leg computes its own ratios because only it knows which measurements
    shared a round-robin. That is right, and it also means nothing checks them.
    One bound does hold unconditionally: with `b_t`, `f_t` the per-repeat
    minima, ``min_t b_t / min_t f_t`` always lies inside
    ``[min_t b_t/f_t, max_t b_t/f_t]``. If it does not, the `ms` arrays and the
    `ratios` block did not come from the same measurements.

    The bracket is checked with a relative slack, because the leg rounds the
    times before writing them (to the nanosecond, `portable.MS_DIGITS`), and a
    bound that fired on the last decimal place would flag honest files.
    """
    notes = []
    slack = 1e-3
    for row in leg.get("cases", []):
        impls = row.get("impls", {})
        for name, rec in impls.items():
            if rec.get("status") == "ok" and rec.get("agrees") is False:
                notes.append(
                    f"{leg['leg_id']}/{row.get('case')}: {name} disagreed with "
                    f"the reference ({rec.get('max_abs_err')}); its timing "
                    f"measures a different function")
        me = impls.get(SUBJECT)
        if not me or me.get("status") != "ok":
            notes.append(f"{leg['leg_id']}/{row.get('case')}: no {SUBJECT}")
            continue
        for name, ratio in row.get("ratios", {}).items():
            base = impls.get(name)
            if not base or base.get("status") != "ok":
                notes.append(f"{leg['leg_id']}/{row['case']}: ratio for "
                             f"{name} with no timing")
                continue
            got = base["ms_min"] / me["ms_min"]
            lo, hi = ratio["min"], ratio["max"]
            if not (lo * (1 - slack) <= got <= hi * (1 + slack)):
                notes.append(
                    f"{leg['leg_id']}/{row['case']}: {name} ratio block "
                    f"[{lo:.3f}, {hi:.3f}] does not bracket ms_min ratio "
                    f"{got:.3f} -- arrays and ratios disagree")
    for row in (leg.get("f32") or {}).get("cases", []):
        where = f"{leg['leg_id']}/{row.get('case')}"
        if row.get("status") != "ok":
            notes.append(f"{where}: {F32_SUBJECT} did not run "
                         f"({row.get('reason')})")
            continue
        if row.get("agrees") is False:
            notes.append(f"{where}: {F32_SUBJECT} disagreed with the float64 "
                         f"kernel ({row.get('max_abs_err')}); its timing "
                         f"measures a different function")
        impls, ratio = row.get("impls", {}), row.get("ratio") or {}
        try:
            got = (impls[F32_BASELINE]["ms_min"]
                   / impls[F32_SUBJECT]["ms_min"])
            lo, hi = ratio["min"], ratio["max"]
        except (KeyError, TypeError, ZeroDivisionError):
            notes.append(f"{where}: {F32_SUBJECT} row has no usable timing")
            continue
        if not (lo * (1 - slack) <= got <= hi * (1 + slack)):
            notes.append(
                f"{where}: {F32_SUBJECT} ratio block [{lo:.3f}, {hi:.3f}] does "
                f"not bracket ms_min ratio {got:.3f} -- arrays and ratios "
                f"disagree")
    return notes


def has_numba(leg):
    return bool(leg.get("software", {}).get("numba"))


# ---------------------------------------------------------------------------
# combining
# ---------------------------------------------------------------------------

def cells(legs):
    """One record per (box, case, baseline) -- the unit of comparison."""
    out = []
    for leg in legs:
        for row in leg.get("cases", []):
            for name, r in row.get("ratios", {}).items():
                out.append({
                    "leg": leg["leg_id"], "case": row["case"],
                    "baseline": name, "median": r["median"],
                    "min": r["min"], "max": r["max"],
                    "resolvable": bool(r.get("resolvable", False)),
                    "n": row.get("n"), "k": row.get("k"),
                    "density": row.get("density"),
                })
    return out


def f32_cells(legs):
    """One record per (box, case) of the `f32` section, shaped like `cells`.

    Shaped alike so `per_case`, `per_box` and `verdict` apply unchanged: the
    rules that keep the main comparison honest -- the box as the unit, the
    geometric mean, a loss as the headline -- hold for this claim too.
    """
    out = []
    for leg in legs:
        for row in (leg.get("f32") or {}).get("cases", []):
            r = row.get("ratio")
            if row.get("status") != "ok" or not r:
                continue
            out.append({
                "leg": leg["leg_id"], "case": row["case"],
                "baseline": F32_BASELINE, "median": r["median"],
                "min": r["min"], "max": r["max"],
                "resolvable": bool(r.get("resolvable", False)),
                "n": row.get("n"), "k": row.get("k"), "density": None,
            })
    return out


def baselines_in(cs):
    seen = []
    for c in cs:
        if c["baseline"] not in seen:
            seen.append(c["baseline"])
    return seen


def per_case(cs, baseline):
    """Per case: the range across boxes, and WHICH box produced the worst."""
    by_case = {}
    for c in cs:
        if c["baseline"] != baseline:
            continue
        by_case.setdefault(c["case"], []).append(c)
    out = []
    for case, group in by_case.items():
        vals = [g["median"] for g in group]
        worst = min(group, key=lambda g: g["median"])
        best = max(group, key=lambda g: g["median"])
        out.append({
            "case": case, "n_boxes": len(group),
            "min": worst["median"], "min_box": worst["leg"],
            "median": median(vals), "max": best["median"],
            "max_box": best["leg"],
            "n_resolvable": sum(1 for g in group if g["resolvable"]),
            "losing_boxes": [g["leg"] for g in group if g["median"] < 1.0],
        })
    out.sort(key=lambda r: r["min"])
    return out


def per_box(cs, baseline):
    """One number per box: the geometric mean of its per-case ratios."""
    by_leg = {}
    for c in cs:
        if c["baseline"] == baseline:
            by_leg.setdefault(c["leg"], []).append(c["median"])
    return {leg: geomean(v) for leg, v in by_leg.items()}


def verdict(cs, baseline):
    group = [c for c in cs if c["baseline"] == baseline]
    if not group:
        return None
    boxes = per_box(cs, baseline)
    won_boxes = [leg for leg, g in boxes.items() if g is not None and g > 1.0]
    lost_cells = [c for c in group if c["median"] < 1.0]
    unresolved = [c for c in group if not c["resolvable"]]
    vals = [c["median"] for c in group]
    return {
        "baseline": baseline,
        "n_boxes": len(boxes), "n_cells": len(group),
        "boxes_won": len(won_boxes), "boxes_lost": len(boxes) - len(won_boxes),
        "cells_won": sum(1 for c in group if c["median"] > 1.0),
        "cells_lost": len(lost_cells),
        "cells_resolvable": len(group) - len(unresolved),
        "range": [min(vals), max(vals)],
        "per_box_geomean": boxes,
        "median_box_geomean": median(list(boxes.values())),
        "losses": [{"leg": c["leg"], "case": c["case"], "ratio": c["median"]}
                   for c in lost_cells],
        "indistinguishable": [{"leg": c["leg"], "case": c["case"],
                               "min": c["min"], "max": c["max"]}
                              for c in unresolved],
    }


def sentence(v, subject="fastjsd"):
    """The one line a reader is most likely to quote."""
    if v is None:
        return ""
    name = v["baseline"]
    if v["cells_lost"]:
        head = (f"{subject} was SLOWER than {name} in {v['cells_lost']} of "
                f"{v['n_cells']} case-box cells "
                f"({v['boxes_won']}/{v['n_boxes']} boxes still had a per-box "
                f"geomean above 1)")
    else:
        head = (f"{subject} was faster than {name} on {v['boxes_won']}/"
                f"{v['n_boxes']} boxes and {v['cells_won']}/{v['n_cells']} "
                f"case-box pairs")
    lo, hi = v["range"]
    gm = v["median_box_geomean"]
    tail = (f"; ratio range {lo:.2f}x-{hi:.2f}x"
            + (f" (median per-box geomean {gm:.2f}x)" if gm else ""))
    if v["cells_resolvable"] < v["n_cells"]:
        n_ind = v["n_cells"] - v["cells_resolvable"]
        tail += (f"; {v['cells_resolvable']}/{v['n_cells']} cells resolvable "
                 f"above their own repeat noise, {n_ind} indistinguishable")
    return head + tail + "."


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def bar(text, char="="):
    return char * max(72, len(text) + 4)


def plural(count, noun):
    if count == 1:
        return f"{count} {noun}"
    suffix = "es" if noun.endswith(("x", "s", "ch", "sh")) else "s"
    return f"{count} {noun}{suffix}"


def report_losses(summary, emit):
    """The headline, or the absence of one. Printed FIRST, always."""
    losses, unresolved = [], []
    for section in summary["configurations"].values():
        for v in section["verdicts"].values():
            losses.extend({**loss, "baseline": v["baseline"]}
                          for loss in v["losses"])
            unresolved.extend({**u, "baseline": v["baseline"]}
                              for u in v["indistinguishable"])
    if losses:
        title = f"LOSSES: fastjsd was SLOWER in {plural(len(losses), 'case-box cell')}"
        emit(bar(title))
        emit(title)
        emit(bar(title))
        emit("")
        emit(f"{'box':16}{'baseline':24}{'ratio':>8}  case")
        for loss in sorted(losses, key=lambda x: x["ratio"]):
            emit(f"{loss['leg']:16}{loss['baseline']:24}"
                 f"{loss['ratio']:>7.2f}x  {loss['case']}")
        emit("")
        emit("This is the headline. Any range quoted below is a range that")
        emit("CONTAINS these cells; it does not excuse them. A speedup claim")
        emit("that does not mention them is not a claim about this hardware.")
        emit("")
    else:
        emit("no losing cells: fastjsd was at least as fast as every baseline")
        emit("on every box and every case in this set.")
        emit("")
    f32 = summary.get("f32")
    f32_losses = f32["verdict"]["losses"] if f32 else []
    if f32_losses:
        title = (f"LOSSES: {F32_SUBJECT} was SLOWER than {F32_BASELINE} in "
                 f"{plural(len(f32_losses), 'case-box cell')}")
        emit(bar(title))
        emit(title)
        emit(bar(title))
        emit("")
        emit(f"{'box':16}{'ratio':>8}  case")
        for loss in sorted(f32_losses, key=lambda x: x["ratio"]):
            emit(f"{loss['leg']:16}{loss['ratio']:>7.2f}x  {loss['case']}")
        emit("")
        emit(f"The 0.2.0 claim is that {F32_SUBJECT} is faster than "
             f"{F32_BASELINE} on dense")
        emit("float32 input. On these boxes it was not, and the range in its")
        emit("section below contains these cells.")
        emit("")
    if unresolved:
        emit(f"INDISTINGUISHABLE: {plural(len(unresolved), 'cell')} whose "
             "across-repeat ratio")
        emit("spread straddles 1.0, so that box cannot tell the two apart:")
        for u in unresolved:
            emit(f"  {u['leg']:16}{u['baseline']:24}"
                 f"[{u['min']:.2f}x, {u['max']:.2f}x]  {u['case']}")
        emit("")


def report_f32(summary, emit):
    """numba-f32 against numba: the 0.2.0 claim, in its own section."""
    f32 = summary.get("f32")
    title = (f"{F32_SUBJECT} vs {F32_BASELINE}   (dense cases, float32 input "
             f"for both)")
    emit(bar(title, "="))
    if not f32:
        emit(title)
        emit(bar(title, "="))
        emit("no leg produced a comparable cell: numba-f32 was absent or failed")
        emit("on every box (each leg's 'f32' block says why).")
        emit("")
        return
    emit(f"{title}   ({plural(f32['n_boxes'], 'box')})")
    emit(f"ratio = {F32_BASELINE} / {F32_SUBJECT}: above 1.0, the float32 "
         "kernel was faster on that box.")
    emit(bar(title, "="))
    emit("")
    emit(f"{'case':34}{'min':>8}{'worst box':>16}"
         f"{'median':>9}{'max':>9}{'boxes':>7}{'resolv':>8}")
    emit("-" * 91)
    for row in f32["per_case"]:
        emit(f"{row['case'][:33]:34}{row['min']:>7.2f}x"
             f"{row['min_box']:>16}{row['median']:>8.2f}x"
             f"{row['max']:>8.2f}x{row['n_boxes']:>7}"
             f"{row['n_resolvable']:>5}/{row['n_boxes']}")
    v = f32["verdict"]
    emit("")
    emit("per-box geometric mean over cases:")
    for leg, gm in sorted(v["per_box_geomean"].items(),
                          key=lambda kv: kv[1] or 0.0):
        emit(f"  {leg:16}{gm:>8.2f}x" if gm else f"  {leg:16}       -")
    if v["indistinguishable"]:
        emit("")
        emit("indistinguishable cells (the across-repeat spread straddles 1.0):")
        for u in v["indistinguishable"]:
            emit(f"  {u['leg']:16}[{u['min']:.2f}x, {u['max']:.2f}x]  "
                 f"{u['case']}")
    emit("")
    emit("  " + sentence(v, subject=F32_SUBJECT))
    acc = summary.get("f32_accuracy") or {}
    if acc:
        emit("")
        emit(f"{F32_SUBJECT} against 50-digit truth of its float32-rounded "
             "input, worst over boxes")
        emit("(deterministic -- the caveats on timing do not apply):")
        for regime, a in acc.items():
            where = f" ({a['worst_abs_box']})" if a["worst_abs"] > 0 else ""
            emit(f"  {regime:18}abs {a['worst_abs']:.1e}{where}"
                 f"   rel {a['worst_rel']:.1e}   {plural(a['n_boxes'], 'box')}")
    emit("")


def report(summary, emit):
    report_losses(summary, emit)

    if summary["excluded"]:
        emit("EXCLUDED boxes (self-graded unusable; reported, not hidden):")
        for leg, reasons in summary["excluded"].items():
            emit(f"  {leg}: {'; '.join(reasons) or 'no reason recorded'}")
        emit("")
    if summary["suspect"]:
        emit("SUSPECT boxes (kept, but their numbers carry a warning):")
        for leg, reasons in summary["suspect"].items():
            emit(f"  {leg}: {'; '.join(reasons) or 'no reason recorded'}")
        emit("")
    if summary["problems"]:
        emit("PROBLEMS in the leg files (read these before the tables):")
        for note in summary["problems"]:
            emit(f"  ! {note}")
        emit("")

    emit("boxes")
    emit(f"{'leg':16}{'config':12}{'cores':>6}{'validity':>10}  cpu")
    emit("-" * 88)
    for box in summary["boxes"]:
        emit(f"{box['leg_id']:16}{box['config']:12}{str(box['cores']):>6}"
             f"{box['validity']:>10}  {(box['cpu'] or '?')[:38]}")
    emit("")

    for config, section in summary["configurations"].items():
        if not section["verdicts"]:
            continue
        emit(bar(config, "="))
        head = f"configuration: {config}   ({plural(section['n_boxes'], 'box')})"
        read = section.get("n_legs_read", section["n_boxes"])
        if read != section["n_boxes"]:
            head += f"   [{read} legs read, {read - section['n_boxes']} with no cell]"
        emit(head)
        if config != "numba":
            emit("These legs ran the NumPy backends only. They measure the")
            emit("support-intersection identity, not the kernel, and are NOT")
            emit("part of the headline range.")
        emit(bar(config, "="))
        for name, v in section["verdicts"].items():
            emit("")
            emit(f"vs {name}")
            emit(f"{'case':34}{'min':>8}{'worst box':>16}"
                 f"{'median':>9}{'max':>9}{'boxes':>7}{'resolv':>8}")
            emit("-" * 91)
            for row in section["per_case"][name]:
                emit(f"{row['case'][:33]:34}{row['min']:>7.2f}x"
                     f"{row['min_box']:>16}{row['median']:>8.2f}x"
                     f"{row['max']:>8.2f}x{row['n_boxes']:>7}"
                     f"{row['n_resolvable']:>5}/{row['n_boxes']}")
            emit("")
            emit("per-box geometric mean over cases:")
            for leg, gm in sorted(v["per_box_geomean"].items(),
                                  key=lambda kv: kv[1] or 0.0):
                emit(f"  {leg:16}{gm:>8.2f}x" if gm else f"  {leg:16}       -")
            emit("")
            emit("  " + sentence(v))
        emit("")

    report_f32(summary, emit)

    emit("consistency verdict")
    emit("-" * 88)
    for line in summary["verdict_lines"]:
        emit("  " + line)
    emit("")
    emit("The unit of independence is the BOX, not the case: cases within a box")
    emit("share hardware, so N boxes x M cases is not N*M observations. Counts")
    emit("over boxes are the claim; counts over cells are detail.")
    emit("")
    if summary["input_hash_mismatch"]:
        emit("INPUT HASH MISMATCH (warning, not a rejection): the synthetic")
        emit("cases are drawn through libm log/exp/pow, so they can differ in")
        emit("the last ulp on another platform. Timings stay comparable; the")
        emit("PAIRED accuracy comparison for these cases does not.")
        for case, hashes in summary["input_hash_mismatch"].items():
            emit(f"  {case}: {len(hashes)} distinct inputs across boxes")
        emit("")
    if summary["exactness_failures"]:
        emit("EXACTNESS FAILURE -- identical rows did not come back exactly 0.0:")
        for leg, impls in summary["exactness_failures"].items():
            emit(f"  {leg}: {', '.join(impls)}")
        emit("  This is deterministic and does NOT depend on runner noise. It is")
        emit("  a real portability defect in the guarantee, not a measurement.")
        emit("")
    if summary["disjoint_failures"]:
        emit("EXACTNESS FAILURE -- disjoint supports did not come back exactly "
             "1.0:")
        for leg, impls in summary["disjoint_failures"].items():
            emit(f"  {leg}: {', '.join(impls)}")
        emit("  Deterministic, like the zero check: a real portability defect in")
        emit("  the guarantee (docs/MATH.md G6), not a measurement.")
        emit("")
    emit("caveats, carried from the legs and reprinted with the numbers:")
    for c in summary["caveats"]:
        emit(f"  - {c}")


def build(legs):
    problems = []
    for leg in legs:
        problems.extend(crosscheck(leg))

    excluded, suspect, kept = {}, {}, []
    for leg in legs:
        v = leg.get("validity", {})
        if v.get("verdict") == "unusable":
            excluded[leg["leg_id"]] = v.get("reasons", [])
            continue
        if v.get("verdict") == "suspect":
            suspect[leg["leg_id"]] = v.get("reasons", [])
        kept.append(leg)

    boxes = []
    for leg in legs:
        m = leg.get("machine", {})
        boxes.append({
            "leg_id": leg["leg_id"],
            "config": "numba" if has_numba(leg) else "numpy-only",
            "cores": m.get("cpu_usable"),
            "cpu": m.get("cpu_model"),
            "validity": leg.get("validity", {}).get("verdict", "?"),
            "path": leg.get("_path"),
        })

    configurations = {}
    for config, group in (("numba", [x for x in kept if has_numba(x)]),
                          ("numpy-only", [x for x in kept if not has_numba(x)])):
        cs = cells(group)
        verdicts, cases_by_base = {}, {}
        for name in baselines_in(cs):
            verdicts[name] = verdict(cs, name)
            cases_by_base[name] = per_case(cs, name)
        # Boxes that actually produced a comparable cell, not boxes that were
        # merely read. A leg whose every case failed contributes nothing to a
        # verdict and must not be counted in the header above it.
        contributing = {c["leg"] for c in cs}
        configurations[config] = {"n_boxes": len(contributing),
                                  "n_legs_read": len(group),
                                  "verdicts": verdicts,
                                  "per_case": cases_by_base}

    # Input identity: same case name, different bytes, on different boxes.
    hashes = {}
    for leg in legs:
        for row in leg.get("cases", []):
            hashes.setdefault(row["case"], set()).add(row.get("input_sha256"))
    mismatch = {c: sorted(h) for c, h in hashes.items() if len(h) > 1}

    # Deterministic checks are read from EVERY leg, excluded ones included:
    # runner noise cannot touch them, so a box too noisy to time is still a
    # box whose exactness result stands.
    failures, disjoint = {}, {}
    for leg in legs:
        acc = leg.get("accuracy", {})
        for key, out in (("duplicate_rows_exact_zero", failures),
                         ("disjoint_supports_exact_one", disjoint)):
            bad = [nm for nm, ok in acc.get(key, {}).items()
                   if nm.startswith("fastjsd") and not ok]
            if bad:
                out[leg["leg_id"]] = bad

    # The f32 claim is a numba claim: a leg without numba has no f32 section.
    fc = f32_cells([x for x in kept if has_numba(x)])
    f32 = None
    if fc:
        f32 = {"n_boxes": len({c["leg"] for c in fc}),
               "verdict": verdict(fc, F32_BASELINE),
               "per_case": per_case(fc, F32_BASELINE)}

    f32_acc = {}
    for leg in legs:
        per = (leg.get("accuracy", {}).get("vs_mpmath") or {}).get(
            "per_regime", {})
        for regime, backends in per.items():
            e = backends.get(F32_SUBJECT)
            if not e:
                continue
            a = f32_acc.setdefault(regime, {"worst_abs": -1.0,
                                            "worst_abs_box": None,
                                            "worst_rel": 0.0, "n_boxes": 0})
            a["n_boxes"] += 1
            a["worst_rel"] = max(a["worst_rel"], e["rel"])
            if e["abs"] > a["worst_abs"]:
                a["worst_abs"], a["worst_abs_box"] = e["abs"], leg["leg_id"]

    lines = []
    for config, section in configurations.items():
        for v in section["verdicts"].values():
            prefix = "" if config == "numba" else "[numpy-only] "
            lines.append(prefix + sentence(v))
    if f32:
        lines.append(f"[{F32_SUBJECT}] "
                     + sentence(f32["verdict"], subject=F32_SUBJECT))
    if not lines:
        lines.append("no comparable cells: every leg is missing ratios.")

    caveats = []
    for leg in legs:
        for c in leg.get("caveats", []):
            if c not in caveats:
                caveats.append(c)

    return {"schema": SUMMARY_SCHEMA, "leg_schema": SCHEMA,
            "n_legs": len(legs), "boxes": boxes,
            "excluded": excluded, "suspect": suspect, "problems": problems,
            "configurations": configurations, "verdict_lines": lines,
            "input_hash_mismatch": mismatch, "exactness_failures": failures,
            "disjoint_failures": disjoint, "f32": f32,
            "f32_accuracy": f32_acc, "caveats": caveats}


def main():
    ap = argparse.ArgumentParser(
        description="combine bench/portable.py legs into a cross-machine claim")
    ap.add_argument("paths", nargs="+",
                    help="leg JSON files, or globs matching them")
    ap.add_argument("--json", type=str, default=None,
                    help="write the aggregated summary here")
    ap.add_argument("--markdown", action="store_true",
                    help="wrap the report in a fenced block for "
                         "$GITHUB_STEP_SUMMARY")
    args = ap.parse_args()

    legs, problems = load(expand(args.paths))
    if not legs:
        for note in problems:
            print(f"  ! {note}", file=sys.stderr)
        print("no readable legs; nothing to aggregate", file=sys.stderr)
        return 2

    summary = build(legs)
    summary["problems"] = problems + summary["problems"]

    out = []
    report(summary, out.append)
    body = "\n".join(out)
    print(f"```\n{body}\n```" if args.markdown else body)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    # A losing cell is a result, not an error: exit 0 and let the reader read.
    return 0


if __name__ == "__main__":
    sys.exit(main())
