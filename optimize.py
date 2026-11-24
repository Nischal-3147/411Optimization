"""
optimize.py — Optimizers and report outputs for Smart Insole Sensor Placement
(BME 411, Group 9). Follows analysis_plan.md steps 6–11.

Problem (project proposal):
    minimize   f(X) = W1 · E(X) + W2 · N
    subject to r <= x_i <= w - r,  r <= y_i <= l - r         i = 1..N
               (2r)^2 - [(x_a - x_b)^2 + (y_a - y_b)^2] < 0  1 <= a < b <= N
               N ∈ Z, 2 <= N <= 16
    where E(X) is the 4-component gait-timing RMSE (HS, TO, step, stride) in ms.

Constraints are HARD: no penalty terms are ever added to E.
  * MOPSO / PSO: positions are clamped to the box after every move; overlapping
    sensors are pushed apart (repair). If repair fails, the particle keeps its
    previous feasible position. Infeasible layouts are never scored or stored.
  * SLSQP: box bounds + analytic non-overlap inequality constraints.
The strict "<" in the overlap constraint is enforced with a small clearance:
centre distance >= 2r + MIN_GAP_CM.

Two gait-event detection methods are optimized side by side (data.py):
  * "sum"        — sum all N sensor pressures, threshold the total (original plan).
  * "first_last" — heel-strike = first sensor to load, toe-off = last to unload.

Run:
    python optimize.py                          # full run, both methods (~25–30 min, 10 cores)
    python optimize.py --quick                  # smoke test (~2 min)
    python optimize.py --aggregation first_last # one method only

Outputs:
    results/sum/  and  results/first_last/  each contain
        pareto_front.png, optimal_layouts.png, convergence_comparison.png,
        cross_cadence_validation.csv, plus supporting CSV/JSON tables.
    results/  method_comparison.png and combined *_all_methods.csv tables.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import data as D  # noqa: E402

# =============================================================================
# 1. Settings
# =============================================================================
RESULTS_DIR: Path = D.BASE_DIR / "results"
TRAIN_PACE: str = "NP"                 # Optimize on Normal Pace only
TEST_PACES: Tuple[str, ...] = ("NP", "SP", "FP")
N_RANGE: Tuple[int, ...] = tuple(range(D.N_MIN, D.N_MAX + 1))
MIN_GAP_CM: float = 1e-3               # Strict non-overlap clearance (cm)
E_REF_MS: float = 1000.0 * D.MISS_PENALTY_SEC   # Worst possible E (hypervolume ref)

# PSO coefficients (shared by MOPSO and the single-objective PSO sweep)
W_START, W_END = 0.9, 0.4              # Linearly decreasing inertia
C1, C2 = 1.5, 1.5                      # Cognitive / social
VMAX_FRAC: float = 0.2                 # Velocity clamp, fraction of variable range

# Layouts drawn in optimal_layouts.png (analysis_plan.md step 11)
PLOT_NS: Tuple[int, ...] = (3, 4, 5, 8)
# SLSQP comparison sensor counts (analysis_plan.md step 8)
SLSQP_NS: Tuple[int, ...] = (4, 6)
# Weight sweep: W1 = 1 per ms, W2 = "ms of error one extra sensor is worth"
W2_VALUES: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

METHOD_LABELS = {
    "sum": "Summed signal",
    "first_last": "First/last sensor",
}

# Colors (validated reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29d"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

PRESETS = {
    "full": dict(seeds=3, particles=100, iters=150,
                 sweep_particles=30, sweep_iters=60,
                 slsqp_starts=10, slsqp_maxiter=100),
    "quick": dict(seeds=1, particles=30, iters=15,
                  sweep_particles=10, sweep_iters=8,
                  slsqp_starts=3, slsqp_maxiter=15),
}


# =============================================================================
# 2. Parallel objective evaluation (one dataset copy per worker process)
# =============================================================================
_DS: Optional[D.InsoleDataset] = None
_TRAIN: Optional[List[D.TrialData]] = None


def _init_worker() -> None:
    global _DS, _TRAIN
    _DS = D.load_cached_dataset()
    _TRAIN = _DS.get_trials_by_pace(TRAIN_PACE)


def _eval_E(job: Tuple[str, np.ndarray]) -> float:
    """E(X) in ms, mean over training trials. No penalty terms. job = (aggregation, xy_flat)."""
    aggregation, xy_flat = job
    xy = np.asarray(xy_flat, dtype=np.float64).reshape(-1, 2)
    return D.score_layout(xy, _DS, _TRAIN, include_overlap_penalty=False,
                          aggregation=aggregation)


class Evaluator:
    """Scores batches of layouts in parallel and counts function evaluations."""

    def __init__(self, pool: Pool, workers: int, aggregation: str):
        self.pool = pool
        self.workers = workers
        self.aggregation = aggregation
        self.nfev = 0

    def __call__(self, layouts: Sequence[np.ndarray]) -> np.ndarray:
        self.nfev += len(layouts)
        chunk = max(1, len(layouts) // (4 * self.workers))
        jobs = [(self.aggregation, np.asarray(l, dtype=np.float64).ravel()) for l in layouts]
        return np.asarray(self.pool.map(_eval_E, jobs, chunksize=chunk), dtype=np.float64)


# =============================================================================
# 3. Geometry: hard constraints (box bounds + non-overlap)
# =============================================================================
class Geometry:
    def __init__(self, ds: D.InsoleDataset):
        (xlo, xhi), (ylo, yhi) = ds.bounds
        self.lo = np.array([xlo, ylo])
        self.hi = np.array([xhi, yhi])
        self.r = ds.sensor_radius_cm
        self.dmin = 2.0 * self.r + MIN_GAP_CM

    def clamp(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(xy, self.lo, self.hi)

    def min_dist(self, xy: np.ndarray) -> float:
        if len(xy) < 2:
            return np.inf
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    def is_feasible(self, xy: np.ndarray) -> bool:
        in_box = np.all(xy >= self.lo - 1e-12) and np.all(xy <= self.hi + 1e-12)
        return bool(in_box and self.min_dist(xy) >= self.dmin - 1e-12)

    def repair(self, xy: np.ndarray, rng: np.random.Generator, max_iter: int = 60
               ) -> Tuple[np.ndarray, bool]:
        """Clamp to the box, then push overlapping pairs apart until feasible."""
        xy = self.clamp(xy.copy())
        n = len(xy)
        for _ in range(max_iter):
            diff = xy[:, None, :] - xy[None, :, :]
            d = np.linalg.norm(diff, axis=-1)
            a_idx, b_idx = np.where(np.triu(d < self.dmin, k=1))
            if len(a_idx) == 0:
                return xy, True
            for a, b in zip(a_idx, b_idx):
                v = xy[a] - xy[b]
                dist = float(np.hypot(*v))
                if dist < 1e-9:
                    ang = rng.uniform(0, 2 * np.pi)
                    u = np.array([np.cos(ang), np.sin(ang)])
                else:
                    u = v / dist
                push = 0.5 * (self.dmin - dist) + 1e-4
                xy[a] += u * push
                xy[b] -= u * push
            xy = self.clamp(xy)
        return xy, self.is_feasible(xy) if n else True

    def random_layout(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform random feasible layout by sequential rejection sampling."""
        for _ in range(100):
            pts: List[np.ndarray] = []
            for _ in range(n):
                for _ in range(2000):
                    p = rng.uniform(self.lo, self.hi)
                    if all(np.hypot(*(p - q)) >= self.dmin for q in pts):
                        pts.append(p)
                        break
                else:
                    break
            if len(pts) == n:
                return np.asarray(pts)
        raise RuntimeError(f"Could not place {n} non-overlapping sensors.")

    def overlap_constraints(self, x: np.ndarray) -> np.ndarray:
        """SLSQP inequality g_ab(x) = d_ab^2 - (2r + gap)^2 >= 0."""
        xy = x.reshape(-1, 2)
        a, b = np.triu_indices(len(xy), k=1)
        diff = xy[a] - xy[b]
        return np.sum(diff * diff, axis=1) - self.dmin ** 2

    def overlap_jacobian(self, x: np.ndarray) -> np.ndarray:
        xy = x.reshape(-1, 2)
        n = len(xy)
        a, b = np.triu_indices(n, k=1)
        diff = xy[a] - xy[b]
        J = np.zeros((len(a), 2 * n))
        rows = np.arange(len(a))
        J[rows, 2 * a] = 2 * diff[:, 0]
        J[rows, 2 * a + 1] = 2 * diff[:, 1]
        J[rows, 2 * b] = -2 * diff[:, 0]
        J[rows, 2 * b + 1] = -2 * diff[:, 1]
        return J


def inertia(it: int, iters: int) -> float:
    return W_START - (W_START - W_END) * it / max(1, iters - 1)


# =============================================================================
# 4. Pareto utilities
# =============================================================================
def dominates(e1: float, n1: int, e2: float, n2: int) -> bool:
    return (e1 <= e2 and n1 <= n2) and (e1 < e2 or n1 < n2)


def pareto_front(best: Dict[int, float]) -> List[int]:
    """Non-dominated N values of a best-E-per-N table (minimize E and N)."""
    front, best_so_far = [], np.inf
    for n in sorted(best):
        if np.isfinite(best[n]) and best[n] < best_so_far:
            front.append(n)
            best_so_far = best[n]
    return front


def hypervolume(best: Dict[int, float], e_ref: float = E_REF_MS,
                n_ref: float = D.N_MAX + 1) -> float:
    """2-D hypervolume (N, E) dominated by the front, w.r.t. (n_ref, e_ref)."""
    front = pareto_front(best)
    hv = 0.0
    for i, n in enumerate(front):
        n_next = front[i + 1] if i + 1 < len(front) else n_ref
        hv += (n_next - n) * max(0.0, e_ref - best[n])
    return hv


def crowding(front_ns: List[int], best: Dict[int, float]) -> np.ndarray:
    """NSGA-II crowding distance along the 2-D front (boundary points = inf)."""
    k = len(front_ns)
    cd = np.zeros(k)
    if k <= 2:
        return np.full(k, np.inf)
    es = np.array([best[n] for n in front_ns])
    ns = np.array(front_ns, dtype=float)
    cd[0] = cd[-1] = np.inf
    e_rng = max(1e-9, es.max() - es.min())
    n_rng = max(1e-9, ns.max() - ns.min())
    for i in range(1, k - 1):
        cd[i] = (es[i - 1] - es[i + 1]) / e_rng + (ns[i + 1] - ns[i - 1]) / n_rng
    return cd


def knee_point(front_ns: List[int], best: Dict[int, float]) -> int:
    """Elbow: front point farthest from the chord joining its two extremes."""
    if len(front_ns) <= 2:
        return front_ns[0]
    ns = np.array(front_ns, dtype=float)
    es = np.array([best[n] for n in front_ns])
    nn = (ns - ns.min()) / (ns.max() - ns.min())
    en = (es - es.min()) / max(1e-9, es.max() - es.min())
    p0, p1 = np.array([nn[0], en[0]]), np.array([nn[-1], en[-1]])
    chord = (p1 - p0) / np.linalg.norm(p1 - p0)
    pts = np.column_stack([nn, en]) - p0
    dist = np.abs(pts[:, 0] * chord[1] - pts[:, 1] * chord[0])
    return int(front_ns[int(np.argmax(dist))])


# =============================================================================
# 5. MOPSO (main method) — analysis_plan.md step 6
# =============================================================================
def run_mopso(ev: Evaluator, geom: Geometry, seed: int, particles: int, iters: int,
              log=print) -> Dict:
    """
    Multi-objective PSO over (E, N). Each particle carries a continuous N plus
    16 (x, y) slots; only the first round(N) slots are active. All 16 slots
    are kept feasible, so every prefix of active sensors is feasible too.
    Leaders: binary tournament on crowding distance over the non-dominated
    archive. Turbulence (mutation) decays over iterations for diversity.
    """
    rng = np.random.default_rng(seed)
    nmax = D.N_MAX
    n_lo, n_hi = D.N_MIN - 0.49, D.N_MAX + 0.49
    vmax_xy = VMAX_FRAC * (geom.hi - geom.lo)
    vmax_n = VMAX_FRAC * (n_hi - n_lo)

    def n_int(nc: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(nc), D.N_MIN, D.N_MAX).astype(int)

    # Initialise: N evenly spread over 2..16, positions random & feasible
    nc = np.linspace(n_lo, n_hi, particles) + rng.uniform(-0.2, 0.2, particles)
    nc = np.clip(nc, n_lo, n_hi)
    xy = np.stack([geom.random_layout(nmax, rng) for _ in range(particles)])
    v_n = np.zeros(particles)
    v_xy = np.zeros_like(xy)

    nfev0 = ev.nfev
    N = n_int(nc)
    E = ev([xy[p, :N[p]] for p in range(particles)])

    pb_nc, pb_xy, pb_E, pb_N = nc.copy(), xy.copy(), E.copy(), N.copy()
    archive: Dict[int, Dict] = {}   # N -> {"E", "xy_full", "nc"}

    def update_archive(E_, N_, xy_, nc_):
        for p in range(particles):
            n = int(N_[p])
            if n not in archive or E_[p] < archive[n]["E"]:
                archive[n] = {"E": float(E_[p]), "xy_full": xy_[p].copy(), "nc": float(n)}

    update_archive(E, N, xy, nc)
    history = []

    def record(it):
        best = {n: a["E"] for n, a in archive.items()}
        history.append({
            "iter": it,
            "nfev": ev.nfev - nfev0,
            "hypervolume": hypervolume(best),
            **{f"E_N{n}": best.get(n, np.nan) for n in N_RANGE},
        })

    record(0)
    for it in range(1, iters + 1):
        w = inertia(it - 1, iters)
        best = {n: a["E"] for n, a in archive.items()}
        front = pareto_front(best)
        cd = crowding(front, best)

        # Leader per particle: binary tournament on crowding distance
        i1 = rng.integers(0, len(front), particles)
        i2 = rng.integers(0, len(front), particles)
        pick = np.where(cd[i1] >= cd[i2], i1, i2)
        lead_xy = np.stack([archive[front[k]]["xy_full"] for k in pick])
        lead_nc = np.array([archive[front[k]]["nc"] for k in pick])

        r1, r2 = rng.random((particles, 1, 1)), rng.random((particles, 1, 1))
        v_xy = w * v_xy + C1 * r1 * (pb_xy - xy) + C2 * r2 * (lead_xy - xy)
        v_xy = np.clip(v_xy, -vmax_xy, vmax_xy)
        r1n, r2n = rng.random(particles), rng.random(particles)
        v_n = w * v_n + C1 * r1n * (pb_nc - nc) + C2 * r2n * (lead_nc - nc)
        v_n = np.clip(v_n, -vmax_n, vmax_n)

        new_xy = xy + v_xy
        new_nc = nc + v_n

        # Turbulence: decaying-probability mutation of N and one sensor
        p_mut = 0.5 * (1.0 - it / iters) ** 2
        scale = 1.0 - it / iters
        for p in np.where(rng.random(particles) < p_mut)[0]:
            new_nc[p] += rng.normal(0.0, 2.0 * scale + 0.5)
            k = rng.integers(0, nmax)
            new_xy[p, k] += rng.normal(0.0, 1.0, 2) * scale * (geom.hi - geom.lo) * 0.25

        hit = (new_nc < n_lo) | (new_nc > n_hi)
        new_nc = np.clip(new_nc, n_lo, n_hi)
        v_n[hit] = 0.0

        # Hard constraints: clamp + repair; failed repair keeps previous position
        for p in range(particles):
            rep, ok = geom.repair(new_xy[p], rng)
            if ok:
                v_xy[p] = rep - xy[p]
                xy[p] = rep
            else:
                v_xy[p] = 0.0
        nc = new_nc
        N = n_int(nc)
        E = ev([xy[p, :N[p]] for p in range(particles)])

        # Personal best by Pareto dominance (ties: 50/50 swap)
        for p in range(particles):
            new_dom = dominates(E[p], N[p], pb_E[p], pb_N[p])
            old_dom = dominates(pb_E[p], pb_N[p], E[p], N[p])
            if new_dom or (not old_dom and rng.random() < 0.5):
                pb_nc[p], pb_xy[p], pb_E[p], pb_N[p] = nc[p], xy[p].copy(), E[p], N[p]

        update_archive(E, N, xy, nc)
        record(it)
        if it % max(1, iters // 10) == 0 or it == iters:
            hv = history[-1]["hypervolume"]
            log(f"  MOPSO seed {seed}: iter {it:4d}/{iters}  nfev {ev.nfev - nfev0:6d}  "
                f"front N={pareto_front({n: a['E'] for n, a in archive.items()})}  HV={hv:.0f}")

    layouts = {n: a["xy_full"][:n].copy() for n, a in archive.items()}
    return {
        "seed": seed,
        "best_E": {n: a["E"] for n, a in archive.items()},
        "layouts": layouts,
        "history": pd.DataFrame(history),
        "nfev": ev.nfev - nfev0,
    }


# =============================================================================
# 6. Single-objective PSO sweep (check) — analysis_plan.md step 7
# =============================================================================
def run_pso(ev: Evaluator, geom: Geometry, n: int, seed: int, particles: int,
            iters: int) -> Dict:
    rng = np.random.default_rng(seed)
    vmax = VMAX_FRAC * (geom.hi - geom.lo)
    xy = np.stack([geom.random_layout(n, rng) for _ in range(particles)])
    v = np.zeros_like(xy)
    nfev0 = ev.nfev
    E = ev(list(xy))
    pb_xy, pb_E = xy.copy(), E.copy()
    g = int(np.argmin(pb_E))
    hist = [(ev.nfev - nfev0, float(pb_E[g]))]
    for it in range(iters):
        w = inertia(it, iters)
        r1, r2 = rng.random((particles, 1, 1)), rng.random((particles, 1, 1))
        v = w * v + C1 * r1 * (pb_xy - xy) + C2 * r2 * (pb_xy[g] - xy)
        v = np.clip(v, -vmax, vmax)
        new_xy = xy + v
        for p in range(particles):
            rep, ok = geom.repair(new_xy[p], rng)
            if ok:
                v[p] = rep - xy[p]
                xy[p] = rep
            else:
                v[p] = 0.0
        E = ev(list(xy))
        better = E < pb_E
        pb_xy[better], pb_E[better] = xy[better], E[better]
        g = int(np.argmin(pb_E))
        hist.append((ev.nfev - nfev0, float(pb_E[g])))
    return {"N": n, "E": float(pb_E[g]), "xy": pb_xy[g].copy(),
            "history": np.array(hist), "nfev": ev.nfev - nfev0}


# =============================================================================
# 7. Multi-start SLSQP comparison — analysis_plan.md step 8
# =============================================================================
def _slsqp_job(args: Tuple) -> Dict:
    """Runs inside a worker: one SLSQP start with finite-difference gradients."""
    n, x0, maxiter, fd_eps, start_id, aggregation = args
    geom = Geometry(_DS)
    nfev = [0]
    cache: Dict[bytes, float] = {}
    history: List[Tuple[int, float]] = []

    def f(x: np.ndarray) -> float:
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key not in cache:
            nfev[0] += 1
            cache[key] = _eval_E((aggregation, x))
        return cache[key]

    def callback(xk: np.ndarray) -> None:
        history.append((nfev[0], f(xk)))

    E0 = f(x0)
    history.append((nfev[0], E0))
    bounds = [(geom.lo[0], geom.hi[0]), (geom.lo[1], geom.hi[1])] * n
    t0 = time.perf_counter()
    res = minimize(
        f, x0, method="SLSQP", bounds=bounds,
        constraints=[{"type": "ineq", "fun": geom.overlap_constraints,
                      "jac": geom.overlap_jacobian}],
        callback=callback,
        options={"maxiter": maxiter, "ftol": 1e-4, "eps": fd_eps},
    )
    runtime = time.perf_counter() - t0
    x = np.asarray(res.x)
    g = geom.overlap_constraints(x)
    box_viol = float(np.max(np.r_[0.0, geom.lo[0] - x[0::2], x[0::2] - geom.hi[0],
                                  geom.lo[1] - x[1::2], x[1::2] - geom.hi[1]]))
    return {
        "N": n, "start": start_id, "E0": E0, "E_final": f(x),
        "nfev": nfev[0], "nit": int(res.nit), "runtime_s": runtime,
        "success": bool(res.success), "message": str(res.message),
        "max_overlap_violation_cm2": float(max(0.0, -g.min())) if len(g) else 0.0,
        "max_box_violation_cm": box_viol,
        "feasible": bool(geom.is_feasible(x.reshape(-1, 2))),
        "xy": x.reshape(-1, 2), "history": np.array(history),
    }


def run_slsqp(pool: Pool, geom: Geometry, ns: Sequence[int], starts: int,
              maxiter: int, seed: int, aggregation: str, fd_eps: float = 0.05) -> List[Dict]:
    rng = np.random.default_rng(seed)
    jobs = [(n, geom.random_layout(n, rng).ravel(), maxiter, fd_eps, s, aggregation)
            for n in ns for s in range(starts)]
    return pool.map(_slsqp_job, jobs, chunksize=1)


# =============================================================================
# 8. Cross-cadence validation — analysis_plan.md step 10
# =============================================================================
def cross_cadence(ds: D.InsoleDataset, layouts: Dict[str, np.ndarray],
                  aggregation: str) -> pd.DataFrame:
    rows = []
    for name, xy in layouts.items():
        per_pace = {}
        for pace in TEST_PACES:
            trials = ds.get_trials_by_pace(pace)
            if not trials:
                continue
            scores = pd.DataFrame(
                [{"patient": tr.patient_id, **D.score_trial(xy, tr, ds, aggregation)}
                 for tr in trials]
            )
            by_pt = scores.groupby("patient")["E_ms"].mean()
            per_pace[pace] = float(scores["E_ms"].mean())
            rows.append({
                "aggregation": aggregation,
                "layout": name,
                "N": len(xy),
                "pace": pace,
                "n_trials": len(scores),
                "n_participants": int(scores["patient"].nunique()),
                "E_mean_ms": float(scores["E_ms"].mean()),
                "E_sd_across_participants_ms": float(by_pt.std(ddof=1)),
                "E_worst_participant_ms": float(by_pt.max()),
                "HS_ms": float(scores["HS_ms"].mean()),
                "TO_ms": float(scores["TO_ms"].mean()),
                "Step_ms": float(scores["Step_ms"].mean()),
                "Stride_ms": float(scores["Stride_ms"].mean()),
                "delta_vs_NP_ms": float(scores["E_ms"].mean()) - per_pace.get("NP", np.nan),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 9. Figures — analysis_plan.md step 11
# =============================================================================
def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_pareto(path: Path, best: Dict[int, float], seed_bests: List[Dict[int, float]],
                sweep: Optional[Dict[int, float]], baseline_E: float, knee: int,
                method: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _style(ax)
    ns = sorted(best)
    front = set(pareto_front(best))

    if len(seed_bests) > 1:
        lo = [min(sb.get(n, np.inf) for sb in seed_bests) for n in ns]
        hi = [max(sb.get(n, -np.inf) for sb in seed_bests) for n in ns]
        ax.vlines(ns, lo, hi, color=BLUE, alpha=0.35, linewidth=4,
                  label=f"MOPSO spread over {len(seed_bests)} seeds")

    ax.plot(ns, [best[n] for n in ns], color=BLUE, linewidth=2, zorder=3)
    ax.scatter([n for n in ns if n in front], [best[n] for n in ns if n in front],
               s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4,
               label="MOPSO Pareto front")
    dom = [n for n in ns if n not in front]
    if dom:
        ax.scatter(dom, [best[n] for n in dom], s=40, facecolor="white", edgecolor=BLUE,
                   linewidth=1.5, zorder=4, label="Best for N (dominated)")

    if sweep:
        sn = sorted(sweep)
        ax.plot(sn, [sweep[n] for n in sn], color=ORANGE, linewidth=2, linestyle="--",
                marker="o", markersize=4, zorder=2, label="Single-objective PSO (check)")

    ax.axhline(baseline_E, color=INK2, linewidth=1, linestyle=":", zorder=1)
    ax.text(D.N_MAX + 0.3, baseline_E, f"All 32 real sensors\n{baseline_E:.1f} ms",
            color=INK2, fontsize=8, va="center", ha="left")

    ax.scatter([knee], [best[knee]], s=220, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
    ax.annotate(f"Elbow: N = {knee}, E = {best[knee]:.1f} ms", (knee, best[knee]),
                xytext=(30, 70), textcoords="offset points", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))

    ax.set_xticks(list(N_RANGE))
    ax.set_xlim(D.N_MIN - 0.5, D.N_MAX + 2.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of sensors, N", color=INK2)
    ax.set_ylabel("Gait-timing RMSE, E (ms)", color=INK2)
    ax.set_title(f"Accuracy vs. sensor count — {method} (trained on Normal Pace)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(0.86, 1.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_layouts(path: Path, ds: D.InsoleDataset, layouts: Dict[int, np.ndarray],
                 best: Dict[int, float], geom: Geometry, method: str) -> None:
    ns = [n for n in PLOT_NS if n in layouts]
    heat = ds.mean_pressure_map(TRAIN_PACE)
    heat = np.ma.masked_where(~ds.insole_mask, heat)
    w, l, r = ds.width_cm, ds.length_cm, ds.sensor_radius_cm

    fig, axes = plt.subplots(1, len(ns), figsize=(2.6 * len(ns) + 0.8, 6.2), dpi=200,
                             squeeze=False)
    im = None
    for ax, n in zip(axes[0], ns):
        im = ax.imshow(heat, origin="lower", extent=(0, w, 0, l), cmap=SEQ_BLUE,
                       aspect="equal", interpolation="bilinear")
        ax.plot(ds.outline_cm[:, 0], ds.outline_cm[:, 1], ".", color=MUTED, markersize=0.6)
        ax.add_patch(Rectangle((0, 0), w, l, fill=False, edgecolor=INK2, linewidth=1))
        ax.add_patch(Rectangle(tuple(geom.lo), *(geom.hi - geom.lo), fill=False,
                               edgecolor=INK2, linewidth=0.7, linestyle=":"))
        ax.scatter(ds.sensor_xy_left[:, 0], ds.sensor_xy_left[:, 1], marker="+", s=14,
                   color=INK2, linewidth=0.7, zorder=3)
        for x, y in layouts[n]:
            ax.add_patch(Circle((x, y), r, facecolor=ORANGE, alpha=0.85,
                                edgecolor="white", linewidth=1.5, zorder=4))
        chk = D.check_constraints(layouts[n], w, l, r)
        ax.set_title(f"N = {n}\nE = {best[n]:.1f} ms", fontsize=10, color=INK)
        ax.text(0.5, -0.06, f"min edge gap {10 * (chk['min_center_dist_cm'] - 2 * r):.2f} mm\n"
                f"feasible: {'yes' if chk['feasible'] else 'NO'}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
        ax.set_xlim(-0.3, w + 0.3)
        ax.set_ylim(-0.3, l + 0.3)
        ax.set_xticks([0, w])
        ax.set_xticklabels(["0", f"{w:.1f}"], fontsize=7, color=INK2)
        ax.set_yticks([0, l])
        ax.set_yticklabels(["heel", "toe"], fontsize=7, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Mean NP pressure (ADC counts)", fontsize=8, color=INK2)
    cbar.ax.tick_params(labelsize=7, colors=INK2)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Optimal layouts — {method}\n(orange = sensor, radius r; + = 32 real sensors; "
                 "dotted box = r ≤ x ≤ w−r, r ≤ y ≤ l−r)", fontsize=9, color=INK2, y=0.99)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(path: Path, mopso_runs: List[Dict], pso: Dict[int, Dict],
                     slsqp: List[Dict], method: str) -> None:
    ns = list(SLSQP_NS)
    fig, axes = plt.subplots(1, len(ns) + 1, figsize=(5.0 * (len(ns) + 1), 4.3), dpi=200)

    for ax, n in zip(axes, ns):
        _style(ax)
        for i, run in enumerate(mopso_runs):
            h = run["history"]
            col = f"E_N{n}"
            ok = h[col].notna()
            ax.step(h.loc[ok, "nfev"], h.loc[ok, col], where="post", color=BLUE, linewidth=2,
                    alpha=1.0 if i == 0 else 0.45,
                    label="MOPSO (best at this N; evals over all N)" if i == 0 else None)
        if n in pso:
            hh = pso[n]["history"]
            ax.step(hh[:, 0], hh[:, 1], where="post", color=ORANGE, linewidth=2,
                    label=f"PSO, N = {n} only")
        runs = [s for s in slsqp if s["N"] == n]
        for j, s in enumerate(runs):
            hh = s["history"]
            ax.plot(hh[:, 0], hh[:, 1], color=AQUA, linewidth=1, alpha=0.8,
                    label="SLSQP iterates (one line per start; dot = final)" if j == 0 else None)
            ax.scatter(hh[-1, 0], hh[-1, 1], s=16, color=AQUA, edgecolor="white",
                       linewidth=1, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations (log scale)", color=INK2)
        ax.set_ylabel("E (ms, log scale)", color=INK2)
        ax.set_title(f"N = {n}", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = axes[-1]
    _style(ax)
    for i, run in enumerate(mopso_runs):
        h = run["history"]
        ax.plot(h["nfev"], h["hypervolume"], color=BLUE, linewidth=2,
                alpha=1.0 if i == 0 else 0.45,
                label=f"MOPSO seed {run['seed']}")
    ax.set_xlabel("Function evaluations", color=INK2)
    ax.set_ylabel("Hypervolume (sensor·ms)", color=INK2)
    ax.set_title("MOPSO front quality", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.suptitle(f"Convergence — {method}", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_method_comparison(path: Path, summaries: Dict[str, Dict]) -> None:
    """Both detection methods' best-E-per-N curves and 32-sensor references."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _style(ax)
    for agg, col in zip(summaries, (BLUE, ORANGE)):
        sm = summaries[agg]
        best, front = sm["best"], set(sm["front"])
        ns = sorted(best)
        ax.plot(ns, [best[n] for n in ns], color=col, linewidth=2, label=METHOD_LABELS[agg])
        ax.scatter([n for n in ns if n in front], [best[n] for n in ns if n in front],
                   s=40, color=col, edgecolor="white", linewidth=1.5, zorder=4)
        dom = [n for n in ns if n not in front]
        ax.scatter(dom, [best[n] for n in dom], s=40, facecolor="white", edgecolor=col,
                   linewidth=1.5, zorder=4)
        ax.axhline(sm["baseline_E"], color=col, linewidth=1, linestyle=":")
        ax.text(D.N_MAX + 0.3, sm["baseline_E"], f"32 real sensors\n{sm['baseline_E']:.1f} ms",
                color=INK2, fontsize=7.5, va="center")
    ax.set_xticks(list(N_RANGE))
    ax.set_xlim(D.N_MIN - 0.5, D.N_MAX + 2.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of sensors, N", color=INK2)
    ax.set_ylabel("Gait-timing RMSE, E (ms)", color=INK2)
    ax.set_title("Detection method comparison (filled = Pareto front, hollow = dominated)",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(0.86, 0.0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# 10. Main pipeline
# =============================================================================
def run_method(agg: str, pool: Pool, workers: int, cfg: Dict, skip_sweep: bool,
               ds: D.InsoleDataset, geom: Geometry, out: Path, log) -> Dict:
    """Full analysis_plan.md steps 6–11 for one detection method; outputs in `out`."""
    method = METHOD_LABELS[agg]
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n######## Detection method: {method} ('{agg}') ########")
    ev = Evaluator(pool, workers, agg)

    # ---- Step 6: MOPSO, several seeds ------------------------------------------
    mopso_runs = []
    for s in range(cfg["seeds"]):
        t0 = time.perf_counter()
        run = run_mopso(ev, geom, seed=s, particles=cfg["particles"], iters=cfg["iters"], log=log)
        run["runtime_s"] = time.perf_counter() - t0
        mopso_runs.append(run)
        log(f"MOPSO seed {s}: {run['nfev']} evals in {run['runtime_s']:.1f} s")

    # Combine seeds: best layout per N across all runs
    best: Dict[int, float] = {}
    layouts: Dict[int, np.ndarray] = {}
    for run in mopso_runs:
        for n, e in run["best_E"].items():
            if n not in best or e < best[n]:
                best[n], layouts[n] = e, run["layouts"][n]
    front = pareto_front(best)
    knee = knee_point(front, best)

    # ---- Step 7: PSO sweep check -----------------------------------------------
    pso: Dict[int, Dict] = {}
    if not skip_sweep:
        t0 = time.perf_counter()
        for n in N_RANGE:
            pso[n] = run_pso(ev, geom, n, seed=100 + n,
                             particles=cfg["sweep_particles"], iters=cfg["sweep_iters"])
            log(f"  PSO N={n:2d}: E={pso[n]['E']:.2f} ms  (MOPSO {best.get(n, np.nan):.2f})")
        log(f"PSO sweep done in {time.perf_counter() - t0:.1f} s")

    # ---- Step 8: multi-start SLSQP ---------------------------------------------
    t0 = time.perf_counter()
    slsqp = run_slsqp(pool, geom, SLSQP_NS, cfg["slsqp_starts"], cfg["slsqp_maxiter"],
                      seed=7, aggregation=agg)
    log(f"SLSQP: {len(slsqp)} starts in {time.perf_counter() - t0:.1f} s")

    # ---- Step 9: pick the optimum (weight sweep + elbow) -----------------------
    weight_rows = []
    for w2 in W2_VALUES:
        f = {n: 1.0 * best[n] + w2 * n for n in front}
        n_star = min(f, key=f.get)
        weight_rows.append({"aggregation": agg, "W1": 1.0, "W2_ms_per_sensor": w2,
                            "N_star": n_star, "E_star_ms": best[n_star], "f_star": f[n_star]})
    weights_df = pd.DataFrame(weight_rows)

    # ---- Step 10: cross-cadence validation -------------------------------------
    eval_layouts = {f"MOPSO N={n}": layouts[n] for n in sorted(layouts)}
    eval_layouts["All 32 real sensors (reference)"] = ds.sensor_xy_left
    cross_df = cross_cadence(ds, eval_layouts, agg)
    cross_df["on_pareto_front"] = cross_df["N"].isin(front) & cross_df["layout"].str.startswith("MOPSO")
    cross_df["is_elbow"] = cross_df["layout"] == f"MOPSO N={knee}"
    ref = cross_df[cross_df["layout"].str.startswith("All 32") & (cross_df["pace"] == TRAIN_PACE)]
    baseline_E = float(ref["E_mean_ms"].iloc[0])

    # ---- Step 11: exports ------------------------------------------------------
    cross_df.to_csv(out / "cross_cadence_validation.csv", index=False, float_format="%.3f")
    weights_df.to_csv(out / "weight_sweep.csv", index=False, float_format="%.3f")

    front_rows = []
    for n in sorted(best):
        chk = D.check_constraints(layouts[n], ds.width_cm, ds.length_cm, ds.sensor_radius_cm)
        seed_es = [r["best_E"].get(n, np.nan) for r in mopso_runs]
        front_rows.append({
            "aggregation": agg, "N": n, "E_ms": best[n], "on_pareto_front": n in front,
            "is_elbow": n == knee,
            "E_seed_min_ms": np.nanmin(seed_es), "E_seed_max_ms": np.nanmax(seed_es),
            "PSO_check_E_ms": pso[n]["E"] if n in pso else np.nan,
            "feasible": chk["feasible"], "min_center_dist_cm": chk["min_center_dist_cm"],
        })
    front_df = pd.DataFrame(front_rows)
    front_df.to_csv(out / "mopso_front.csv", index=False, float_format="%.3f")

    slsqp_df = pd.DataFrame([{k: v for k, v in s.items() if k not in ("xy", "history")} for s in slsqp])
    slsqp_df.insert(0, "aggregation", agg)
    slsqp_df.to_csv(out / "slsqp_runs.csv", index=False, float_format="%.3f")

    perf_rows = [{"method": f"MOPSO seed {r['seed']}", "N": "2–16", "nfev": r["nfev"],
                  "runtime_s": r["runtime_s"], "best_E_ms": np.nan} for r in mopso_runs]
    for n in SLSQP_NS:
        sub = slsqp_df[slsqp_df["N"] == n]
        perf_rows.append({"method": "SLSQP (multi-start)", "N": n, "nfev": int(sub["nfev"].sum()),
                          "runtime_s": float(sub["runtime_s"].sum()),
                          "best_E_ms": float(sub["E_final"].min())})
        if n in pso:
            perf_rows.append({"method": "PSO (single N)", "N": n, "nfev": pso[n]["nfev"],
                              "runtime_s": np.nan, "best_E_ms": pso[n]["E"]})
        perf_rows.append({"method": "MOPSO (combined seeds)", "N": n, "nfev": np.nan,
                          "runtime_s": np.nan, "best_E_ms": best.get(n, np.nan)})
    pd.DataFrame(perf_rows).to_csv(out / "algorithm_performance.csv", index=False, float_format="%.3f")

    with open(out / "optimal_layouts.json", "w") as fh:
        json.dump({"aggregation": agg,
                   "units": "cm, left-insole frame (x: 0..w, y: 0=heel..l)",
                   "width_cm": ds.width_cm, "length_cm": ds.length_cm,
                   "sensor_radius_cm": ds.sensor_radius_cm, "elbow_N": knee, "pareto_front_N": front,
                   "layouts": {str(n): {"E_ms": best[n], "xy": layouts[n].round(4).tolist()}
                               for n in sorted(layouts)}}, fh, indent=2)

    plot_pareto(out / "pareto_front.png", best, [r["best_E"] for r in mopso_runs],
                {n: p["E"] for n, p in pso.items()}, baseline_E, knee, method)
    plot_layouts(out / "optimal_layouts.png", ds, layouts, best, geom, method)
    plot_convergence(out / "convergence_comparison.png", mopso_runs, pso, slsqp, method)

    # ---- Summary ---------------------------------------------------------------
    log(f"\n=== Summary: {method} ===")
    log(f"Pareto front N: {front} | elbow N* = {knee} (E = {best[knee]:.2f} ms) | "
        f"32-sensor reference E = {baseline_E:.2f} ms")
    log("Weight sweep (W1 = 1, W2 = ms per sensor): " +
        ", ".join(f"W2={r['W2_ms_per_sensor']:g}→N*={r['N_star']}" for r in weight_rows))
    for n in SLSQP_NS:
        sub = slsqp_df[slsqp_df["N"] == n]
        log(f"SLSQP N={n}: final E {sub['E_final'].min():.1f}–{sub['E_final'].max():.1f} ms over "
            f"{len(sub)} starts (MOPSO {best.get(n, np.nan):.1f}); "
            f"feasible {int(sub['feasible'].sum())}/{len(sub)}")
    piv = cross_df.pivot_table(index="layout", columns="pace", values="E_mean_ms", sort=False)
    log("Cross-cadence E (ms):\n" + piv.round(1).to_string())

    return {"best": best, "front": front, "knee": knee, "baseline_E": baseline_E,
            "front_df": front_df, "cross_df": cross_df, "weights_df": weights_df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    ap.add_argument("--aggregation", choices=(*D.AGGREGATIONS, "both"), default="both",
                    help="sensor-combining rule(s) to optimize (default: both)")
    ap.add_argument("--seeds", type=int, help="number of independent MOPSO runs")
    ap.add_argument("--particles", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--slsqp-starts", type=int)
    ap.add_argument("--skip-sweep", action="store_true", help="skip the PSO-per-N check")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    cfg = dict(PRESETS["quick" if args.quick else "full"])
    for k in ("seeds", "particles", "iters", "slsqp_starts"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    aggs = list(D.AGGREGATIONS) if args.aggregation == "both" else [args.aggregation]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    ds = D.load_cached_dataset()
    geom = Geometry(ds)
    train = ds.get_trials_by_pace(TRAIN_PACE)
    log(f"Config: {cfg} | methods={aggs} | workers={args.workers}")
    log(f"Insole w={ds.width_cm:.3f} cm, l={ds.length_cm:.3f} cm, r={ds.sensor_radius_cm:.3f} cm; "
        f"training on {len(train)} {TRAIN_PACE} trials "
        f"({len({t.patient_id for t in train})} participants)")

    summaries: Dict[str, Dict] = {}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for agg in aggs:
            summaries[agg] = run_method(agg, pool, args.workers, cfg, args.skip_sweep,
                                        ds, geom, out / agg, log)

    # ---- Side-by-side comparison of the detection methods -----------------------
    if len(summaries) > 1:
        pd.concat([s["front_df"] for s in summaries.values()]).to_csv(
            out / "method_comparison_front.csv", index=False, float_format="%.3f")
        pd.concat([s["cross_df"] for s in summaries.values()]).to_csv(
            out / "cross_cadence_validation_all_methods.csv", index=False, float_format="%.3f")
        pd.concat([s["weights_df"] for s in summaries.values()]).to_csv(
            out / "weight_sweep_all_methods.csv", index=False, float_format="%.3f")
        plot_method_comparison(out / "method_comparison.png", summaries)
        log("\n=== Method comparison ===")
        for agg, sm in summaries.items():
            log(f"{METHOD_LABELS[agg]:>20}: front N={sm['front']}, elbow N*={sm['knee']} "
                f"(E={sm['best'][sm['knee']]:.2f} ms), best overall E="
                f"{min(sm['best'].values()):.2f} ms, 32-sensor ref={sm['baseline_E']:.2f} ms")

    log(f"Total runtime {time.perf_counter() - t_start:.1f} s. Outputs in {out}")
    (out / "run_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
