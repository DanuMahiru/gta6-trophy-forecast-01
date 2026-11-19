"""GTA VI trophy rarity forecaster for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case's target.

How it works (per trophy, all in logit space, where logit(p) = ln(p/(1-p))
with p the earn rate as a fraction):
  1. Fit a straight line of logit(rate) against ln(1 + hours since release)
     over the case history's last 24 hours, with stale repeats collapsed:
     the drift of the last day is what the next days inherit, while the
     launch-night flood (rates collapsing as new players poured in) is a
     different regime and must not drag the slope. With shorter histories
     the window is the whole history.
  2. Shrink each trophy's slope toward a pooled slope predicted from its
     current level (a line of slope against logit(last rate) across all
     trophies): rare trophies and common trophies drift differently, and
     pooling steadies the noisiest fits (rates print to 2 decimals, so a
     rare trophy's slope from a handful of reads is mostly noise).
  3. Read the shrunk line at the target for the median.
  4. If the whole page is stale (the history holds at most two distinct
     rate vectors), Exophase has not refreshed for hours and the next
     refresh brings a flood of new tracked players who have earned little:
     shift medians down and widen the spread, scaled by how short the
     horizon is. A refresh is the whole story over 90 minutes; over days
     the trend matters more. Fast-earned trophies (Chapter 1, fishing,
     arcade) dilute less than the rest, because their earner counts grow
     with the flood — Chapter 2 is in between.
  5. Spread: fit residual scatter (with leverage at the target) plus a
     horizon random-walk term plus a staleness term, floored and capped,
     so the quantiles are honest about what is not known.

v2 changes vs v1 (2026-11-20): flat shift 1.5 -> 1.7, stale sigma
0.55 -> 0.45, earn-speed bonus added (fast 3.0, mid 0.5), slope
shrinkage K 5 -> 0 (final-like sims: the pooled prior only adds bias
when every trophy has many distinct reads). Trend window stays 24h:
36h/48h windows lost badly on the sims.

v3 changes vs v2 (2026-11-20): ADAPTIVE trend window — the shortest of
12/18/24/36/48h holding at least 3 distinct rate vectors, else all
distinct reads (a 12h window beats 24h on final-like sims, mean 0.138
vs 0.213, because the recent slope is flatter than the 24h average
when the truth saturates; the adaptive rule keeps it robust when the
real page refreshes slowly — the 14:10 UTC snapshot was still live at
22:03 UTC on Nov 19). Flat shift 1.7 -> 1.6 with growth 1.0 -> 1.5
(same 90-min shift, bigger long-horizon shift: the real 11h page move
was ~-2.5 to -2.9 logit, which growth 1.0 under-shifts). Stale sigma
0.45 -> 0.30 (narrower wins on practice, noisy proxies and the
median-bias stress test; wider only helps by 0.008 when the median is
biased the expensive way). MOVE_CAP 2.5 -> 3.0 (the cap bound the 11h
real move). Near-100% median cap logit(99.98) -> logit(99.99) (the cap
alone cost ~0.35 loss on Gone Fishing in every practice test; the
trophy sits at 99.99% in reality). Slope damping was tried (0.5/1/2)
and REJECTED: it hurts every sim scenario, because the shorter window
already captures the saturation effect.

Standard library only, deterministic (no randomness, no wall clock: every
time comes from the case file), reads every *.in in the input directory and
writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE_FALLBACK = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Trend window: the last 12 hours of distinct reads carry the drift the
# target inherits — v3: on final-like sims a 12h window beats 24h badly
# (mean 0.138 vs 0.213; the recent slope is flatter than the 24h average
# when the truth saturates, so the straight line overshoots less). But a
# 12h window only identifies a slope when the page refreshed at least 3
# times inside it, and the real Exophase page refreshes slowly (the
# 14:10 UTC snapshot was still live at 22:03 UTC on Nov 19), so the
# window is ADAPTIVE: the shortest of 12/18/24/36/48h holding at least 3
# distinct rate vectors, else all distinct reads. (v2: fixed 24h; 36h/48h
# tested and lost badly — the launch flood drags the slope.)
WINDOW_OPTIONS = (12.0, 18.0, 24.0, 36.0, 48.0)
WINDOW_H = 12.0  # default; forecast_case picks adaptively per case
# Near-100% median cap: a trophy everyone earns (Gone Fishing) sits at
# 99.99% and must not be capped a whole 0.69 logit below it (v2 capped at
# logit(99.98); the cap alone cost ~0.35 loss on that one trophy in every
# practice test).
NEAR_FULL_MEDIAN_CAP = 99.99
# Prior strength (in effective observations) for slope shrinkage.
# v2: 0 — on final-like sims the pooled prior only adds bias (every
# trophy has many distinct reads in a 2.5-day history, so its own fit is
# well determined); K=0 beat K=3/5/8 on all three sim scenarios.
SHRINK_K = 0.0
# Caps to keep any extrapolation from running away.
SLOPE_CAP = 2.5
MOVE_CAP = 3.0
# Stale-page dilution: how far medians shift down and how much spread the
# refresh risk adds. The shift has a fixed part (the imminent refresh,
# which is the whole story over 90 minutes) plus a part that grows with
# the horizon (dilution keeps compounding while the page sits stale).
FLAT_SHIFT_BASE = 1.6
FLAT_SHIFT_GROWTH = 1.5
FLAT_SIGMA = 0.30
# Earn-speed bonus: not every trophy dilutes alike. Early activities
# (Chapter 1, fishing, arcade) are earned fast, so while new players flood
# in, their earner counts grow almost as fast and their rates barely fall;
# Chapter 2 is in between; late story and completionist trophies earn
# slowly and dilute almost in full. The bonus grows with the horizon the
# same way the shift's growth part does, and is subtracted from the shift.
# v2 values: base 1.5 -> 1.7 and sigma 0.55 -> 0.45 from the extended
# sweep (36 + 54 combos on synthetic practice + the real 14:10 UTC page);
# the bonus itself is calibrated on the real page (Chapter 1's rate moved
# only -0.54 logit in 11h while most trophies moved about -2.9).
BONUS_FAST = 3.0
BONUS_MID = 0.5
# Spread pieces.
HORIZON_SIGMA_PER_SQRT_DAY = 0.15
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.1
NEAR_FULL_SIGMA = 0.8


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def round4(x):
    """Round to 4 decimals, half up (matches the kit starter's rounding)."""
    q = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    v = float(q)
    return int(v) if v.is_integer() else v


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, sxx, xbar, n)."""
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return ybar, 0.0, sxx, xbar, len(xs)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return ybar - b * xbar, b, sxx, xbar, len(xs)


def ols_line(xs, ys):
    """Plain least squares y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ybar, 0.0
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    return ybar - b * xbar, b


def earn_speed_bonus(trophy):
    """How much less this trophy dilutes than the flat shift, in units of
    ln(1 + horizon_hours / 12). Early activities earn fast (bonus), late
    ones earn slowly (no bonus)."""
    d = trophy.get("description", "")
    if "Chapter 1" in d:
        return BONUS_FAST
    if "Chapter 2" in d:
        return BONUS_MID
    low = d.lower()
    if "fishing" in low or "arcade" in low:
        return BONUS_FAST
    return 0.0


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    release = parse_time(case.get("game", {}).get("release_date", "")) or RELEASE_FALLBACK
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    if target is None or checkpoint is None:
        raise ValueError(f"{name} has unparsable at/target")
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log(max(1.0 + (target - release).total_seconds() / 3600.0, 0.05))

    # Collapse consecutive identical reads: a page Exophase has not
    # refreshed is one observation, not dozens.
    distinct = []
    seen = set()
    for read in history:
        key = (tuple(read["rates"]), read.get("players_tracked"))
        if key not in seen:
            seen.add(key)
            distinct.append(read)
    page_stale = len(seen) <= 2

    last = history[-1]
    t_last = (parse_time(last["at"]) - release).total_seconds() / 3600.0

    # The trend window: the shortest window holding at least 3 distinct
    # reads (see WINDOW_OPTIONS above). Shorter histories use all they have.
    window = distinct
    for w in WINDOW_OPTIONS:
        cand = [r for r in distinct
                if (parse_time(r["at"]) - release).total_seconds() / 3600.0 >= t_last - w]
        if len(cand) >= 3:
            window = cand
            break
    if len(window) < 3:
        window = distinct

    # Per-trophy trend fits on the window.
    fits = []
    for i in range(n_tr):
        xs, ys = [], []
        for read in window:
            r = read["rates"][i]
            if r is None:
                continue
            at = parse_time(read["at"])
            if at is None:
                continue
            t = (at - release).total_seconds() / 3600.0
            xs.append(math.log(max(1.0 + t, 0.05)))
            ys.append(logit(r))
        if len(xs) >= 3 and len(set(ys)) >= 2:
            a, b, sxx, xbar, _ = wls_line(xs, ys, [1.0] * len(xs))
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            dof = max(len(xs) - 2, 1)
            s2 = sum(e * e for e in resid) / dof
            fits.append({"a": a, "b": b, "sxx": sxx, "xbar": xbar,
                         "n": len(xs), "s": math.sqrt(max(s2, 0.0))})
        else:
            fits.append(None)

    # Pooled slope prior: slope as a line in logit(last rate), over the
    # trophies whose own fit has signal.
    pool_a, pool_b = 0.0, 0.0
    pool_xs, pool_ys = [], []
    for i, f in enumerate(fits):
        if f is not None:
            pool_xs.append(logit(last["rates"][i] if last["rates"][i] is not None else 0.0))
            pool_ys.append(f["b"])
    if len(pool_xs) >= 4:
        pool_a, pool_b = ols_line(pool_xs, pool_ys)

    # The stale-page refresh is the whole story over a short horizon and
    # keeps compounding over a long one while the page sits stale.
    horizon_hours = case["horizon_minutes"] / 60.0
    flat_shift = FLAT_SHIFT_BASE + FLAT_SHIFT_GROWTH * math.log(1.0 + horizon_hours / 12.0)
    horizon_log = math.log(1.0 + horizon_hours / 12.0)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.0
        last_lg = logit(last_rate)
        f = fits[i]
        if f is not None:
            b_pool = pool_a + pool_b * last_lg
            n_eff = min(f["n"], 8)
            b_star = (n_eff * f["b"] + SHRINK_K * b_pool) / (n_eff + SHRINK_K)
            b_star = max(-SLOPE_CAP, min(SLOPE_CAP, b_star))
            # Refit the intercept with the shrunk slope held fixed, over
            # the same window.
            pts = []
            for read in window:
                r = read["rates"][i]
                if r is None:
                    continue
                at = parse_time(read["at"])
                if at is None:
                    continue
                t = (at - release).total_seconds() / 3600.0
                pts.append((math.log(max(1.0 + t, 0.05)), logit(r)))
            a_star = (sum(y - b_star * x for x, y in pts) / len(pts)) if pts else f["a"]
            median_lg = a_star + b_star * x_target
            # Prediction uncertainty: residual scatter with leverage at
            # the target, in logit units.
            lev = 1.0 / max(f["n"], 1) + (x_target - f["xbar"]) ** 2 / f["sxx"] if f["sxx"] > 1e-12 else 1.0
            sigma_fit = min(f["s"], 0.8) * math.sqrt(1.0 + lev)
        else:
            median_lg = last_lg
            sigma_fit = 0.0

        # A page that has not moved is about to: the next refresh brings a
        # flood of new tracked players who have earned little, so rates
        # fall. Everything but a trophy already at ~100% drops. Fast-earned
        # trophies dilute less: their earner counts grow with the flood.
        sigma_stale = 0.0
        if page_stale:
            if last_rate < 99.5:
                median_lg -= flat_shift - earn_speed_bonus(trophy) * horizon_log
            sigma_stale = FLAT_SIGMA if last_rate < 99.5 else 0.3

        # Keep the median move from the last read inside sane bounds.
        move = median_lg - last_lg
        if move > MOVE_CAP:
            median_lg = last_lg + MOVE_CAP
        elif move < -MOVE_CAP:
            median_lg = last_lg - MOVE_CAP

        sigma = math.sqrt(
            sigma_fit ** 2
            + (HORIZON_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + sigma_stale ** 2
        )
        if last_rate >= 99.0:
            # Near 100% a small rate fall is a huge logit move; allow room.
            sigma = max(sigma, NEAR_FULL_SIGMA)
            median_lg = min(median_lg, logit(NEAR_FULL_MEDIAN_CAP))
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))

        qs = [round4(min(99.99, max(0.01, inv_logit(median_lg + z * sigma)))) for z in Z]
        # Guarantee non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
