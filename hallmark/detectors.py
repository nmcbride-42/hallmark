"""Versioned LLM-free detector baseline: transforms absorbed from the proposer.

The proposer pre-test had a predeclared interpretation rule: "if new
transforms account for the gain, preserve them as versioned domain detectors
and rerun WITHOUT the LLM." Every function here was originally written by an
LLM proposer (see `hallmark.inventor`), passed the predeclared success rule on
the blind probe set, and was then absorbed as ordinary code: reviewed,
versioned, no model anywhere. The next proposer has to beat this library,
which already contains its predecessors' best output.

Combined score: each transform is z-scored across the batch being ranked and
the maximum taken; a target is interesting if ANY principle fires strongly.

BASELINE_VERSION bumps on any change; rankings from different versions are
not comparable (same discipline as verifier spec hashes).

Hit counts quoted in the comments below are reported historical results on the
60-curve probe set (20 catalogued eclipsing binaries, 40 controls). Reproduce
them with `scripts/score_detectors.py` after `scripts/fetch_probe_set.py`.
"""

import math
import numpy as np

BASELINE_VERSION = "2026-10-29"


# --- lomb_scargle_peak: peak normalized Lomb-Scargle power over 0.05-100 day
# trial periods. Probe set: 7/10 catalogued EBs in top-10, p=0.0116.
def lomb_scargle_peak(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    f = flux[mask]
    if len(t) < 30:
        return 0.0
    med = np.median(f)
    if not np.isfinite(med) or med == 0.0:
        return 0.0
    f = f / med - 1.0
    T = float(np.ptp(t))
    if T <= 0.0:
        return 0.0
    fc = f - np.mean(f)
    var2 = float(np.dot(fc, fc))
    if var2 == 0.0:
        return 0.0
    freq_lo = 1.0 / min(T * 0.5, 100.0)
    freq_hi = 20.0  # 0.05-day minimum period
    n_freq = min(max(int(4.0 * T * freq_hi), 2), 4000)
    freqs = np.linspace(freq_lo, freq_hi, n_freq)
    best = 0.0
    B = 200  # batch size: keeps peak memory ~100 MB
    for i0 in range(0, n_freq, B):
        om = 2.0 * np.pi * freqs[i0:i0 + B]   # (B,)
        ph = np.outer(om, t)                    # (B, N)
        tau = np.arctan2(
            np.sin(2.0 * ph).sum(axis=1),
            np.cos(2.0 * ph).sum(axis=1)
        ) / (2.0 * om)
        ph -= om[:, None] * tau[:, None]
        C = np.cos(ph)   # (B, N)
        S = np.sin(ph)
        YC = C @ fc      # (B,)
        YS = S @ fc
        CC = np.maximum((C * C).sum(axis=1), 1e-30)
        SS = np.maximum((S * S).sum(axis=1), 1e-30)
        # Standard Horne-Baliunas normalisation: P in [0, N/2]
        p = (YC * YC / CC + YS * YS / SS) * (len(t) / (2.0 * var2))
        v = float(p.max())
        if v > best:
            best = v
    return best


# --- string_length_min: one minus the minimum Lafler-Kinman phase-folded
# string length over log-spaced trial periods. Probe set: 8/10, p=0.0014.
def string_length_min(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    f = flux[mask]
    if len(t) < 30:
        return 0.0
    med = np.median(f)
    if not np.isfinite(med) or med == 0.0:
        return 0.0
    f = f / med - 1.0
    T = float(np.ptp(t))
    if T <= 0.0:
        return 0.0
    var_f = float(np.var(f))
    if var_f == 0.0:
        return 0.0
    p_lo = 0.05
    p_hi = min(T * 0.5, 100.0)
    if p_hi <= p_lo:
        return 0.0
    # Log-spaced: denser coverage at short periods where aliases crowd
    periods = np.exp(np.linspace(np.log(p_lo), np.log(p_hi), 1500))
    N = len(t)
    denom = 2.0 * (N - 1) * var_f
    min_theta = 1.0
    for P in periods:
        phase = (t / P) % 1.0
        idx = np.argsort(phase)
        fs = f[idx]
        d = np.diff(fs)
        # theta ~ 1 for noise, << 1 for smooth phased curve
        theta = float(np.dot(d, d)) / denom
        if theta < min_theta:
            min_theta = theta
    # Return 0 for non-periodic (theta ~ 1), positive for periodic (theta < 1)
    return float(max(0.0, 1.0 - min_theta))


# --- acf_secondary_peak: highest secondary peak of the FFT autocorrelation of
# the binned light curve. Probe set: 8/10, p=0.0014.
def acf_secondary_peak(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    f = flux[mask]
    if len(t) < 50:
        return 0.0
    med = np.median(f)
    if not np.isfinite(med) or med == 0.0:
        return 0.0
    f = f / med - 1.0
    T = float(np.ptp(t))
    if T < 1.0:
        return 0.0

    # Estimate native cadence from median spacing of sorted timestamps
    sorted_t = np.sort(t)
    raw_diffs = np.diff(sorted_t)
    pos_diffs = raw_diffs[raw_diffs > 0.0]
    dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.02
    dt = max(dt, 0.01)

    # Bin flux onto regular grid; cap at 8000 bins for speed
    n_bins = min(int(T / dt) + 2, 8000)
    if n_bins < 20:
        return 0.0
    t_min = float(np.min(t))
    dt_grid = T / float(n_bins - 1)

    idx = np.clip(np.round((t - t_min) / dt_grid).astype(int), 0, n_bins - 1)
    f_sum = np.bincount(idx, weights=f, minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    # Empty bins filled with local mean (zero after centering)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_grid = np.where(cnt > 0.0, f_sum / cnt, 0.0)
    f_grid -= np.mean(f_grid)

    # Linear ACF via zero-padded FFT
    N2 = 2 * n_bins
    F = np.fft.rfft(f_grid, n=N2)
    acf = np.fft.irfft(F * np.conj(F))[:n_bins].real
    if acf[0] <= 0.0:
        return 0.0
    acf = acf / acf[0]  # normalise so lag-0 = 1.0

    # Search for local maxima between 0.05-day lag and T/2
    lag_min = max(int(np.ceil(0.05 / dt_grid)), 2)
    lag_max = min(int(T * 0.5 / dt_grid), n_bins - 2)
    if lag_max <= lag_min + 1:
        return 0.0
    seg = acf[lag_min:lag_max + 1]
    if len(seg) < 3:
        return float(max(0.0, float(np.max(seg))))
    peaks = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])
    if not np.any(peaks):
        # No local max — return raw maximum as a weaker signal
        return float(max(0.0, float(np.max(seg))))
    return float(max(0.0, float(seg[1:-1][peaks].max())))



# --- composite_eb_score: dual-branch detector. Branch 1 is a gap-masked,
# multi-transit box-dip SNR that requires >=2 individually significant events;
# branch 2 is a vectorized phase-folded ANOVA F-statistic for continuous
# variability. Written in a later multi-round proposer/critic run (two model
# lineages critiquing each other's code using label-free diagnostics only).
# Probe set: 9/10 top-10 (p=9.2e-5), 15/20 top-20 (p=2.5e-6).
def composite_eb_score(time, flux) -> float:
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 200:
        return 0.0
    o = np.argsort(t)
    t = t[o]
    med_f = float(np.median(f[o]))
    if not np.isfinite(med_f) or med_f == 0.0:
        return 0.0
    f = f[o] / med_f
    br = np.where(np.diff(t) > 0.15)[0]
    st, en = np.r_[t[0], t[br + 1]], np.r_[t[br], t[-1]]
    k = np.searchsorted(st, t, "right") - 1
    keep = (t - st[k] > 0.15) & (en[k] - t > 0.15)
    t, f, k = t[keep], f[keep], k[keep]
    if t.size < 200:
        return 0.0
    dt = 0.005
    b = ((t - t[0]) / dt).astype(int)
    n = np.bincount(b)
    g = n > 0
    tb = t[0] + (np.nonzero(g)[0] + 0.5) * dt
    fb = np.bincount(b, f)[g] / n[g]
    kb = np.rint(np.bincount(b, k)[g] / n[g]).astype(int)
    tt = tb - tb[0]
    span = float(tt[-1])
    if span <= 0.5:
        return 0.0
    _, si = np.unique(kb, return_inverse=True)
    sm = np.array([np.median(fb[si == j]) for j in range(si.max() + 1)])
    c = (tt / 0.75).astype(int)
    _, ci = np.unique(c, return_inverse=True)
    cm = np.array([np.median(fb[ci == j]) for j in range(ci.max() + 1)])
    ct = np.bincount(ci, tb) / np.bincount(ci)
    r = fb - np.interp(tb, ct, cm)
    r -= np.median(r)
    sig = 1.4826 * np.median(np.abs(r)) + 1e-9
    best = 0.0
    for P in np.exp(np.linspace(math.log(0.3), math.log(15.5), 300)):
        if P > 0.9 * span:
            break
        nb = max(30, int(P / 0.01))
        ph = ((tt % P) / P * nb).astype(int) % nb
        S = np.bincount(ph, r, nb)
        N = np.bincount(ph, minlength=nb).astype(float)
        cS = np.r_[0.0, np.cumsum(np.r_[S, S[:16]])]
        cN = np.r_[0.0, np.cumsum(np.r_[N, N[:16]])]
        for w in (1, 2, 4, 8, 16):
            ss = cS[w:w + nb] - cS[:nb]
            nn = cN[w:w + nb] - cN[:nb]
            snr = -ss / (sig * np.sqrt(np.maximum(nn, 1.0)))
            i = int(np.argmax(snr))
            v = float(snr[i])
            if v <= best or nn[i] < 3:
                continue
            sel = ((ph - i) % nb) < w
            ep = np.floor((tt[sel] - (i / nb) * P) / P)
            _, inv = np.unique(ep, return_inverse=True)
            es = np.bincount(inv, r[sel])
            ec = np.bincount(inv)
            if np.sum(es / (sig * np.sqrt(ec)) < -2.0) < 2:
                continue
            frac = float((-es).max() / (-ss[i] + 1e-12))
            best = max(best, v * min(1.0, 2.0 * (1.0 - frac)))
    w2 = np.clip(fb - sm[si], -1e9, 5 * sig)
    w2 -= np.median(w2)
    Nt = w2.size
    nq = 10
    tot = float(np.sum(w2 ** 2)) + 1e-30
    nf = int(min(6000, max(500, 19.75 * span * 8)))
    fr = np.linspace(0.25, 20.0, nf)
    bF, bf = 0.0, 0.25
    for j in range(0, nf, 256):
        fc = fr[j:j + 256]
        mm = fc.size
        idx = ((np.outer(fc, tt) % 1.0) * nq).astype(int) + np.arange(mm)[:, None] * nq
        S = np.bincount(idx.ravel(), np.broadcast_to(w2, (mm, Nt)).ravel(), mm * nq).reshape(mm, nq)
        N = np.bincount(idx.ravel(), minlength=mm * nq).reshape(mm, nq)
        ssb = (S ** 2 / np.maximum(N, 1)).sum(1)
        F = (ssb / (nq - 1)) / ((tot - ssb) / (Nt - nq) + 1e-30)
        q = int(np.argmax(F))
        if F[q] > bF:
            bF, bf = float(F[q]), float(fc[q])
    ph = ((tt * bf) % 1.0 * nq).astype(int) % nq
    mu = np.bincount(ph, w2, nq) / np.maximum(np.bincount(ph, minlength=nq), 1)
    asym = -mu.min() / (abs(mu.max()) + 1e-12)
    sp = max(0.0, math.sqrt(bF) - 2.0) * min(2.0, max(0.6, asym))
    return float(max(best, sp))


def estimate_period(time, flux) -> dict:
    """Return both the composite score and the best-fitting orbital period
    (from both the detached multi-transit dip branch and the ANOVA harmonic branch)."""
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 200:
        return {"score": 0.0, "period_d": None, "branch": "none"}
    o = np.argsort(t)
    t = t[o]
    med_f = float(np.median(f[o]))
    if not np.isfinite(med_f) or med_f == 0.0:
        return {"score": 0.0, "period_d": None, "branch": "none"}
    f = f[o] / med_f
    br = np.where(np.diff(t) > 0.15)[0]
    st, en = np.r_[t[0], t[br + 1]], np.r_[t[br], t[-1]]
    k = np.searchsorted(st, t, "right") - 1
    keep = (t - st[k] > 0.15) & (en[k] - t > 0.15)
    t, f, k = t[keep], f[keep], k[keep]
    if t.size < 200:
        return {"score": 0.0, "period_d": None, "branch": "none"}
    dt = 0.005
    b = ((t - t[0]) / dt).astype(int)
    n = np.bincount(b)
    g = n > 0
    tb = t[0] + (np.nonzero(g)[0] + 0.5) * dt
    fb = np.bincount(b, f)[g] / n[g]
    kb = np.rint(np.bincount(b, k)[g] / n[g]).astype(int)
    tt = tb - tb[0]
    span = float(tt[-1])
    if span <= 0.5:
        return {"score": 0.0, "period_d": None, "branch": "none"}
    _, si = np.unique(kb, return_inverse=True)
    sm = np.array([np.median(fb[si == j]) for j in range(si.max() + 1)])
    c = (tt / 0.75).astype(int)
    _, ci = np.unique(c, return_inverse=True)
    cm = np.array([np.median(fb[ci == j]) for j in range(ci.max() + 1)])
    ct = np.bincount(ci, tb) / np.bincount(ci)
    r = fb - np.interp(tb, ct, cm)
    r -= np.median(r)
    sig = 1.4826 * np.median(np.abs(r)) + 1e-9
    best_dip = 0.0
    best_dip_p = None
    for P in np.exp(np.linspace(math.log(0.3), math.log(min(20.0, 0.55 * span)), 400)):
        if P > 0.9 * span:
            break
        nb = max(30, int(P / 0.01))
        ph = ((tt % P) / P * nb).astype(int) % nb
        S = np.bincount(ph, r, nb)
        N = np.bincount(ph, minlength=nb).astype(float)
        cS = np.r_[0.0, np.cumsum(np.r_[S, S[:16]])]
        cN = np.r_[0.0, np.cumsum(np.r_[N, N[:16]])]
        for w in (1, 2, 4, 8, 16):
            ss = cS[w:w + nb] - cS[:nb]
            nn = cN[w:w + nb] - cN[:nb]
            snr = -ss / (sig * np.sqrt(np.maximum(nn, 1.0)))
            i = int(np.argmax(snr))
            v = float(snr[i])
            if v <= best_dip or nn[i] < 3:
                continue
            sel = ((ph - i) % nb) < w
            ep = np.floor((tt[sel] - (i / nb) * P) / P)
            _, inv = np.unique(ep, return_inverse=True)
            es = np.bincount(inv, r[sel])
            ec = np.bincount(inv)
            if np.sum(es / (sig * np.sqrt(ec)) < -2.0) < 2:
                continue
            frac = float((-es).max() / (-ss[i] + 1e-12))
            cand_score = v * min(1.0, 2.0 * (1.0 - frac))
            if cand_score > best_dip:
                best_dip = cand_score
                best_dip_p = float(P)

    w2 = np.clip(fb - sm[si], -1e9, 5 * sig)
    w2 -= np.median(w2)
    Nt = w2.size
    nq = 10
    tot = float(np.sum(w2 ** 2)) + 1e-30
    nf = int(min(6000, max(500, 19.75 * span * 8)))
    fr = np.linspace(0.25, 20.0, nf)
    bF, bf = 0.0, 0.25
    for j in range(0, nf, 256):
        fc = fr[j:j + 256]
        mm = fc.size
        idx = ((np.outer(fc, tt) % 1.0) * nq).astype(int) + np.arange(mm)[:, None] * nq
        S = np.bincount(idx.ravel(), np.broadcast_to(w2, (mm, Nt)).ravel(), mm * nq).reshape(mm, nq)
        N = np.bincount(idx.ravel(), minlength=mm * nq).reshape(mm, nq)
        ssb = (S ** 2 / np.maximum(N, 1)).sum(1)
        F = (ssb / (nq - 1)) / ((tot - ssb) / (Nt - nq) + 1e-30)
        q = int(np.argmax(F))
        if F[q] > bF:
            bF, bf = float(F[q]), float(fc[q])
    ph = ((tt * bf) % 1.0 * nq).astype(int) % nq
    mu = np.bincount(ph, w2, nq) / np.maximum(np.bincount(ph, minlength=nq), 1)
    asym = -mu.min() / (abs(mu.max()) + 1e-12)
    sp = max(0.0, math.sqrt(bF) - 2.0) * min(2.0, max(0.6, asym))
    anova_p = float(1.0 / bf) if bf > 0 else None

    score_val = float(max(best_dip, sp))
    if best_dip >= sp and best_dip_p is not None:
        return {
            "score": round(score_val, 4),
            "period_d": round(best_dip_p, 6),
            "dip_score": round(best_dip, 4),
            "dip_period_d": round(best_dip_p, 6),
            "anova_score": round(sp, 4),
            "anova_period_d": round(anova_p, 6) if anova_p else None,
            "branch": "detached_box_dip",
        }
    return {
        "score": round(score_val, 4),
        "period_d": round(anova_p, 6) if anova_p else best_dip_p,
        "dip_score": round(best_dip, 4),
        "dip_period_d": round(best_dip_p, 6) if best_dip_p else None,
        "anova_score": round(sp, 4),
        "anova_period_d": round(anova_p, 6) if anova_p else None,
        "branch": "phase_anova",
    }


FIRST_ROUND_TRANSFORMS = [lomb_scargle_peak, string_length_min, acf_secondary_peak]
TRANSFORMS = FIRST_ROUND_TRANSFORMS + [composite_eb_score]


def baseline_scores(curves):
    """curves: {key: (time, flux)} -> {key: combined_score}.

    Z-scores each transform across the batch, takes the max. Failures score
    -inf for that transform rather than crashing the batch.
    """
    import signal

    def _timed(fn, t, f, seconds=10.0):
        def _h(signum, frame):
            raise TimeoutError
        old = signal.signal(signal.SIGALRM, _h)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return float(fn(np.asarray(t, float), np.asarray(f, float)))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    per = {fn.__name__: {} for fn in TRANSFORMS}
    for key, (t, f) in curves.items():
        for fn in TRANSFORMS:
            try:
                per[fn.__name__][key] = _timed(fn, t, f)
            except Exception:
                per[fn.__name__][key] = float("nan")
    return combine_scores(per)


def combine_scores(per):
    """per: {transform_name: {key: raw_score}} -> {key: max z-score}."""
    zs = {}
    for name, vals in per.items():
        arr = np.array([v for v in vals.values() if np.isfinite(v)])
        mu, sd = (arr.mean(), arr.std()) if len(arr) > 2 else (0.0, 1.0)
        zs[name] = {k: ((v - mu) / sd if np.isfinite(v) and sd > 0 else float("-inf"))
                    for k, v in vals.items()}
    keys = next(iter(per.values())).keys() if per else []
    return {key: max(zs[name][key] for name in per) for key in keys}
