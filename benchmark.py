"""
benchmark.py  -  runs ALL experiments and writes figures + tables into ./results
==============================================================================

    python benchmark.py            full run (about 1-3 minutes)
    python benchmark.py --quick    small run for a fast check

Everything is evaluated on TEST data with random seeds that were never used for
training (train 1000-, validation 2000-, test 5000-).  All filters are causal.

Experiments
  A  Simulated click test (20 targets x 40 sessions per tremor level)
  B  Lag, damage to a healthy hand, residual tremor when holding still
  C  Trade-off curve (lag vs residual tremor) incl. low-pass / Kalman sweeps
  D  Robustness: tremor frequencies / shapes the AI never saw in training
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy import signal

import tremor_lib as T

QUICK = "--quick" in sys.argv
N_SESS = 8 if QUICK else 40
N_TARGETS = 20
TEST_SEED = 5000
OUT = "results"
os.makedirs(OUT, exist_ok=True)

SEVERITIES = [("Healthy (0 px)", 0.0), ("Mild (4 px)", 4.0),
              ("Moderate (8 px)", 8.0), ("Severe (12 px)", 12.0)]

# ---------------------------------------------------------------- filters
m = np.load("model.npz")
AI_NN = T.LearnedNN({k: m[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")})
AI_LIN = T.LearnedLinear(m["ridge"])
# baselines tuned so that their tremor suppression MATCHES the AI network (~85 %)
LP = T.LowPass(2.5)
KAL = T.KalmanCV(q=100.0, r=25.0)
NOF = T.NoFilter()
FILTERS = [NOF, LP, KAL, AI_LIN, AI_NN]
NAMES = [f.name for f in FILTERS]

# ---------------------------------------------------------------- style (validated palette)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COL = {"No filter": MUTED, LP.name: "#eb6834", KAL.name: "#1baf7a",
       AI_LIN.name: "#eda100", AI_NN.name: "#2a78d6"}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "font.family": "sans-serif",
    })
    return plt


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ================================================================= A: click test
def run_clicktest(amp, n_sess, seed, tremor_fn=None):
    acc = {f.name: dict(hit=[], time=[], err=[], rad=[], rmse=[], sess_time=[]) for f in FILTERS}
    for i in range(n_sess):
        rs = np.random.default_rng(seed + i)
        sess = T.make_session(rs, n_targets=N_TARGETS)
        n = len(sess["intended"])
        if tremor_fn is None:
            raw, _ = T.add_tremor_to_session(sess, amp, rs)
        else:
            raw = sess["intended"] + tremor_fn(n, amp, rs) + 0.3 * rs.standard_normal((n, 2))
        radii = [t["radius"] for t in sess["trials"]]
        for f in FILTERS:
            out = f.process(raw)
            h, t, e = T.score_session(out, sess)
            a = acc[f.name]
            a["hit"] += list(h)
            a["time"] += list(t)
            a["err"] += list(e)
            a["rad"] += radii
            a["rmse"].append(T.rms(out - sess["intended"]))
            a["sess_time"].append(t.mean())
    rows = {}
    for name, a in acc.items():
        hit, rad = np.array(a["hit"]), np.array(a["rad"])
        lo, hi = wilson(int(hit.sum()), len(hit))
        st = np.array(a["sess_time"])
        small = hit[rad == 8]
        rows[name] = dict(
            hit_pct=100 * hit.mean(), hit_lo=100 * lo, hit_hi=100 * hi,
            hit_small_pct=100 * small.mean() if len(small) else float("nan"),
            time_s=float(np.mean(a["time"])),
            time_ci=float(1.96 * st.std(ddof=1) / np.sqrt(len(st))),
            click_err_px=float(np.mean(a["err"])),
            path_rmse_px=float(np.mean(a["rmse"])), n_trials=len(hit))
    return rows


# ================================================================= B: global metrics
def global_metrics():
    out = {}
    for f in FILTERS:
        lags = [T.lag_ms(f, np.random.default_rng(300 + i)) for i in range(3)]
        dist = [T.distortion_px(f, np.random.default_rng(310 + i)) for i in range(3)]
        still = {amp: float(np.mean([T.still_residual_pct(f, amp, np.random.default_rng(320 + i))
                                     for i in range(4)])) for amp in (4.0, 8.0, 12.0)}
        out[f.name] = dict(lag_ms=float(np.mean(lags)), healthy_distortion_px=float(np.mean(dist)),
                           still_pct=still)
    return out


# ================================================================= C: trade-off sweep
def tradeoff():
    pts = {"lp": [], "kal": []}
    for fc in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        f = T.LowPass(fc)
        pts["lp"].append((fc, T.lag_ms(f, np.random.default_rng(3)),
                          float(np.mean([T.still_residual_pct(f, 8, np.random.default_rng(40 + i)) for i in range(4)]))))
    for q in (10, 30, 100, 300, 1000, 1e4, 1e5):
        f = T.KalmanCV(q=q, r=25.0)
        pts["kal"].append((q, T.lag_ms(f, np.random.default_rng(3)),
                           float(np.mean([T.still_residual_pct(f, 8, np.random.default_rng(40 + i)) for i in range(4)]))))
    return pts


# ================================================================= D: robustness
def trem_fixed(f0, harmonic=0.12):
    return lambda n, amp, rs: T.make_tremor(n, f0, amp, rs, harmonic=harmonic)


def trem_bursts(n, amp, rs):
    base = T.make_tremor(n, rs.uniform(4, 7), amp * 1.3, rs)
    gate = 1.0 / (1.0 + np.exp(-4.0 * T._slow_noise(n, 1.0, rs)))
    return base * gate[:, None]


def still_from(trem_fn, f, amp, seeds=4, n=6000):
    v = []
    for i in range(seeds):
        rs = np.random.default_rng(600 + i)
        v.append(T.still_residual_pct(f, amp, rs, trem=trem_fn(n, amp, rs)))
    return float(np.mean(v))


def robustness():
    freqs = [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
    curve = {f.name: [still_from(trem_fixed(fq), f, 8.0) for fq in freqs] for f in (LP, KAL, AI_NN)}
    conds = [("Typical tremor 4-7 Hz (as trained)", None, 8.0),
             ("Slow tremor 3 Hz", trem_fixed(3.0), 8.0),
             ("Fast tremor 9.5 Hz", trem_fixed(9.5), 8.0),
             ("Sharp, non-sinusoidal shake", trem_fixed(5.5, harmonic=0.5), 8.0),
             ("Intermittent bursts", trem_bursts, 8.0),
             ("Very large tremor 20 px (training max 16)", None, 20.0)]
    table = []
    for label, fn, amp in conds:
        ct = run_clicktest(amp, max(6, N_SESS // 3), TEST_SEED + 500, tremor_fn=fn)
        fn_still = fn if fn is not None else (lambda n, a, rs: T.make_tremor(n, rs.uniform(4, 7), a, rs))
        row = {"condition": label}
        for f in FILTERS:
            row[f.name] = dict(hit_pct=ct[f.name]["hit_pct"], still_pct=still_from(fn_still, f, amp))
        table.append(row)
    return freqs, curve, table


# ================================================================= figures
def fig_example(plt):
    rs = np.random.default_rng(7001)
    sess = T.make_session(rs, n_targets=8)
    raw, _ = T.add_tremor_to_session(sess, 8.0, rs, f0=5.2)
    outs = {f.name: f.process(raw) for f in FILTERS}
    tr = next((t for t in sess["trials"] if t["radius"] <= 12), sess["trials"][2])
    a, b = tr["t0"], tr["t_arr"] + 100
    t = (np.arange(a, b) - a) / T.FS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    show = [NOF, LP, KAL, AI_NN]
    ax[0].plot(t, sess["intended"][a:b, 0], color=INK, lw=1.0, ls=(0, (4, 2)), label="What the person intends", zorder=5)
    for f in show:
        ax[0].plot(t, outs[f.name][a:b, 0], color=COL[f.name], lw=1.6 if f is not NOF else 1.0, label=f.name, zorder=3)
    ax[0].set_xlabel("time since the target appeared (s)")
    ax[0].set_ylabel("cursor x position (px)")
    ax[0].set_title("A whole reach: low-pass and Kalman trail the hand", loc="left", fontsize=10, color=INK)
    ax[0].legend(loc="lower right", fontsize=8)
    c, r = tr["center"][0], tr["radius"]
    a2, b2 = tr["t_arr"], tr["t_arr"] + 90
    t2 = (np.arange(a2, b2) - a2) / T.FS
    ax[1].axhspan(-r, r, color=GRID, alpha=0.9, lw=0, label=f"target ({2*r:g} px wide)")
    ax[1].axhline(r, color=INK2, lw=0.8, ls=":")
    ax[1].axhline(-r, color=INK2, lw=0.8, ls=":")
    for f in show:
        ax[1].plot(t2, outs[f.name][a2:b2, 0] - c, color=COL[f.name], lw=1.6 if f is not NOF else 1.0, label=f.name)
    ax[1].set_ylim(-28, 28)
    ax[1].set_xlabel("time after the hand arrives (s)")
    ax[1].set_ylabel("cursor x relative to target centre (px)")
    ax[1].set_title("Hovering: the shaky cursor leaves the target", loc="left", fontsize=10, color=INK)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_example_traces.png", dpi=160)
    plt.close(fig)


def fig_cloud(plt):
    pos = {f.name: [] for f in FILTERS}
    for i in range(8 if QUICK else 15):
        rs = np.random.default_rng(TEST_SEED + 900 + i)
        sess = T.make_session(rs, n_targets=N_TARGETS)
        raw, _ = T.add_tremor_to_session(sess, 8.0, rs)
        for f in (NOF, LP, KAL, AI_NN):
            out = f.process(raw)
            for tr in sess["trials"]:
                seg = out[tr["t_arr"] + 30: tr["t_arr"] + 90] - tr["center"]
                pos[f.name].append(seg)
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2), sharex=True, sharey=True)
    for ax, f in zip(axs, (NOF, LP, KAL, AI_NN)):
        P = np.vstack(pos[f.name])
        inside = (np.linalg.norm(P, axis=1) <= 8).mean() * 100
        ax.scatter(P[:, 0], P[:, 1], s=2, color=COL[f.name], alpha=0.25, lw=0, rasterized=True)
        ax.add_patch(plt.Circle((0, 0), 8, fill=False, ec=INK, lw=1.0))
        ax.set_aspect("equal")
        ax.set_xlim(-22, 22)
        ax.set_ylim(-22, 22)
        ax.set_title(f"{f.name}\n{inside:.0f} % of the time on the target", fontsize=9, color=INK)
        ax.grid(False)
    axs[0].set_ylabel("y offset (px)")
    for ax in axs:
        ax.set_xlabel("x offset (px)")
    fig.suptitle("Where the cursor sits while hovering over the smallest target (black circle, 16 px wide), "
                 "moderate tremor; 0.3-0.9 s after arrival", fontsize=9.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fig2_hover_cloud.png", dpi=160)
    plt.close(fig)


def fig_spectrum(plt):
    acc = {f.name: [] for f in (NOF, LP, KAL, AI_NN)}
    for i in range(10 if QUICK else 20):
        rs = np.random.default_rng(TEST_SEED + 1200 + i)
        sess = T.make_session(rs, n_targets=N_TARGETS)
        raw, _ = T.add_tremor_to_session(sess, 8.0, rs)
        for f in (NOF, LP, KAL, AI_NN):
            err = f.process(raw) - sess["intended"]
            fr, P = signal.welch(err[:, 0], fs=T.FS, nperseg=256)
            acc[f.name].append(P)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axvspan(4, 7, color=GRID, alpha=0.7, lw=0)
    ax.text(5.5, 0.9, "tremor band", ha="center", va="top", fontsize=8, color=INK2, transform=ax.get_xaxis_transform())
    for f in (NOF, LP, KAL, AI_NN):
        ax.semilogy(fr, np.mean(acc[f.name], axis=0), color=COL[f.name], lw=1.6, label=f.name)
    ax.set_xlim(0.3, 15)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("error power (px$^2$/Hz)")
    ax.set_title("Cursor error versus intended path, by frequency", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_error_spectrum.png", dpi=160)
    plt.close(fig)


def fig_hits(plt, click):
    labels = [s for s, a in SEVERITIES if a > 0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.16
    x = np.arange(len(labels))
    for j, f in enumerate(FILTERS):
        vals = [click[s][f.name]["hit_pct"] for s in labels]
        xs = x + (j - 2) * (w + 0.015)
        ax.bar(xs, vals, w, color=COL[f.name], label=f.name, lw=0)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("clicks that hit the target (%)")
    ax.set_title("Simulated click test: share of clicks that land on the target", loc="left", fontsize=10, color=INK)
    ax.legend(ncol=5, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_hit_rate.png", dpi=160)
    plt.close(fig)


def fig_tradeoff(plt, glob, tr):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    lp = np.array([(l, s) for _, l, s in tr["lp"]])
    kal = np.array([(l, s) for _, l, s in tr["kal"]])
    ax.plot(lp[:, 0], lp[:, 1], "-o", color=COL[LP.name], lw=1.6, ms=5, label="Low-pass (cut-off 1.5 to 6 Hz)")
    for (fc, l, s) in tr["lp"]:
        if fc in (1.5, 2.5, 4.0, 6.0):
            ax.annotate(f"{fc:g} Hz", (l, s), textcoords="offset points", xytext=(6, 5), fontsize=7.5, color=INK2)
    ax.plot(kal[:, 0], kal[:, 1], "-s", color=COL[KAL.name], lw=1.6, ms=5, label="Kalman filter (different settings)")
    for nm, p in ((AI_LIN, "AI linear"), (AI_NN, "AI neural net")):
        g = glob[nm.name]
        ax.plot(g["lag_ms"], g["still_pct"][8.0], "D", color=COL[nm.name], ms=9 if nm is AI_NN else 7,
                mec=SURFACE, mew=1.5, label=nm.name, zorder=5)
    ax.plot(0, 100, "o", color=MUTED, ms=7, label="No filter", zorder=5)
    ax.annotate("better: low lag, little tremor", xy=(4, 2), xytext=(28, 5), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1), va="center")
    ax.set_xlabel("lag while moving (ms)  -  lower is better")
    ax.set_ylabel("tremor left when holding still (%)  -  lower is better")
    ax.set_xlim(-8, 175)
    ax.set_ylim(-4, 108)
    ax.set_title("The trade-off: less tremor usually costs lag, except for the AI model", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_tradeoff.png", dpi=160)
    plt.close(fig)


def fig_lagcost(plt, click):
    s = "Moderate (8 px)"
    vals = [(f.name, click[s][f.name]["path_rmse_px"]) for f in FILTERS]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(vals))[::-1]
    for yy, (nm, v) in zip(y, vals):
        ax.barh(yy, v, 0.55, color=COL[nm], lw=0)
        ax.text(v + 0.8, yy, f"{v:.1f} px", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [v[0] for v in vals])
    ax.set_xlabel("average distance between displayed cursor and where the hand wants it (px)")
    ax.set_title("Cost of lag: the smoothing filters leave the cursor behind the hand", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v in vals) * 1.18)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_path_error.png", dpi=160)
    plt.close(fig)


def fig_robust(plt, freqs, curve):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvspan(3.5, 8.0, color=GRID, alpha=0.7, lw=0)
    ax.text(5.75, 0.97, "frequencies the AI saw in training", ha="center", va="top", fontsize=8,
            color=INK2, transform=ax.get_xaxis_transform())
    for nm, vals in curve.items():
        ax.plot(freqs, vals, "-o", color=COL[nm], lw=1.6, ms=4, label=nm)
    ax.set_xlabel("tremor frequency (Hz)")
    ax.set_ylabel("tremor left when holding still (%)")
    ax.set_ylim(0, max(105, max(max(v) for v in curve.values()) * 1.08))
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    ax.text(12, 101, "100 % = no filter", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.set_title("Robustness: the AI model works inside its trained range only", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.9))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_robustness.png", dpi=160)
    plt.close(fig)


# ================================================================= reporting
def write_reports(click, glob, trade, freqs, curve, ood):
    with open(f"{OUT}/results_clicktest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tremor", "filter", "hit_pct", "hit_ci_low", "hit_ci_high", "hit_pct_smallest_targets",
                    "mean_time_s", "time_ci95_s", "mean_click_error_px", "path_error_px", "n_trials"])
        for s, _ in SEVERITIES:
            for f in FILTERS:
                r = click[s][f.name]
                w.writerow([s, f.name, f"{r['hit_pct']:.1f}", f"{r['hit_lo']:.1f}", f"{r['hit_hi']:.1f}",
                            f"{r['hit_small_pct']:.1f}", f"{r['time_s']:.3f}", f"{r['time_ci']:.3f}",
                            f"{r['click_err_px']:.2f}", f"{r['path_rmse_px']:.2f}", r["n_trials"]])
    with open(f"{OUT}/results.json", "w") as fh:
        json.dump(dict(click=click, global_metrics=glob, tradeoff=trade, robustness_curve=dict(freqs=freqs, **curve),
                       robustness_table=ood, n_sessions=N_SESS, targets_per_session=N_TARGETS), fh, indent=1)

    L = []
    L.append("# Benchmark results (auto-generated by benchmark.py)\n")
    L.append(f"Test data: {N_SESS} simulated sessions x {N_TARGETS} targets per tremor level "
             f"= {N_SESS*N_TARGETS} clicks per filter and level. Test seeds were never used for training.\n")
    L.append("## A. Simulated click test\n")
    for s, _ in SEVERITIES:
        L.append(f"**{s}**\n")
        L.append("| Filter | Hit rate | 95% CI | Hit rate, smallest targets | Time per click | Click error | Path error while moving |")
        L.append("|---|---|---|---|---|---|---|")
        for f in FILTERS:
            r = click[s][f.name]
            L.append(f"| {f.name} | {r['hit_pct']:.1f} % | {r['hit_lo']:.1f}-{r['hit_hi']:.1f} | {r['hit_small_pct']:.1f} % | "
                     f"{r['time_s']:.2f} s | {r['click_err_px']:.1f} px | {r['path_rmse_px']:.1f} px |")
        L.append("")
    L.append("## B. Lag, damage to a healthy hand, tremor left when holding still\n")
    L.append("| Filter | Lag while moving | Distortion of a healthy hand | Tremor left (4 px) | (8 px) | (12 px) |")
    L.append("|---|---|---|---|---|---|")
    for f in FILTERS:
        g = glob[f.name]
        L.append(f"| {f.name} | {g['lag_ms']:.0f} ms | {g['healthy_distortion_px']:.2f} px | "
                 f"{g['still_pct'][4.0]:.0f} % | {g['still_pct'][8.0]:.0f} % | {g['still_pct'][12.0]:.0f} % |")
    L.append("")
    L.append("## D. Robustness (tremor the AI did not see in training); hit rate / tremor left when still\n")
    L.append("| Condition | " + " | ".join(NAMES) + " |")
    L.append("|---|" + "---|" * len(NAMES))
    for row in ood:
        L.append(f"| {row['condition']} | " + " | ".join(
            f"{row[n]['hit_pct']:.0f} % / {row[n]['still_pct']:.0f} %" for n in NAMES) + " |")
    L.append("")
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(L))


if __name__ == "__main__":
    t0 = time.time()
    print(f"[{'quick' if QUICK else 'full'} run] sessions per level: {N_SESS}")
    click = {}
    for s, amp in SEVERITIES:
        click[s] = run_clicktest(amp, N_SESS, TEST_SEED)
        print(f"  A  {s:16s} " + "  ".join(f"{n}: {click[s][n]['hit_pct']:.0f}%" for n in NAMES) + f"   ({time.time()-t0:.0f}s)")
    glob = global_metrics()
    print("  B  done", f"({time.time()-t0:.0f}s)")
    trade = tradeoff()
    print("  C  done", f"({time.time()-t0:.0f}s)")
    freqs, curve, ood = robustness()
    print("  D  done", f"({time.time()-t0:.0f}s)")
    plt = mpl()
    fig_example(plt)
    fig_cloud(plt)
    fig_spectrum(plt)
    fig_hits(plt, click)
    fig_tradeoff(plt, glob, trade)
    fig_lagcost(plt, click)
    fig_robust(plt, freqs, curve)
    write_reports(click, glob, trade, freqs, curve, ood)
    print(f"All done in {time.time()-t0:.0f}s  ->  ./{OUT}/")
