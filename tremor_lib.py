"""
tremor_lib.py  -  core engine of the Tremor-Cancelling Cursor project
=====================================================================

Everything that is shared by training, benchmarking and the tests lives here:

  1. A TREMOR SIMULATOR       (synthetic 4-7 Hz hand shake, elliptical, drifting)
  2. A SIMULATED USER         (human-like reaching movements, Fitts-law timing,
                               corrective sub-movements, "wait until the cursor
                               looks settled, then press" click behaviour)
  3. FILTERS (all causal = usable live, they never look into the future)
       - NoFilter
       - LowPass          classic Butterworth low-pass (baseline #1)
       - KalmanCV         constant-velocity Kalman filter (baseline #2)
       - LearnedLinear    ridge-regression tremor estimator (trained on simulation)
       - LearnedNN        small neural network tremor estimator (trained on simulation)
  4. METRICS

Only numpy + scipy are needed.
"""
import numpy as np
from scipy import signal

FS = 100.0                      # simulation / filter sample rate in Hz
SCREEN = (1280.0, 720.0)        # virtual screen in pixels


# =====================================================================
# 1. TREMOR SIMULATOR
# =====================================================================
def _slow_noise(n, tau_s, rng, fs=FS):
    """Unit-variance, slowly varying random signal (AR(1) / Ornstein-Uhlenbeck)."""
    a = np.exp(-1.0 / (tau_s * fs))
    burn = int(6 * tau_s * fs)
    w = rng.standard_normal(n + burn)
    y = signal.lfilter([np.sqrt(1.0 - a * a)], [1.0, -a], w)
    return y[burn:]


def make_tremor(n, f0, amp, rng, fs=FS, harmonic=0.12,
                f_wander=0.4, amp_wander=0.25):
    """
    Synthetic physiological tremor, shape (n, 2) in pixels.

    f0        centre frequency in Hz (Parkinson's rest tremor ~4-6 Hz,
              essential tremor ~4-8 Hz)
    amp       peak amplitude of the x-axis shake in pixels (y gets 50-100 %)
    The frequency and amplitude slowly drift (like a real tremor), the
    two axes are phase-shifted (elliptical shake) and a weak 2nd harmonic
    is added.
    """
    fi = np.clip(f0 + f_wander * _slow_noise(n, 2.0, rng, fs), 2.5, 12.0)
    phase = 2 * np.pi * np.cumsum(fi) / fs + rng.uniform(0, 2 * np.pi)
    ratio_y = rng.uniform(0.5, 1.0)
    dphi = rng.uniform(0, 2 * np.pi)
    psi = rng.uniform(0, 2 * np.pi, size=2)
    out = np.empty((n, 2))
    for k, (A, ph0) in enumerate(((amp, 0.0), (amp * ratio_y, dphi))):
        a_t = np.clip(1.0 + amp_wander * _slow_noise(n, 1.5, rng, fs), 0.2, 2.0)
        out[:, k] = A * a_t * (np.sin(phase + ph0) + harmonic * np.sin(2 * phase + psi[k]))
    return out


# =====================================================================
# 2. HUMAN MOVEMENT (the INTENDED path of the cursor, without tremor)
# =====================================================================
def min_jerk(p0, p1, T, fs=FS):
    """Minimum-jerk trajectory (classic model of human reaching), excludes p0."""
    n = max(2, int(round(T * fs)))
    tau = np.arange(1, n + 1) / n
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    return p0[None, :] + (p1 - p0)[None, :] * s[:, None]


def fitts_time(D, W, rng):
    """Movement time (s) from Fitts's law:  MT = a + b*log2(D/W + 1)  (+ variation)."""
    return (0.10 + 0.12 * np.log2(D / W + 1.0)) * rng.uniform(0.8, 1.2)


def _flick(pos, rng, fs, lo, hi):
    """A fast mouse flick (80-900 px in 0.08-0.5 s, sometimes a quick return) followed by a hold.
    Sudden stops look a bit like tremor to a filter, so the learned filters are trained on them too."""
    ang = rng.uniform(0, 2 * np.pi)
    tgt = np.clip(pos + rng.uniform(80, 900) * np.array([np.cos(ang), np.sin(ang)]), lo, hi)
    n = max(2, int(round(rng.uniform(0.08, 0.5) * fs)))
    tau = np.arange(1, n + 1) / n
    kind = int(rng.integers(3))                        # constant speed / minimum jerk / decelerating
    s = tau if kind == 0 else (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5 if kind == 1 else 1 - (1 - tau) ** 2)
    parts = [pos + (tgt - pos) * s[:, None]]
    if rng.random() < 0.2:                             # quick return
        n2 = max(2, int(round(rng.uniform(0.08, 0.4) * fs)))
        t2 = np.arange(1, n2 + 1) / n2
        parts.append(tgt + (pos - tgt) * t2[:, None])
        tgt = pos
    parts.append(np.repeat(tgt[None, :], int(rng.uniform(0.2, 1.5) * fs), axis=0))
    return np.vstack(parts)


def make_intention(n, rng, fs=FS, screen=SCREEN, p_flick=0.0):
    """
    Diverse, human-like intended motion of length n, shape (n, 2).
    Mixture of: holding still, reaches with corrective sub-movements and
    smooth wandering (+ fast flicks with probability p_flick). Used to TRAIN the
    learned filters so that they do not only see one kind of movement.
    p_flick=0 (default) is the plain mixture used for the lag / distortion metrics.
    """
    lo = np.array([30.0, 30.0])
    hi = np.array([screen[0] - 30.0, screen[1] - 30.0])
    pos = rng.uniform(lo, hi)
    chunks, total = [], 0
    while total < n:
        if p_flick > 0 and rng.random() < p_flick:     # fast flick
            seg = _flick(pos, rng, fs, lo, hi)
            chunks.append(seg)
            pos = seg[-1]
            total += len(seg)
            continue
        u = rng.random()
        if u < 0.25:                                   # hold still
            m = max(2, int(rng.uniform(0.3, 2.0) * fs))
            seg = np.repeat(pos[None, :], m, axis=0)
        elif u < 0.75:                                 # reach + correction
            tgt = rng.uniform(lo, hi)
            D = float(np.linalg.norm(tgt - pos)) + 1e-9
            W = rng.uniform(16, 48)
            p1 = np.clip(tgt + rng.normal(0, 0.05 * D, 2), lo, hi)
            seg1 = min_jerk(pos, p1, fitts_time(D, W, rng), fs)
            pause = np.repeat(p1[None, :], int(rng.uniform(0.05, 0.2) * fs), axis=0)
            seg2 = min_jerk(p1, tgt, rng.uniform(0.1, 0.25), fs)
            seg = np.vstack([seg1, pause, seg2])
        else:                                          # smooth wandering
            m = max(2, int(rng.uniform(0.5, 2.0) * fs))
            speed = rng.uniform(20, 300)
            v = np.stack([_slow_noise(m, rng.uniform(0.2, 0.5), rng, fs)
                          for _ in range(2)], axis=1) * speed
            seg = np.clip(pos[None, :] + np.cumsum(v, axis=0) / fs, lo, hi)
        chunks.append(seg)
        pos = seg[-1]
        total += len(seg)
    return np.vstack(chunks)[:n]


def make_training_sequence(n, rng, fs=FS):
    """One training sequence: returns raw (n,2), tremor label (n,2), intention (n,2)."""
    inten = make_intention(n, rng, fs, p_flick=0.35)
    if rng.random() < 0.10:                            # 10 % healthy hands -> must do no harm
        amp = 0.0
    else:
        amp = rng.uniform(0.5, 16.0)
    f0 = rng.uniform(3.5, 8.0)
    trem = make_tremor(n, f0, amp, rng, fs) if amp > 0 else np.zeros((n, 2))
    noise = 0.3 * rng.standard_normal((n, 2))          # sensor / micro jitter
    return inten + trem + noise, trem, inten


# =====================================================================
# 3. SIMULATED CLICK TEST ("the user")
# =====================================================================
def make_session(rng, n_targets=20, radii=(8, 12, 18, 24), fs=FS,
                 screen=SCREEN, dwell_s=1.3):
    """
    A sequence of n_targets click tasks, described by the INTENDED cursor path.
    Each task: reaction pause -> fast aimed movement (5 % endpoint scatter)
    -> short pause -> corrective sub-movement (1.2 px scatter) -> dwell.
    Target radii are in pixels (8 px radius = a 16 px wide button).
    """
    lo = np.array([60.0, 60.0])
    hi = np.array([screen[0] - 60.0, screen[1] - 60.0])
    pos = rng.uniform(lo, hi)
    first = np.repeat(pos[None, :], int(0.5 * fs), axis=0)
    chunks, idx = [first], len(first)
    trials = []
    for _ in range(n_targets):
        radius = float(rng.choice(radii))
        for _try in range(200):
            tgt = rng.uniform(lo, hi)
            D = float(np.linalg.norm(tgt - pos))
            if 200.0 <= D <= 900.0:
                break
        t0 = idx
        hold = np.repeat(pos[None, :], int(rng.uniform(0.25, 0.45) * fs), axis=0)
        p1 = np.clip(tgt + rng.normal(0, 0.05 * D, 2), [10, 10],
                     [screen[0] - 10, screen[1] - 10])
        seg1 = min_jerk(pos, p1, fitts_time(D, 2 * radius, rng), fs)
        pause = np.repeat(p1[None, :], int(rng.uniform(0.08, 0.18) * fs), axis=0)
        end = tgt + rng.normal(0, 1.2, 2)
        seg2 = min_jerk(p1, end, rng.uniform(0.12, 0.25), fs)
        dwell = np.repeat(end[None, :], int(dwell_s * fs), axis=0)
        press = float(rng.normal(0.10, 0.02))          # finger press delay (s)
        for c in (hold, seg1, pause, seg2):
            idx += len(c)
        t_arr = idx - 1                                # last sample of the movement
        idx += len(dwell)
        chunks += [hold, seg1, pause, seg2, dwell]
        trials.append(dict(center=tgt, radius=radius, t0=t0, t_arr=t_arr, press=press))
        pos = end
    return dict(intended=np.vstack(chunks), trials=trials)


def add_tremor_to_session(session, amp, rng, f0=None, fs=FS, noise_px=0.3):
    """Return the raw (shaky) input = intended + tremor + small noise."""
    inten = session["intended"]
    n = len(inten)
    if f0 is None:
        f0 = rng.uniform(4.0, 7.0)
    trem = make_tremor(n, f0, amp, rng, fs) if amp > 0 else np.zeros((n, 2))
    return inten + trem + noise_px * rng.standard_normal((n, 2)), trem


def score_session(displayed, session, fs=FS, settle_s=0.06, timeout_s=0.9):
    """
    Simulated clicking behaviour on the DISPLAYED cursor (the output of a filter):
      * once the intended movement is over, the user waits until the displayed
        cursor has been inside the target for `settle_s` seconds,
      * (gives up after `timeout_s` and presses anyway),
      * then presses after a human delay (~100 ms). The click lands wherever
        the cursor is at that moment -> hit if inside the target circle.
    Returns arrays: hit(bool), time(s) from target appearance to click, error(px).
    """
    s = int(round(settle_s * fs))
    to = int(round(timeout_s * fs))
    hits, times, errs = [], [], []
    for tr in session["trials"]:
        c, r, a = tr["center"], tr["radius"], tr["t_arr"]
        seg = displayed[a:a + to + 1]
        inside = np.linalg.norm(seg - c, axis=1) <= r
        run, j_set = 0, to
        for j, ins in enumerate(inside):
            run = run + 1 if ins else 0
            if run >= s:
                j_set = j
                break
        tc = a + j_set + max(3, int(round(tr["press"] * fs)))
        err = float(np.linalg.norm(displayed[tc] - c))
        hits.append(err <= r)
        times.append((tc - tr["t0"]) / fs)
        errs.append(err)
    return np.array(hits), np.array(times), np.array(errs)


# =====================================================================
# 4. FILTERS  (every one implements  .process(raw[n,2]) -> out[n,2],  causal)
# =====================================================================
def _sosfilt_axes(sos, x):
    """Causal sos filtering of each column, initial state = steady state for x[0]."""
    zi0 = signal.sosfilt_zi(sos)
    out = np.empty_like(x, dtype=float)
    for k in range(x.shape[1]):
        out[:, k], _ = signal.sosfilt(sos, x[:, k], zi=zi0 * x[0, k])
    return out


class NoFilter:
    name = "No filter"

    def process(self, raw):
        return raw.copy()


class LowPass:
    """Butterworth low-pass: removes everything faster than fc (also fast intended motion -> lag)."""

    def __init__(self, fc=2.0, order=2, fs=FS):
        self.fc, self.order, self.fs = fc, order, fs
        self.sos = signal.butter(order, fc, btype="low", fs=fs, output="sos")
        self.name = f"Low-pass {fc:g} Hz"

    def process(self, raw):
        return _sosfilt_axes(self.sos, raw)


class KalmanCV:
    """
    Constant-velocity Kalman filter. State = [position, velocity] per axis.
    q  = how much sudden acceleration (intended motion) we allow
    r  = measurement noise variance (px^2) -> the tremor is treated as noise
    """

    def __init__(self, q=2e5, r=25.0, fs=FS):
        self.q, self.r, self.fs = q, r, fs
        self.name = "Kalman filter"

    def process(self, raw):
        dt = 1.0 / self.fs
        q, r = self.q, self.r
        dt2, dt3 = dt * dt, dt ** 3
        out = np.empty_like(raw, dtype=float)
        for k in range(raw.shape[1]):
            z = raw[:, k]
            p, v = float(z[0]), 0.0
            P00, P01, P11 = r, 0.0, 1e4
            o = out[:, k]
            for i in range(len(z)):
                # predict
                p = p + dt * v
                P00 = P00 + 2 * dt * P01 + dt2 * P11 + q * dt3 / 3.0
                P01 = P01 + dt * P11 + q * dt2 / 2.0
                P11 = P11 + q * dt
                # update
                S = P00 + r
                K0, K1 = P00 / S, P01 / S
                innov = z[i] - p
                p += K0 * innov
                v += K1 * innov
                P00, P01, P11 = (1 - K0) * P00, (1 - K0) * P01, P11 - K1 * P01
                o[i] = p
        return out


# ---- learned estimators -------------------------------------------------
BP_SOS = signal.butter(2, [3.0, 10.0], btype="bandpass", fs=FS, output="sos")
WIN = 40            # 0.4 s of history
IN_SCALE = 10.0     # pixels -> network units
IN_CLIP = 6.0
OUT_SCALE = 10.0    # network units -> pixels


def bandpass_windows(x1d, win=WIN):
    """
    Causal features for one axis: tremor-band (3-10 Hz) signal, windows of the
    last `win` samples. Row t holds b[t-win+1 .. t] (zero padded at the start).
    Returns (n, win) float32.
    """
    b, _ = signal.sosfilt(BP_SOS, x1d, zi=signal.sosfilt_zi(BP_SOS) * x1d[0])
    padded = np.concatenate([np.zeros(win - 1), b])
    W = np.lib.stride_tricks.sliding_window_view(padded, win)
    return W.astype(np.float32)


def nn_features(W):
    return np.clip(W / IN_SCALE, -IN_CLIP, IN_CLIP).astype(np.float32)


class LearnedLinear:
    """Ridge-regression FIR on the tremor-band window -> tremor estimate (trained on simulation)."""

    def __init__(self, h):
        self.h = np.asarray(h, float)
        self.name = "AI linear"

    def process(self, raw):
        out = np.empty_like(raw, dtype=float)
        for k in range(raw.shape[1]):
            W = bandpass_windows(raw[:, k])
            out[:, k] = raw[:, k] - W @ self.h
        return out


class LearnedNN:
    """Small neural network (40 -> 64 -> 32 -> 1, tanh) that estimates the tremor from the
    last 0.4 s of tremor-band signal; the estimate is subtracted from the raw input."""

    def __init__(self, params):
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.name = "AI neural net"

    def estimate(self, W):
        X = nn_features(W)
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        return ((h2 @ p["W3"] + p["b3"])[:, 0] * OUT_SCALE).astype(np.float64)

    def process(self, raw):
        out = np.empty_like(raw, dtype=float)
        for k in range(raw.shape[1]):
            out[:, k] = raw[:, k] - self.estimate(bandpass_windows(raw[:, k]))
        return out


# =====================================================================
# 5. METRICS
# =====================================================================
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def lag_ms(filt, rng, fs=FS):
    """Delay of the filter on tremor-free movement (best time shift of the output vs. the input)."""
    inten = make_intention(6000, rng, fs)
    out = filt.process(inten)
    best, best_e = 0, 1e18
    for sh in range(0, 31):                     # 0 .. 300 ms
        e = rms(out[sh:] - inten[:len(inten) - sh]) if sh else rms(out - inten)
        if e < best_e:
            best, best_e = sh, e
    return best * 1000.0 / fs


def distortion_px(filt, rng, fs=FS):
    """RMS difference output-input on tremor-free movement (how much a healthy hand is disturbed)."""
    inten = make_intention(6000, rng, fs)
    return rms(filt.process(inten) - inten)


def still_residual_pct(filt, amp, rng, fs=FS, n=6000, f0=None, trem=None):
    """Residual tremor (% of original RMS) while the hand holds still.
    f0 / trem can be given to test other tremor frequencies or shapes."""
    if trem is None:
        if f0 is None:
            f0 = rng.uniform(4.0, 7.0)
        trem = make_tremor(n, f0, amp, rng, fs)
    n = len(trem)
    base = np.array([[640.0, 360.0]])
    raw = base + trem + 0.3 * rng.standard_normal((n, 2))
    out = filt.process(raw)
    sl = slice(300, None)                        # ignore start-up
    return 100.0 * rms(out[sl] - base) / rms(raw[sl] - base)
