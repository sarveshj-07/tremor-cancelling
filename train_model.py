"""
train_model.py  -  trains the two "AI" tremor estimators on SIMULATED data
=========================================================================

 * AI linear : ridge regression (a learned FIR filter)
 * AI neural : small MLP 40 -> 64 -> 32 -> 1 (tanh), trained with Adam (pure numpy)

Input  : the last 0.4 s of the tremor-band (3-10 Hz) signal of one axis
Target : the TRUE tremor value at the current moment (known, because we simulate it)
Output : estimated tremor  ->  cursor = raw input - estimated tremor

Run:   python train_model.py            (about 1-3 minutes, CPU only)
Makes: model.npz   (used by benchmark.py)
       model.json  (used by the browser demo)
       training_curve.png
"""
import json
import sys
import time

import numpy as np

import tremor_lib as T

TRAIN_SEED, VAL_SEED = 1000, 2000          # test data uses other seeds (see benchmark.py)


def build_dataset(n_seq, seed, seq_len=3000, stride=3):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n_seq):
        raw, trem, _ = T.make_training_sequence(seq_len, rng)
        for ax in range(2):
            W = T.bandpass_windows(raw[:, ax])
            off = int(rng.integers(0, stride))
            Xs.append(W[off::stride])
            ys.append(trem[off::stride, ax].astype(np.float32))
    return np.vstack(Xs), np.concatenate(ys)


def fit_ridge(X, y, lam=1e-2):
    X = X.astype(np.float64)
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y.astype(np.float64))


def train_nn(Xtr, ytr, Xva, yva, hidden=(64, 32), epochs=30, batch=1024,
             lr0=3e-3, lr1=1e-4, seed=0, log=print):
    rng = np.random.default_rng(seed)
    Ftr, Fva = T.nn_features(Xtr), T.nn_features(Xva)
    ttr = (ytr / T.OUT_SCALE).astype(np.float32)
    tva = (yva / T.OUT_SCALE).astype(np.float32)
    sizes = [Ftr.shape[1], hidden[0], hidden[1], 1]
    P = {}
    for i in range(3):
        lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
        P[f"W{i+1}"] = rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])).astype(np.float32)
        P[f"b{i+1}"] = np.zeros(sizes[i + 1], np.float32)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    step = 0

    def forward(F, P):
        h1 = np.tanh(F @ P["W1"] + P["b1"])
        h2 = np.tanh(h1 @ P["W2"] + P["b2"])
        return h1, h2, (h2 @ P["W3"] + P["b3"])[:, 0]

    best, best_P = 1e9, None
    n = len(Ftr)
    for ep in range(epochs):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + np.cos(np.pi * ep / max(1, epochs - 1)))
        perm = rng.permutation(n)
        tl = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Ftr[idx], ttr[idx]
            h1, h2, out = forward(xb, P)
            err = out - yb
            tl += float((err ** 2).sum())
            g = (2.0 / len(idx)) * err[:, None]
            G = {}
            G["W3"] = h2.T @ g
            G["b3"] = g.sum(0)
            gz2 = (g @ P["W3"].T) * (1 - h2 ** 2)
            G["W2"] = h1.T @ gz2
            G["b2"] = gz2.sum(0)
            gz1 = (gz2 @ P["W2"].T) * (1 - h1 ** 2)
            G["W1"] = xb.T @ gz1
            G["b1"] = gz1.sum(0)
            step += 1
            for k in P:
                m[k] = b1c * m[k] + (1 - b1c) * G[k]
                v2[k] = b2c * v2[k] + (1 - b2c) * G[k] ** 2
                mh = m[k] / (1 - b1c ** step)
                vh = v2[k] / (1 - b2c ** step)
                P[k] = (P[k] - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        _, _, vo = forward(Fva, P)
        val_mse_px = float(np.mean((vo - tva) ** 2)) * T.OUT_SCALE ** 2
        log(f"  epoch {ep+1:2d}/{epochs}  train MSE {tl/n*T.OUT_SCALE**2:7.3f} px^2   "
            f"val MSE {val_mse_px:7.3f} px^2   lr {lr:.5f}")
        hist.append((ep + 1, tl / n * T.OUT_SCALE ** 2, val_mse_px))
        if val_mse_px < best:
            best, best_P = val_mse_px, {k: v.copy() for k, v in P.items()}
    return best_P, best


hist = []

if __name__ == "__main__":
    t0 = time.time()
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print(f"Generating training data ({n_train} simulated 30 s recordings)...")
    Xtr, ytr = build_dataset(n_train, TRAIN_SEED)
    Xva, yva = build_dataset(60, VAL_SEED)
    print(f"  train windows: {len(Xtr):,}   validation windows: {len(Xva):,}   ({time.time()-t0:.0f}s)")
    var = float(np.mean(yva ** 2))
    print(f"  validation tremor variance (= error if we predict 'no tremor'): {var:.3f} px^2")

    h = fit_ridge(Xtr, ytr)
    mse_lin = float(np.mean((Xva.astype(np.float64) @ h - yva) ** 2))
    print(f"AI linear (ridge)  validation MSE: {mse_lin:.3f} px^2  -> explains {100*(1-mse_lin/var):.1f} % of the tremor")

    print("Training neural network...")
    P, best = train_nn(Xtr, ytr, Xva, yva)
    print(f"AI neural net      validation MSE: {best:.3f} px^2  -> explains {100*(1-best/var):.1f} % of the tremor")

    np.savez("model.npz", ridge=h, **P)
    js = {"win": T.WIN, "in_scale": T.IN_SCALE, "in_clip": T.IN_CLIP, "out_scale": T.OUT_SCALE,
          "fs": T.FS,
          "bp_sos": T.BP_SOS.tolist(),
          "bp_zi": __import__("scipy.signal", fromlist=["sosfilt_zi"]).sosfilt_zi(T.BP_SOS).tolist(),
          "ridge": h.tolist(),
          **{k: v.astype(float).tolist() for k, v in P.items()}}
    with open("model.json", "w") as f:
        json.dump(js, f)
    print(f"Saved model.npz and model.json   (total {time.time()-t0:.0f}s)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = np.array(hist)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(e[:, 0], e[:, 1], label="training error", color="#1f77b4")
        ax.plot(e[:, 0], e[:, 2], label="validation error (unseen data)", color="#d62728")
        ax.axhline(var, ls=":", color="gray", label="no model (predict zero tremor)")
        ax.axhline(mse_lin, ls="--", color="#2ca02c", label="AI linear")
        ax.set_xlabel("epoch")
        ax.set_ylabel("tremor-estimate error (px$^2$)")
        ax.set_title("Neural network learning curve")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("training_curve.png", dpi=160)
    except Exception as ex:                       # plotting must never break training
        print("plot skipped:", ex)
