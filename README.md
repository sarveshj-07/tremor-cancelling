# Tremor-Cancelling Cursor

An assistive-technology project for people with hand tremor (Parkinson's disease, essential tremor).
A trained neural network watches the cursor, estimates the tremor in it, and subtracts it, so the
cursor follows the intended movement and not the shake. It adds no delay, and it needs no hardware
beyond an ordinary mouse. Everything runs on a normal laptop and costs nothing.

**Important:** all results come from a *simulation* (simulated tremor, simulated user). No real patients
were involved. See "Limitations" and say so in your presentation.

---

## 1. Fastest way to use it

Double-click **`demo.html`** (Chrome, Edge or Firefox). No installation, no internet, no server.

* **Free play** - click the targets. Press **1** (off), **2** (low-pass), **3** (Kalman), **4** (AI network).
  **F** flips between off and AI. **T** turns the tremor on and off, **G** shows or hides the grey "shaky hand" ring.
* **Challenge** - four rounds on identical targets, one round per filter, in random order. Blind mode hides the
  filter name until the results screen, so nobody is biased. The results screen has a **Download CSV** button.
* Sliders change tremor strength (px) and frequency (Hz). Below 3.5 Hz or above 8 Hz the page warns you that the
  AI was not trained there.

Use a mouse or trackpad. On a touch screen the click lands at the filtered cursor, not under your finger.

## 2. Re-create everything from scratch

```
pip install -r requirements.txt
python run_all.py            # trains the model, runs the benchmark, rebuilds demo.html, runs the tests
```

Takes about two minutes on a laptop. Random seeds are fixed, so you get the same numbers every time.
`python run_all.py --skip-train` reuses the shipped `model.npz`.

| File | What it is |
|---|---|
| `demo.html` | The live demo (single file, everything inside) |
| `tremor_lib.py` | Tremor simulator, simulated user, all filters, metrics |
| `train_model.py` | Trains the two AI estimators on simulated data (pure NumPy) |
| `benchmark.py` | Runs every experiment, writes `results/` (tables, CSV, 7 figures) |
| `model.npz`, `model.json` | The trained model (NumPy / browser format) |
| `filters.js` | The same filters in JavaScript (used by the demo) |
| `export_params.py`, `build_demo.py` | Pack the model into the demo and build `demo.html` |
| `demo_template.html` | Source of the demo page |
| `tests/` | Automated tests (see section 8) |
| `results/` | `RESULTS.md`, `results_clicktest.csv`, `results.json`, `fig1`-`fig7` |

## 3. The idea in one minute

Tremor is a fast, rhythmic shake, roughly 4 to 8 Hz in this project. Deliberate movement ("go to that button")
is slow and smooth. They live at different speeds, so software can separate them.

Three filters are compared against "no filter":

1. **Low-pass filter** (classic baseline, cut-off 2.5 Hz). Removes everything fast. Also removes fast intended movement, so the cursor **lags** behind the hand and cuts corners.
2. **Kalman filter** (classic baseline). Treats tremor as measurement noise. Same lag problem when set to remove as much tremor.
3. **AI neural network** (this project). Looks at the last 0.4 s of tremor-band signal (3-10 Hz) and *estimates the tremor itself*, then subtracts it from the cursor:
   `cursor shown = hand position - estimated tremor`. Because it subtracts the shake instead of smoothing the whole path, slow deliberate motion passes through untouched.

Network: 40 inputs, two hidden layers (64 and 32, tanh), 1 output, about 4,700 weights. Trained with Adam on 400 simulated
30-second recordings (800,000 windows). On unseen validation data it explains **89.7 %** of the tremor.
A plain linear estimator (ridge regression) explains only 53 %, which is why a neural network is justified.

All filters are *causal*: they use only the past, so they could run live.

## 4. Results (simulation, 800 clicks per filter and level)

Moderate tremor (8 px) and severe tremor (12 px), full tables in `results/RESULTS.md`:

| | Hit rate 8 px | Hit rate 12 px | Smallest targets, 12 px | Lag | Disturbance of a healthy hand | Tremor left when still (8 px) |
|---|---|---|---|---|---|---|
| No filter | 89.0 % | 76.9 % | 35.2 % | 0 ms | 0.00 px | 100 % |
| Low-pass 2.5 Hz | 100.0 % | 99.6 % | 98.5 % | **90 ms** | **42.3 px** | 20 % |
| Kalman | 97.9 % | 96.9 % | 88.9 % | **90 ms** | **55.0 px** | 18 % |
| AI linear | 98.9 % | 93.6 % | 77.9 % | 0 ms | 1.8 px | 53 % |
| **AI neural net** | 99.6 % | 99.6 % | 98.5 % | **0 ms** | **0.75 px** | 17 % |

How to read this honestly:

* Any tremor-removing filter hugely beats no filter (12 px tremor: 77 % to about 99.6 % of clicks hit).
* On **hit rate alone the AI is not better than the simple low-pass filter.** The simulated user patiently waits for the cursor to settle before clicking, which hides the cost of lag.
* The AI's advantage is **how** it gets there: **zero lag** (90 ms for the others) and it barely disturbs a hand that is not shaking (0.75 px versus 42 px and 55 px).
  The low-pass cursor trails about 32 px behind the hand while moving, the AI about 1 px (`fig6`). In the live demo you can feel this difference with your own hand, which is the strongest part of the presentation.
* It is also worth saying that the Kalman filter, tuned to remove as much tremor as the low-pass, overshoots after fast moves, which is why its click time is longer (1.85 s versus 1.39 s).

## 5. Limitations (put these on a slide)

1. **Simulated data only.** Tremor is generated (4-7 Hz, drifting, elliptical, with a weak harmonic), not recorded from patients. The test tremor comes from the same generator as the training tremor, so the headline numbers are friendlier than reality.
2. **The AI only works inside the range it was trained on.** Robustness test (`fig7`, `RESULTS.md` section D): for a 9.5 Hz tremor the AI leaves 73 % of the tremor while the low-pass leaves 7 %; at 3 Hz it leaves 61 % (low-pass 58 %, Kalman 34 %). A fix is to train on a wider range, or to adapt to each person's tremor.
3. **Simulated user.** Real people overshoot, hesitate and tire in ways the model does not capture. Real tremor can also change during a task.
4. **Not a medical device.** No clinical testing.
5. **Only horizontal and vertical cursor motion.** No pressure, no drag-and-drop tests, no keyboard or touch input.

## 6. A 3-minute live demo

1. Open `demo.html`, Free play, strength 8 px. Press **1** and try the smallest target: the grey ring and the dot are the same, and it shakes.
2. Press **2** (low-pass): the shake stops, but flick quickly across the screen and the cursor trails behind and overshoots.
3. Press **4** (AI): the shake stops and the cursor stays under your hand. Flip with **F** to show before and after.
4. Switch to **Challenge**, blind, 12 targets, and let a classmate play. The results screen appears at the end.
5. Finish by dragging the frequency slider to 9.5 Hz and showing the warning: this is the limitation, and what you would do next.

## 7. Likely questions from teachers (short answers)

* **Why not just use a low-pass filter?** It is the baseline, and it works for hit rate. But it lags 90 ms and distorts normal movement by 42 px, so the cursor feels sluggish. The AI keeps the hand-to-cursor connection (0 ms, under 1 px).
* **Why is your AI not clearly better on hit rate?** Because the simulated user waits until the cursor settles; lag only costs time and comfort, not accuracy. That is why the lag and distortion tables exist.
* **Is the tremor real?** No, it is simulated to match the published 4-8 Hz range. The next step is validating on recorded tremor data and with volunteers, with ethics approval.
* **Can it run live?** Yes. It is causal, uses 0.4 s of history, and does about 5,000 multiplications per sample, so it runs easily in a browser at 100 Hz.
* **How was it trained?** On 400 simulated 30 s recordings with the true tremor known, so the network learns "given the last 0.4 s, what is the tremor now". Test data uses different random seeds from training and validation.
* **What would you improve?** Wider training range, per-user calibration, real recordings, and a version that runs as an operating-system background tool.

## 8. Tests

```
python tests/run_tests.py          # unit tests of simulator, filters, metrics
node tests/test_js_parity.js       # JavaScript filters equal the Python filters
python tests/test_browser.py       # a bot plays the demo in headless Chrome (needs Playwright)
```

## 9. Suggested 8-slide outline

1. Problem: tremor makes small targets hard to click. 2. Key insight: tremor is fast, intention is slow.
3. Method: three filters, the AI subtracts the estimated tremor. 4. How the AI was trained (simulation, `training_curve.png`).
5. Experiment: simulated click test (`fig4`, `fig2`). 6. The real difference: lag and distortion (`fig1`, `fig5`, `fig6`).
7. Limitations and robustness (`fig7`). 8. Live demo and future work.
