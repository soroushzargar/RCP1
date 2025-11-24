# RCP1

Implementation of **RCP1 — Robust Conformal Prediction with One Sample**, from the paper:

**“One Sample is Enough to Make Conformal Prediction Robust”**
*Link to paper: `[[PLACEHOLDER]](https://openreview.net/forum?id=h5NsMrUK4g&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2025%2FConference%2FAuthors%23your-submissions))`*

If you use this repository in research or applications, please cite the paper:

```
@article{zargarbashi2025one,
  title={One Sample is Enough to Make Conformal Prediction Robust},
  author={Zargarbashi, Soroush H and Akhondzadeh, Mohammad Sadegh and Bojchevski, Aleksandar},
  journal={arXiv preprint arXiv:2506.16553},
  year={2025}
}
```

---

## Overview

RCP1 is a **single-sample robust conformal prediction method**.
Unlike previous smoothing-based approaches (e.g., BinCP, RSCP+) that require **tens to hundreds** of model evaluations per input, RCP1:

* Uses **only one noise-augmented forward pass** per data point
* Works with **any model**, **any conformity score**, and **any smoothing scheme**
* Provides **worst-case adversarial coverage guarantees**, certified using binary robustness certificates
* Matches the prediction-set quality of multi-sample methods while being orders of magnitude faster

At a high level, RCP1 **certifies the conformal procedure itself**, not the individual scores.
This allows replacing the nominal coverage (1-\alpha) with a corrected value (1-\alpha') such that the certified lower bound remains (1-\alpha) under the threat model.

---

## Installation

```bash
git clone https://github.com/soroushzargar/RCP1.git
cd RCP1
pip install -r requirements.txt
```

---

## Usage

A minimal workflow:

### 1. Train a base model

Any architecture and score function can be used (classification or regression).

### 2. Prepare a calibration set

Ensure it is exchangeable with the test distribution.

### 3. Calibrate RCP1

```python
from rcp1 import RCP1

rcp = RCP1(
    model=model,
    alpha=0.1,                   # target coverage
    smoothing_sigma=0.5,         # noise level for smoothing
    certificate="gaussian_l2"    # placeholder; adapt to your code
)

rcp.calibrate(calibration_data)
```

This step draws **one noise sample per calibration input**, computes conformity scores, applies the certificate, and computes the corrected quantile (q).

### 4. Construct prediction sets at test time

```python
prediction_set = rcp.predict_set(x_test)
```

Internally this:

* Draws one new noise sample (\epsilon)
* Computes the conformity score on (x+\epsilon)
* Returns
  [
  C(x) = { y : s(x+\epsilon, y) \ge q }.
  ]

### 5. Evaluate

Compute marginal coverage, robust coverage, average set size, etc.

---

## Directory Structure

```
RCP1/
├── rcp1/               # Core implementation
│   ├── calibration.py
│   ├── prediction.py
│   ├── certificates.py
│   └── ...
├── examples/           # Example scripts or notebooks
├── requirements.txt
└── README.md
```

---

## Paper Link and Supplementary Material

* Paper PDF: `[PLACEHOLDER]`
* Supplementary material: `[PLACEHOLDER]`

---

## License

MIT License. See `LICENSE`.

---

## Contact

Open an issue on this repository for questions, bugs, or requests.
