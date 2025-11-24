# RCP1

Implementation for the paper “**One Sample is Enough to Make Conformal Prediction Robust**” (link to paper: (https://openreview.net/forum?id=h5NsMrUK4g&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2025%2FConference%2FAuthors%23your-submissions))).  
Please cite the paper if you use this code:  
> [Zargarbashi, S. H., Akhondzadeh, M. S., & Bojchevski, A. (2025). *One Sample is Enough to Make Conformal Prediction Robust*. arXiv:2506.16553

---

## Overview

This repository implements **RCP1** (Robust Conformal Prediction with a Single Sample), a method for constructing prediction sets with robustness against worst-case input perturbations while requiring only one forward pass per input. :contentReference[oaicite:1]{index=1}  
In contrast to typical randomized-smoothing based robust conformal methods (which perform many passes per input), RCP1 uses a noise-augmented input and a conservative calibration threshold to achieve robustness. :contentReference[oaicite:2]{index=2}

---

## Features

- Model‐agnostic implementation: works with any black-box model, for classification or regression.  
- Efficient: only a single (augmented) model inference required per test sample.  
- Code and examples to replicate or build upon the method in your own tasks.

---

## Installation

```bash
# Clone this repository
git clone https://github.com/USERNAME/RCP1.git
cd RCP1

# (Optionally) create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
cd bin_cp
pip install -e .
```
