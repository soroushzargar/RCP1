import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os


from bin_cp.helpers.storage import load_smooth_prediction
from bin_cp.helpers.tensor import get_smooth_scores, get_cal_mask, quantization_pdf, bound_tensor
from bin_cp.robust.confidence import bernstein_bound, dkw_cdf
from bin_cp.robust.confidence import clopper_pearson_lower
from bin_cp.robust.bounds import mean_bounds_l2, CDF_bounds_l2

from bin_cp.cp.core import ConformalClassifier as CP
from bin_cp.cp.scores import APSScore, TPSScore, LogitScore

from bin_cp.methods.robust_cp import RobustCP, VanillaSmoothCP
from bin_cp.methods.cas import CAS
from bin_cp.methods.bincp import BinCP
from bin_cp.methods.rcp_one import RCP1
# from qrcp.methods.binary import QRCPThresholds
import time

from tqdm import tqdm

from sacred import Experiment

ex = Experiment("BinCP-L2-Gaussian-Comparison")

@ex.config
def config():
    #region primary configs of the experiment

    output_dir = "../../results/"

    dataset_name = "cifar10"
    model_sigma = 0.25
    n_classes=10
    n_datapoints = 2048
    smoothing_sigma = 0.25
    n_samples = 10000
    n_trial_samples = 120

    score_method = "TPS"
    calibration_budget = 0.1
    n_iterations = 100

    confidence = 0.999
    coverage_range = [0.85, 0.9, 0.95]      
    # coverage_range = [0.9]
    r_range = [0.0, 0.06, 0.12, 0.18, 0.25, 0.37, 0.5, 0.75]
    # r_range = [0.12,  0.25, 0.5, ]

    #endregion

@ex.automain
def run(
    output_dir, dataset_name, n_classes, n_datapoints,
    model_sigma, smoothing_sigma, n_samples, n_trial_samples,
    score_method, calibration_budget, n_iterations,
    coverage_range, r_range, confidence
):
    output_dir = pathlib.Path(output_dir)/dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = pathlib.Path("/home/c01saha/CISPA-projects/agrauq-2024/Robust_CP/models/")
    dataset_dir = pathlib.Path("/home/c01saha/CISPA-projects/agrauq-2024/datasets/")
    logits_dir = pathlib.Path("/home/c01saha/CISPA-projects/agrauq-2024/Robust_CP/logits/")

    #region loding smooth logit predictions
    if n_samples < n_trial_samples:
        print(f"Number of trial samples is set to {n_trial_samples} as it is smaller than the number of samples.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        smooth_prediction = load_smooth_prediction(dataset_name=dataset_name,
            model_sigma=model_sigma,
            n_datapoints=n_datapoints,
            smoothing_sigma=smoothing_sigma,
            n_samples=n_samples,
            models_dir=models_dir,
            dataset_dir=dataset_dir,
            logits_dir=logits_dir,)
        n_classes = 10 if dataset_name == "cifar10" else None
    except FileNotFoundError as e:
        print("Smooth predictions not found, you can generate them using bin/smooth_logits_clean.py")
        print("Full description of the error: ", e)
    #endregion

    # region computing the conformal scores.
    score_pipeline = [
        TPSScore(softmax=True) if score_method == "TPS" else APSScore(softmax=True)] # defining the score function
    cp = CP(score_pipeline=score_pipeline, coverage_guarantee=0.9) # the guarantee can vary later by cp.coverage_guarantee
    smooth_scores = get_smooth_scores(smooth_prediction.logits, cp, mean=False)
    smooth_scores = smooth_scores[:, :, :n_trial_samples]
    y_true_mask = F.one_hot(smooth_prediction.y_true, num_classes=10).bool().to(device)
    print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints and {n_samples} samples: Score method: {score_method}")

    result = []

    for r in r_range:
        for coverage_guarantee in coverage_range:
            cal_mask = get_cal_mask(smooth_scores, calibration_budget)
            n_dcal = cal_mask.sum().item()

            cp_methods = {
                "CAS": CAS(smoothing_sigma=smoothing_sigma, confidence_level=confidence, n_dcal=n_dcal, n_classes=n_classes, nominal_coverage=coverage_guarantee, r=r),
                "BinCP": BinCP(smoothing_sigma=smoothing_sigma, confidence_level=confidence, n_dcal=n_dcal, n_classes=n_classes,
                            p_base=0.8, scheme="guass", nominal_coverage=coverage_guarantee, r=r),
                "RCP1": RCP1(smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes, nominal_coverage=coverage_guarantee, r=r),
            }

            for method_name, method in cp_methods.items():
                print(f"Pre-computing {method_name} for r={r} and coverage={coverage_guarantee}")
                method.pre_compute(smooth_scores, smooth_prediction.y_true)

            print(f"Running the experiment for r={r} and coverage={coverage_guarantee}")
            for iteration in tqdm(range(n_iterations)):
                cal_mask = get_cal_mask(smooth_scores, calibration_budget)
                eval_mask = ~cal_mask

                for method_name, method in cp_methods.items():
                    threshold = method.calibrate_from_scores(smooth_scores[cal_mask], smooth_prediction.y_true[cal_mask])
                    pred_set = method.predict_from_scores(smooth_scores[eval_mask], return_scores=False)
                    covered = (pred_set)[torch.arange(pred_set.shape[0]), smooth_prediction.y_true[eval_mask]]
                    set_size = pred_set.sum(dim=1)

                    empirical_coverage = covered.float().mean().item()
                    avg_set_size = set_size.float().mean().item()
                    result.append({
                        "method": method_name,
                        "empirical_coverage": empirical_coverage,
                        "avg_set_size": avg_set_size,
                        "r": r,
                        "coverage_guarantee": coverage_guarantee,
                        "threshold": threshold,
                        "internal_coverage_level": method.internal_cp.coverage_guarantee,
                        "below_1": (set_size <= 1).float().mean().item(),
                        "below_3": (set_size <= 3).float().mean().item(),
                        "below_5": (set_size <= 5).float().mean().item(),
                        "below_1_coverage": covered[set_size <= 1].float().mean().item(),
                        "below_3_coverage": covered[set_size <= 3].float().mean().item(),
                        "below_5_coverage": covered[set_size <= 5].float().mean().item(),
                        "n_samples": n_trial_samples,
                    })


    result = pd.DataFrame(result)
    result.to_csv(output_dir/f"results_l2-gauss_{dataset_name}_{score_method}_clean_r-0.0_samples-{n_trial_samples}_model-{model_sigma}_smoothing-{smoothing_sigma}.csv", index=False)
    result_summary = result.groupby(["coverage_guarantee", "r", "method"]).mean()
    result_summary.to_csv(output_dir/f"results_l2-gauss_{dataset_name}_{score_method}_clean_r-0.0_samples-{n_trial_samples}_model-{model_sigma}_smoothing-{smoothing_sigma}_summary.csv", index=False)