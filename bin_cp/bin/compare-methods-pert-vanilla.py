import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from bin_cp.helpers.storage import load_smooth_prediction
from bin_cp.helpers.tensor import get_smooth_scores, get_cal_mask, quantization_pdf, bound_tensor
from bin_cp.robust.confidence import bernstein_bound, dkw_cdf
from bin_cp.robust.confidence import clopper_pearson_lower
from bin_cp.robust.bounds import mean_bounds_l2, CDF_bounds_l2

from bin_cp.cp.core import ConformalClassifier as CP
from bin_cp.cp.scores import APSScore, TPSScore

from bin_cp.methods.robust_cp import RobustCP, VanillaSmoothCP
from bin_cp.methods.cas import CAS
from bin_cp.methods.bin import BinCP
from bin_cp.methods.binary import BinCPThresholds

from tqdm import tqdm

import logging
logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True


from sacred import Experiment

experiment = Experiment("EvaluateRobustMethods")
experiment.logger = logger

@experiment.config
def default_config():
    
    #region primary configs of the experiment

    result_folder = "../../../output-results"

    dataset_name = "cifar10"
    model_sigma = 0.25
    n_classes=10
    n_datapoints = 2048
    smoothing_sigma = 0.25
    n_samples = 10000
    n_trial_samples = 2000

    score_method = "TPS"
    calibration_budget = 0.1
    n_iterations = 100
    confidence = 0.99

    r=0.12
    #endregion

@experiment.automain
def run(
    dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, n_trial_samples, n_classes,
    score_method, calibration_budget, n_iterations, confidence,
    result_folder, r):


    coverage_range = [0.9]
    r_range = [r]

    #region loding smooth logit predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smooth_prediction = load_smooth_prediction(dataset_name=dataset_name,
        model_sigma=model_sigma,
        n_datapoints=n_datapoints,
        smoothing_sigma=smoothing_sigma,
        n_samples=n_samples)
    n_classes = 10 if dataset_name == "cifar10" else None

    smooth_prediction_pert = load_smooth_prediction(dataset_name=dataset_name,
        model_sigma=model_sigma,
        n_datapoints=n_datapoints,
        smoothing_sigma=smoothing_sigma,
        n_samples=n_samples, r=r)
    #endregion

    #region defining basic setup for conformal evaluation

    score_pipeline = [
        TPSScore(softmax=True) if score_method == "TPS" else APSScore(softmax=True)] # defining the score function
    cp = CP(score_pipeline=score_pipeline, coverage_guarantee=0.9) # the guarantee can vary later by cp.coverage_guarantee
    smooth_scores = get_smooth_scores(smooth_prediction.logits, cp, mean=False).to(device)
    smooth_scores_pert = get_smooth_scores(smooth_prediction_pert.logits, cp, mean=False).to(device)
    y_true_mask = F.one_hot(smooth_prediction.y_true, num_classes=10).bool().to(device)
    mean_scores = smooth_scores.mean(dim=-1)
    smooth_scores = smooth_scores[:, :, :n_trial_samples]
    smooth_scores_pert = smooth_scores_pert[:, :, :n_trial_samples]

    #endregion
    print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints and {n_samples} samples: Score method: {score_method}")

    cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
    n_dcal = cal_mask.sum().item()


    bin_results = []

    # here goes a for
    # r = 0.12
    for r in r_range:
        print("Computing for r=", r)
        # making classes

        bin_cp = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes,
                            r=0, confidence_level=confidence,
                            error_correction=False,
                            p_base=0.6)

        # bin_cp.pre_compute(smooth_scores, smooth_prediction.y_true)
        print("bin pre-computed")

        # here goes a for
        # coverage_guarantee = 0.9
        for coverage_guarantee in coverage_range:
            print(f"Running for r={r}, coverage={coverage_guarantee}")
            bin_cp.set_nominal_coverage(coverage_guarantee)

            # here goes a for
            for iter_i in tqdm(range(n_iterations)):
                cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
                eval_mask = ~cal_mask

                # evaluating bin
                threshold_bin = bin_cp.calibrate_from_scores(smooth_scores[cal_mask], smooth_prediction.y_true[cal_mask])
                pred_set_bin = bin_cp.predict_from_scores(smooth_scores_pert[eval_mask])

                empirical_coverage_bin = bin_cp.internal_cp.coverage(pred_set_bin, y_true_mask[eval_mask])
                average_set_size_bin = pred_set_bin.sum(dim=1).float().mean().item()

                bin_results.append({
                    "method": "bin",
                    "coverage_guarantee": coverage_guarantee,
                    "iteration": iter_i,
                    "r": r,
                    "smoothing_sigma": smoothing_sigma,
                    "model_sigma": model_sigma,
                    "threshold": threshold_bin,
                    "empirical_coverage": empirical_coverage_bin,
                    "average_set_size": average_set_size_bin,
                    "score_method": score_method,
                    "confidence_level": confidence,
                    "dataset_name": dataset_name,
                    "calibration_budget": calibration_budget,
                })

    bin_results = pd.DataFrame(bin_results)
    bin_results.to_csv(f"{result_folder}/bin-VANILLA_results-{dataset_name}-smooth{smoothing_sigma}-model{model_sigma}-{score_method}-nsamples{n_trial_samples}-conf{confidence}-r_{r}-adversarial.csv", index=False)
