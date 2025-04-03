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

experiment = Experiment("EvaluateCP")
experiment.logger = logger

@experiment.config
def default_config():
    #region primary configs of the experiment

    dataset_name = "cifar10"
    model_sigma = 0.25
    n_datapoints = 2048
    smoothing_sigma = 0.25
    n_samples = 10000
    n_trial_samples = 1000

    score_method = "TPS"
    calibration_budget = 0.1
    n_iterations = 20

    lambda_base = 0.3
    p_base = 0.9
    r=0.0

    #endregion


@experiment.automain
def run(
    dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, n_trial_samples,
    score_method, calibration_budget, n_iterations, lambda_base, p_base, r):
    
    coverage_range = [0.85, 0.9, 0.95]

    #region loding smooth logit predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smooth_prediction = load_smooth_prediction(dataset_name=dataset_name,
        model_sigma=model_sigma,
        n_datapoints=n_datapoints,
        smoothing_sigma=smoothing_sigma,
        n_samples=n_samples)
    n_classes = 10 if dataset_name == "cifar10" else None
    #endregion

    #region defining basic setup for conformal evaluation

    # score_bias = (1 if score_method == "APS" else 0) # this ensures that the scores are always in [0, 1]
    score_pipeline = [
        TPSScore(softmax=True) if score_method == "TPS" else APSScore(softmax=True)] # defining the score function
    cp = CP(score_pipeline=score_pipeline, coverage_guarantee=0.9) # the guarantee can vary later by cp.coverage_guarantee
    smooth_scores = get_smooth_scores(smooth_prediction.logits, cp, mean=False)
    y_true_mask = F.one_hot(smooth_prediction.y_true, num_classes=10).bool().to(device)
    mean_scores = smooth_scores.mean(dim=-1)

    print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints and {n_samples} samples: Score method: {score_method}")
    #endregion

    vanilla_cp = VanillaSmoothCP(nominal_coverage=0.9)
    vanilla_results = []

    vanilla_cp.pre_compute(smooth_scores, smooth_prediction.y_true)

    for coverage_guarantee in coverage_range:
        vanilla_cp.set_nominal_coverage(coverage_guarantee)

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
            eval_mask = ~cal_mask
            threshold = vanilla_cp.pre_compute_calibrate(cal_mask)
            pred_set = vanilla_cp.pre_compute_predict(eval_mask)

            empirical_coverage = vanilla_cp.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            vanilla_results.append({
                "iteration": iter_i,
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size,
                "score_method": score_method,  
            })

    vanilla_results = pd.DataFrame(vanilla_results)
    vanilla_results[vanilla_results["coverage_guarantee"] == 0.9].mean()

    bin_cp_lambda = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=calibration_budget, n_classes=n_classes,
                    error_correction=False, r=r
                    lambda_base=0.3)
    bin_results_lambda = []

    bin_cp_lambda.pre_compute(smooth_scores, smooth_prediction.y_true)

    # for coverage_guarantee in coverage_range:
    for coverage_guarantee in coverage_range:
        r = 0
        bin_cp_lambda.set_nominal_coverage(coverage_guarantee)

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
            eval_mask = ~cal_mask
            threshold = bin_cp_lambda.pre_compute_calibrate(cal_mask)
            pred_set = bin_cp_lambda.predict_from_scores(smooth_scores[eval_mask])

            empirical_coverage = bin_cp_lambda.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            bin_results_lambda.append({
                "iteration": iter_i,
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size,
                "score_method": score_method, 
                "lambda_base": lambda_base,
            })

    bin_results_lambda = pd.DataFrame(bin_results_lambda)
    bin_results_lambda[bin_results_lambda["coverage_guarantee"] == 0.9].mean()

    bin_cp_p = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=calibration_budget, n_classes=n_classes,
                    error_correction=False,
                    p_base=0.9)
    bin_results_p = []

    bin_cp_p.pre_compute(smooth_scores, smooth_prediction.y_true)

    # for coverage_guarantee in coverage_range:
    for coverage_guarantee in coverage_range:
        bin_cp_p.set_nominal_coverage(coverage_guarantee)

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
            eval_mask = ~cal_mask
            threshold = bin_cp_p.calibrate_from_scores(smooth_scores[cal_mask], smooth_prediction.y_true[cal_mask])
            pred_set = bin_cp_p.predict_from_scores(smooth_scores[eval_mask])

            empirical_coverage = bin_cp_p.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            bin_results_p.append({
                "iteration": iter_i,
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size,
                "score_method": score_method, 
            })

    bin_results_p = pd.DataFrame(bin_results_p)
    bin_results_p#.to_csv(f"result_folder/vanilla_results.csv", index=False)
    bin_results_p[bin_results_p["coverage_guarantee"] == 0.9].mean()


    eta = 0.001
    cal_mask = get_cal_mask(mean_scores, calibration_budget)
    cal_budget = cal_mask.sum()

    cas_results = []

    r = 0
    cas_cp = CAS(nominal_coverage=0.9, r=0, smoothing_sigma=smoothing_sigma, confidence_level=1-eta, n_dcal=cal_budget, n_classes=n_classes, error_correction=False)
    cas_cp.pre_compute(smooth_scores, smooth_prediction.y_true)
    print("Scores computed!")

    # for coverage_guarantee in tqdm(coverage_range):
    for coverage_guarantee in tqdm([0.9]):
    # for coverage_guarantee in coverage_range:
        cas_cp.set_nominal_coverage(coverage_guarantee)

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(mean_scores, calibration_budget)
            eval_mask = ~cal_mask

            threshold = cas_cp.pre_compute_calibrate(cal_mask)
            pred_set = cas_cp.pre_compute_predict(eval_mask)

            empirical_coverage = cas_cp.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            cas_results.append({
                "method": "cas",
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size
            })

        # print(f"Coverage: {empirical_coverage}, Average Set Size: {average_set_size}")
        # print(f"coverage_guarantee: {cas_cp.final_coverage} -- {cas_cp.internal_cp.coverage_guarantee}")

    cas_results = pd.DataFrame(cas_results)
    # cas_results.to_csv(f"./cas_results-class.csv", index=False)
    cas_results[cas_results["coverage_guarantee"] == 0.9].mean()
    