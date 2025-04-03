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
from bin_cp.cp.scores import APSScore, TPSScore, LogitScore

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

experiment = Experiment("EvaluateLogits")
experiment.logger = logger

@experiment.config
def default_config():
    
    #region primary configs of the experiment

    result_folder = "../../../output-results"

    dataset_name = "cifar10"
    model_sigma = 0.5
    n_classes=10
    n_datapoints = 2048
    smoothing_sigma = 0.5
    n_samples = 10000
    n_trial_samples = 2000

    score_method = "APS"
    calibration_budget = 0.1
    n_iterations = 100
    confidence = 0.99

    #endregion

@experiment.automain
def run(
    dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, n_trial_samples, n_classes,
    score_method, calibration_budget, n_iterations, confidence,
    result_folder):


    coverage_range = [0.85, 0.9, 0.95]
    r_range = [0, 0.06, 0.12, 0.18, 0.25, 0.37, 0.5, 0.75]



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

    score_pipelineLogits = [
        LogitScore()] # defining the score function
    score_pipeline = [
        TPSScore(softmax=True) if score_method == "TPS" else APSScore(softmax=True)] # defining the score function
    
    cp = CP(score_pipeline=score_pipeline, coverage_guarantee=0.9) # the guarantee can vary later by cp.coverage_guarantee
    smooth_scores = get_smooth_scores(smooth_prediction.logits, cp, mean=False)
    smooth_scores = smooth_scores[:, :, :n_trial_samples]

    smooth_scores_logits = get_smooth_scores(smooth_prediction.logits, cp, mean=False)
    smooth_scores_logits = smooth_scores_logits[:, :, :n_trial_samples]

    y_true_mask = F.one_hot(smooth_prediction.y_true, num_classes=10).bool().to(device)
    mean_scores = smooth_scores.mean(dim=-1)
    #endregion
    print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints and {n_trial_samples} samples: Score method: {score_method}")

    cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
    n_dcal = cal_mask.sum().item()

    vanilla_cp = VanillaSmoothCP(nominal_coverage=0.9)
    vanilla_results = []

    vanilla_cp.pre_compute(smooth_scores, smooth_prediction.y_true)

    for coverage_guarantee in coverage_range:
        r = 0
        vanilla_cp.set_nominal_coverage(coverage_guarantee)

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
            eval_mask = ~cal_mask
            threshold = vanilla_cp.pre_compute_calibrate(cal_mask)
            pred_set = vanilla_cp.pre_compute_predict(eval_mask)

            empirical_coverage = vanilla_cp.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            vanilla_results.append({
                "method": "vanilla", 
                "iteration": iter_i,
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size,
                "score_method": score_method,
                "dataset_name": dataset_name,
                "calibration_budget": calibration_budget,
            })

    vanilla_results = pd.DataFrame(vanilla_results)
    vanilla_results.to_csv(f"{result_folder}/vanilla_results-{dataset_name}-smooth{smoothing_sigma}-model{model_sigma}-LOGITS-nsamples{n_trial_samples}.csv", index=False)
    print(vanilla_results[vanilla_results["coverage_guarantee"] == 0.9].mean())


    cas_results = []
    bin_results = []
    bin_logit_results = []

    # here goes a for
    # r = 0.12
    for r in r_range:
        print("Computing for r=", r)
        # making classes
        cas_cp = CAS(nominal_coverage=0.9, r=r, smoothing_sigma=smoothing_sigma, confidence_level=confidence, n_dcal=n_dcal, n_classes=n_classes, 
                        error_correction=True)
        cas_cp.pre_compute(smooth_scores, smooth_prediction.y_true)

        print("CAS pre-computed")

        bin_cp = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes,
                            r=r, confidence_level=confidence,
                            error_correction=True,
                            p_base=0.6)
        bin_cp_logits = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes,
                            r=r, confidence_level=confidence,
                            error_correction=True,
                            p_base=0.6)

        # bin_cp.pre_compute(smooth_scores, smooth_prediction.y_true)
        print("bin pre-computed")

        # here goes a for
        # coverage_guarantee = 0.9
        for coverage_guarantee in coverage_range:
            print(f"Running for r={r}, coverage={coverage_guarantee}")
            cas_cp.set_nominal_coverage(coverage_guarantee)
            bin_cp.set_nominal_coverage(coverage_guarantee)
            bin_cp_logits.set_nominal_coverage(coverage_guarantee)

            # here goes a for
            for iter_i in tqdm(range(n_iterations)):
                cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
                eval_mask = ~cal_mask

                # evaluating cas
                threshold_cas = cas_cp.pre_compute_calibrate(cal_mask)
                pred_set_cas = cas_cp.pre_compute_predict(eval_mask)

                empirical_coverage_cas = cas_cp.internal_cp.coverage(pred_set_cas, y_true_mask[eval_mask])
                average_set_size_cas = pred_set_cas.sum(dim=1).float().mean().item()

                cas_results.append({
                    "method": "cas",
                    "coverage_guarantee": coverage_guarantee,
                    "iteration": iter_i,
                    "r": r,
                    "smoothing_sigma": smoothing_sigma,
                    "model_sigma": model_sigma,
                    "threshold": threshold_cas,
                    "empirical_coverage": empirical_coverage_cas,
                    "average_set_size": average_set_size_cas,
                    "score_method": score_method,
                    "confidence_level": confidence,
                    "dataset_name": dataset_name,
                    "calibration_budget": calibration_budget,
                })

                # evaluating bin
                threshold_bin = bin_cp.calibrate_from_scores(smooth_scores[cal_mask], smooth_prediction.y_true[cal_mask])
                pred_set_bin = bin_cp.predict_from_scores(smooth_scores[eval_mask])

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
                
                # evaluating bin logits
                threshold_bin_logits = bin_cp_logits.calibrate_from_scores(smooth_scores_logits[cal_mask], smooth_prediction.y_true[cal_mask])
                pred_set_bin_logits = bin_cp_logits.predict_from_scores(smooth_scores_logits[eval_mask])

                empirical_coverage_bin_logits = bin_cp_logits.internal_cp.coverage(pred_set_bin_logits, y_true_mask[eval_mask])
                average_set_size_bin_logits = pred_set_bin_logits.sum(dim=1).float().mean().item()

                bin_logit_results.append({
                    "method": "bin_logits",
                    "coverage_guarantee": coverage_guarantee,
                    "iteration": iter_i,
                    "r": r,
                    "smoothing_sigma": smoothing_sigma,
                    "model_sigma": model_sigma,
                    "threshold": threshold_bin_logits,
                    "empirical_coverage": empirical_coverage_bin_logits,
                    "average_set_size": average_set_size_bin_logits,
                    "score_method": "Logits",
                    "confidence_level": confidence,
                    "dataset_name": dataset_name,
                    "calibration_budget": calibration_budget,
                })
                if iter_i % 10 == 0:
                    print(cas_results[-1])
                    print(bin_results[-1])
                    print(bin_logit_results[-1])

    cas_results = pd.DataFrame(cas_results)
    bin_results = pd.DataFrame(bin_results)
    bin_logit_results = pd.DataFrame(bin_logit_results)

    cas_results.to_csv(f"{result_folder}/cas_results-{dataset_name}-smooth{smoothing_sigma}-model{model_sigma}-LOGITS-nsamples{n_trial_samples}-conf{confidence}.csv", index=False)
    bin_results.to_csv(f"{result_folder}/bin_results-{dataset_name}-smooth{smoothing_sigma}-model{model_sigma}-LOGITS-nsamples{n_trial_samples}-conf{confidence}.csv", index=False)
    bin_logit_results.to_csv(f"{result_folder}/bin_logit_results-{dataset_name}-smooth{smoothing_sigma}-model{model_sigma}-LOGITS-nsamples{n_trial_samples}-conf{confidence}.csv", index=False)
