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

from tqdm import tqdm

#region primary configs of the experiment

dataset_name = "cifar10"
model_sigma = 0.25
n_datapoints = 2048
smoothing_sigma = 0.25
n_samples = 10000
n_trial_samples = 1000

score_method = "TPS"
calibration_budget = 0.1
n_iterations = 100

#endregion

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

coverage_range = np.linspace(0.85, 0.99, 15).round(3)
# r_range = np.linspace(0, smoothing_sigma * 2, 11).round(3)
r_range = [0.0, 0.06, 0.12, 0.18, 0.25, 0.37, 0.5, 0.75]

#endregion

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
            "iteration": iter_i,
            "coverage_guarantee": coverage_guarantee,
            "r": r,
            "smoothing_sigma": smoothing_sigma,
            "model_sigma": model_sigma,
            "threshold": threshold,
            "empirical_coverage": empirical_coverage,
            "average_set_size": average_set_size
        })

vanilla_results = pd.DataFrame(vanilla_results)
vanilla_results#.to_csv(f"result_folder/vanilla_results.csv", index=False)


eta = 0.001
cal_mask = get_cal_mask(mean_scores, calibration_budget)
cal_budget = cal_mask.sum()

cas_results = []

for r in r_range:
    print(f"r: {r}")
    cas_cp = CAS(nominal_coverage=0.9, r=r, smoothing_sigma=smoothing_sigma, confidence_level=1-eta, n_dcal=cal_budget, n_classes=n_classes)
    cas_cp.pre_compute(smooth_scores, smooth_prediction.y_true)
    print("Scores computed!")

    for coverage_guarantee in tqdm(coverage_range):
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
cas_results.to_csv(f"./cas_results-class.csv", index=False)

class bin_cp(RobustCP):
    def __init__(self, smoothing_sigma=0.0, confidence_level=0.999, n_dcal=None, n_classes=None, cutoff_prob=0.5, **kwargs):
        self.smoothing_sigma = smoothing_sigma
        self.eta = 1 - confidence_level
        self.cutoff_prob = cutoff_prob

        super().__init__(**kwargs)
        # self.confidence_adv = norm.cdf(self.r, scale=self.smoothing_sigma) # this does not include the cutoff probability
        self.confidence_adv = norm.cdf(norm.ppf(self.cutoff_prob) + self.r / self.smoothing_sigma)
        self.n_dcal = n_dcal
        self.n_classes = n_classes
        self.cutoff_prob = cutoff_prob
    
    def correct_coverage_guarantee(self):
        self.final_coverage = self.nominal_coverage + self.eta
        # print("I am called with ", self.final_coverage)

    def calibrate_from_scores(self, S_sample, y, return_scores=False):
        true_smooth_scores = S_sample[torch.arange(S_sample.shape[0]), y]
        dkw_results = [dkw_cdf(true_smooth_scores[i], confidence=1 - self.eta/(2 * self.n_dcal), bonferroni_tasks=1, type="upper") for i in range(true_smooth_scores.shape[0])]
        calibration_scores = torch.tensor([dkw_results[i][1][dkw_results[i][0] < self.cutoff_prob][-1] for i in range(len(dkw_results))])
        # print("calibration_dim ", calibration_scores.shape, calibration_scores.ndim)
        self.conformal_threshold = self.calibrate_from_refined_scores(calibration_scores, y)
        if return_scores:
            return self.conformal_threshold, calibration_scores
        return self.conformal_threshold
    
    def predict_from_scores(self, S_sample, return_scores=False):
        pred_bins = (S_sample <= self.conformal_threshold).sum(dim=-1)
        corrected_bins = torch.tensor(
                clopper_pearson_lower(pred_bins.cpu(), S_sample.shape[-1], alpha=self.eta/(self.n_dcal + self.n_classes))).to(device)

        test_scores = corrected_bins
        
        pred_sets = self.predict_from_refined_scores(test_scores)
        if return_scores:
            return pred_sets, S_sample
        return pred_sets

    def pre_compute_predict(self, test_mask):
        S_sample = self.precompute__test_scores[test_mask]
        pred_bins = (S_sample <= self.conformal_threshold).sum(dim=-1)
        corrected_bins = torch.tensor(
                clopper_pearson_lower(pred_bins.cpu(), S_sample.shape[-1], alpha=self.eta/(self.n_dcal + self.n_classes))).to(device)

        test_scores = corrected_bins
        
        pred_sets = self.predict_from_refined_scores(test_scores)
        return pred_sets
    
    def predict_from_refined_scores(self, test_scores):
        pred_set = ~(test_scores > self.confidence_adv)
        return pred_set
    
    def calibrate_from_refined_scores(self, calibration_scores, y):
        if calibration_scores.ndim == 1:
            y_true_mask = torch.ones(size=(calibration_scores.shape[0], 1), dtype=torch.bool)
            # print(y_true_mask)
            # print(calibration_scores.shape)
            self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores.reshape(-1, 1), y_true_mask)
        else:
            y_true_mask = F.one_hot(y, num_classes=calibration_scores.shape[1]).bool()
            self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores, y_true_mask)
        return self.conformal_threshold

eta = 0.001
cal_mask = get_cal_mask(mean_scores, calibration_budget)
cal_budget = cal_mask.sum()

bin_cp_results = []

for r in r_range:
# r = 0.25
    print(f"r: {r}")
    qr_cp = bin_cp(nominal_coverage=0.9, r=r, smoothing_sigma=smoothing_sigma, confidence_level=1-eta, n_dcal=cal_budget, n_classes=n_classes)
    qr_cp.pre_compute(smooth_scores, smooth_prediction.y_true)
    print("Scores computed!")

    for coverage_guarantee in tqdm(coverage_range):

        qr_cp.set_nominal_coverage(coverage_guarantee)
        # print(f"coverage_guarantee: {qr_cp.final_coverage}")
        # print(f"internal coverage guarantee: {qr_cp.internal_cp.coverage_guarantee}")

        for iter_i in range(n_iterations):
            cal_mask = get_cal_mask(mean_scores, calibration_budget)
            eval_mask = ~cal_mask

            threshold = qr_cp.pre_compute_calibrate(cal_mask)
            pred_set = qr_cp.pre_compute_predict(eval_mask)

            empirical_coverage = qr_cp.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
            average_set_size = pred_set.sum(dim=1).float().mean().item()

            bin_cp_results.append({
                "method": "BinCP",
                "coverage_guarantee": coverage_guarantee,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "model_sigma": model_sigma,
                "threshold": threshold,
                "empirical_coverage": empirical_coverage,
                "average_set_size": average_set_size
            })

            # print(f"Coverage: {empirical_coverage}, Average Set Size: {average_set_size}")
            # print(f"coverage_guarantee: {qr_cp.final_coverage} -- {qr_cp.internal_cp.coverage_guarantee}")

bin_cp_results = pd.DataFrame(bin_cp_results)
bin_cp_results.to_csv(f"./bin_cp_results-class.csv", index=False)
