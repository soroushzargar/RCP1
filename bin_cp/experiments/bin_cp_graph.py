cora_file = "<location to cora smooth file>/GCN-cora_ml-0_01-0_6-0_0-0_0-smooth_logits.pth"

import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import norm

from bin_cp.helpers.storage import load_smooth_prediction
from bin_cp.helpers.tensor import get_smooth_scores, get_cal_mask, quantization_pdf, bound_tensor
from bin_cp.robust.confidence import bernstein_bound, dkw_cdf
from bin_cp.robust.confidence import clopper_pearson_lower
from bin_cp.robust.bounds import mean_bounds_l2, CDF_bounds_l2

from bin_cp.cp.core import ConformalClassifier as CP
from bin_cp.cp.scores import APSScore, TPSScore

from bin_cp.methods.robust_cp import RobustCP, VanillaSmoothCP
from bin_cp.methods.cas import CAS, SparseCAS
from bin_cp.methods.bin import BinCP, SparseBinCP
from bin_cp.methods.binary import QRCPThresholds
import time

from tqdm import tqdm
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_folder = "../../../output-results"

dataset_name = "CoraML"


n_trial_samples = 2000

score_method = "TPS"
calibration_budget = 0.1
n_iterations = 100 # TODO: change to 100
nominal_coverage = 0.9

score_pipeline = [
    TPSScore(softmax=True) if score_method == "TPS" else APSScore(softmax=True)] # defining the score function
cp = CP(score_pipeline=score_pipeline, coverage_guarantee=0.9) # the guarantee can vary later by cp.coverage_guarantee


from torch_geometric.datasets import CitationFull
dataset = CitationFull(root='/tmp/CoraML', name='cora_ml')
y_true = (dataset.data.y).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = torch.load(cora_file)
print("shape = ", logits.shape)


y_pred = F.softmax(logits, dim=-1).to(device).mean(dim=1).argmax(dim=1)
print("Acc = ", (y_pred == y_true).float().mean())

n_classes = logits.shape[-1]

n_datapoints = logits.shape[0]
n_samples = n_trial_samples
logits = logits.permute(0, 1, 2).to(device)
y_true = y_true.to(device)
y_pred = y_pred.to(device)

smooth_scores = get_smooth_scores(logits, cp, mean=False)
y_true_mask = F.one_hot(y_true, num_classes=n_classes).bool().to(device)
smooth_scores = smooth_scores[:, :, :n_trial_samples]
mean_scores = smooth_scores.mean(dim=-1)
# endregion
print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints and {n_samples} samples: Score method: {score_method}")
# logits[0,: ,0].min()

cal_mask = get_cal_mask(mean_scores, calibration_budget)
n_dcal = cal_mask.sum().item()

confidence = 0.99
smoothing_sigma = (0.01, 0.6)
error_correction = True


coverage_range = [0.9]

#region loding smooth logit predictions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

#region defining basic setup for conformal evaluation

y_true_mask = F.one_hot(y_true, num_classes=n_classes).bool().to(device)
mean_scores = smooth_scores.mean(dim=-1)
#endregion
print(f"Loading {dataset_name} dataset with {n_datapoints} datapoints: Score method: {score_method}")


vanilla_cp = VanillaSmoothCP(nominal_coverage=0.9)
vanilla_results = []

print("Precomputing vanilla CP")

# vanilla_cp.pre_compute(smooth_scores, y_true)

for coverage_guarantee in coverage_range:
    r = 0
    vanilla_cp.set_nominal_coverage(coverage_guarantee)

    for iter_i in range(n_iterations):
        cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
        eval_mask = ~cal_mask
        threshold = vanilla_cp.calibrate_from_scores(smooth_scores[cal_mask], y_true[cal_mask])
        pred_set = vanilla_cp.predict_from_scores(smooth_scores[eval_mask])

        empirical_coverage = vanilla_cp.internal_cp.coverage(pred_set, y_true_mask[eval_mask])
        average_set_size = pred_set.sum(dim=1).float().mean().item()

        vanilla_results.append({
            "method": "vanilla", 
            "iteration": iter_i,
            "coverage_guarantee": coverage_guarantee,
            "r": r,
            "smoothing_sigma": smoothing_sigma,
            "threshold": threshold,
            "empirical_coverage": empirical_coverage,
            "average_set_size": average_set_size,
            "score_method": score_method,
            "dataset_name": dataset_name,
            "calibration_budget": calibration_budget,
        })

vanilla_results = pd.DataFrame(vanilla_results)
vanilla_results.to_csv(f"{result_folder}/vanilla_results-{dataset_name}-smooth{smoothing_sigma}--{score_method}-nsamples{n_samples}.csv", index=False)
vanilla_results[vanilla_results["coverage_guarantee"] == 0.9].mean()


r_range = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
# r_range = [(0, 3)]
cas_results = []
bin_results = []

for r in r_range:
    print("Computing for r=", r)
    # making classes
    cas_cp = SparseCAS(nominal_coverage=0.9, r=r, smoothing_sigma=smoothing_sigma, confidence_level=confidence, n_dcal=n_dcal, n_classes=n_classes, 
                    error_correction=error_correction)
    cas_cp.pre_compute(smooth_scores, y_true)
    bin_cp = SparseBinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes,
                        r=r, confidence_level=confidence,
                        error_correction=error_correction,
                        p_base=0.905)
    

    print("CAS pre-computed")

    # bin_cp = BinCP(nominal_coverage=0.9, smoothing_sigma=smoothing_sigma, n_dcal=n_dcal, n_classes=n_classes,
    #                     r=r, confidence_level=confidence,
    #                     error_correction=error_correction,
    #                     p_base=0.6)

    # bin_cp.pre_compute(smooth_scores, y_true)
    print("bin pre-computed")

    # here goes a for
    # coverage_guarantee = 0.9
    for coverage_guarantee in coverage_range:
        print(f"Running for r={r}, coverage={coverage_guarantee}")
        cas_cp.set_nominal_coverage(coverage_guarantee)
        # bin_cp.set_nominal_coverage(coverage_guarantee)

        # here goes a for
        for iter_i in tqdm(range(n_iterations)):
            cal_mask = get_cal_mask(smooth_scores.mean(dim=-1), calibration_budget)
            eval_mask = ~cal_mask

            # evaluating cas
            # threshold_cas = cas_cp.calibrate_from_scores(smooth_scores[cal_mask], y_true[cal_mask])
            # pred_set_cas = cas_cp.predict_from_scores(smooth_scores[eval_mask])
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
                "threshold": threshold_cas,
                "empirical_coverage": empirical_coverage_cas,
                "average_set_size": average_set_size_cas,
                "score_method": score_method,
                "confidence_level": confidence,
                "dataset_name": dataset_name,
                "calibration_budget": calibration_budget,
            })
            print(f"CAS: Cov: {empirical_coverage_cas}, Size = {average_set_size_cas}")
            
            threshold_bin = bin_cp.calibrate_from_scores(smooth_scores[cal_mask], y_true[cal_mask])
            pred_set_bin = bin_cp.predict_from_scores(smooth_scores[eval_mask])

            empirical_coverage_bin = bin_cp.internal_cp.coverage(pred_set_bin, y_true_mask[eval_mask])
            average_set_size_bin = pred_set_bin.sum(dim=1).float().mean().item()

            bin_results.append({
                "method": "bin",
                "coverage_guarantee": coverage_guarantee,
                "iteration": iter_i,
                "r": r,
                "smoothing_sigma": smoothing_sigma,
                "threshold": threshold_bin,
                "empirical_coverage": empirical_coverage_bin,
                "average_set_size": average_set_size_bin,
                "score_method": score_method,
                "confidence_level": confidence,
                "dataset_name": dataset_name,
                "calibration_budget": calibration_budget,
            })
            print(f"bin: Cov: {empirical_coverage_bin}, Size = {average_set_size_bin}")

cas_results = pd.DataFrame(cas_results)
bin_results = pd.DataFrame(bin_results)

cas_results.to_csv(f"{result_folder}/cas_results-{dataset_name}-smooth{smoothing_sigma}-{score_method}-nsamples{n_samples}-conf{confidence}.csv", index=False)
bin_results.to_csv(f"{result_folder}/bin_results-{dataset_name}-smooth{smoothing_sigma}-{score_method}-nsamples{n_samples}-conf{confidence}.csv", index=False)
