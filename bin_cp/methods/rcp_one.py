import torch
from typing import Literal
import math
from scipy.stats import norm
from bin_cp.methods.robust_cp import RobustCP
from bin_cp.robust.confidence import clopper_pearson_lower, clopper_pearson_upper
from sparse_smoothing.cert import regions_binary, compute_rho, binary_certificate_grid, compute_rho_for_many

class RCP1(RobustCP):
    def __init__(self, smoothing_sigma=0.0, n_dcal=None, n_classes=None,
                 scheme:Literal["guass", "sparse", "exact", "laplace-l1", "uniform", "uniform-l1", "uniform-l2"]="guass",
                 dataset_key=None,
                 **kwargs):
        self.smoothing_sigma = smoothing_sigma
        self.n_dcal = n_dcal
        self.n_classes = n_classes
        self.scheme = scheme
        self.dataset_key = dataset_key        

        super().__init__(stage="calibration", **kwargs)

    def correct_coverage_guarantee(self):
        self.final_coverage = self.compute_threat_p(
            self.nominal_coverage, sigma=self.smoothing_sigma, r=self.r, type="upper",
            scheme=self.scheme)
        return self.final_coverage

    def compute_lower_bound_scores(self, S_sampled: torch.Tensor, y=None):
        if S_sampled.ndim > 2:
            scores = S_sampled[:, :, 0]
        else:
            scores = S_sampled
        return scores
        
    def predict_from_scores(self, S_sampled, return_scores=False):
        
        if S_sampled.ndim > 2:
            scores = S_sampled[:, :, 0]
        else:
            scores = S_sampled

        pred_set = scores >= self.internal_cp.quantile_threshold
            # print(f"base={self.p_base}, conf={confidence_p}")

        return pred_set
    
    @staticmethod
    def compute_threat_p(p, r, sigma, scheme:Literal["guass", "sparse", "laplace-l1", "uniform-l1", "uniform-l2"]="guass", type:Literal["lower", "upper"]="lower", dataset_key=None):
        if scheme == "guass":
            if type == "lower":
                conf_p = norm.cdf(norm.ppf(p, scale=sigma) - r, scale=sigma)
            else:
                conf_p = norm.cdf(norm.ppf(p, scale=sigma) + r, scale=sigma)
            return conf_p
        if scheme == "exact":
            if type == "lower":
                conf_p = p - 1/(2*(sigma) * math.sqrt(3)) * r
            else:
                conf_p = p + 1/(2*(sigma) * math.sqrt(3)) * r
            return conf_p
        if scheme == "laplace-l1":
            from bin_cp.robust.noises import Laplace, get_dim
            from bin_cp.robust.robust_bounds import compute_upper_p_from_r, compute_lower_p_from_r
            r_function = Laplace(dim=get_dim(dataset_key or "cifar10"), sigma=sigma)
            if dataset_key is None:
                Warning("Dataset key is not provided. Using CIFAR10 as default")
            if type == "lower":
                conf_p = compute_lower_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            else:
                conf_p = compute_upper_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            return conf_p
        if scheme == "uniform-l1":
            from bin_cp.robust.noises import Uniform, get_dim
            from bin_cp.robust.robust_bounds import compute_upper_p_from_r, compute_lower_p_from_r
            r_function = Uniform(dim=get_dim(dataset_key or "cifar"), sigma=sigma)
            if dataset_key is None:
                Warning("Dataset key is not provided, using CIFAR10 as default")
            if type == "lower":
                conf_p = compute_lower_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            else:
                conf_p = compute_upper_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            return conf_p
        if scheme == "uniform-l2":
            from bin_cp.robust.noises import Uniform, get_dim
            from bin_cp.robust.robust_bounds import compute_upper_p_from_r, compute_lower_p_from_r
            r_function = Uniform(dim=get_dim(dataset_key or "cifar"), sigma=sigma)
            if dataset_key is None:
                Warning("Dataset key is not provided, using CIFAR10 as default")
            if type == "lower":
                conf_p = compute_lower_p_from_r(torch.tensor(p), r, r_function.certify_l2)
            else:
                conf_p = compute_upper_p_from_r(torch.tensor(p), r, r_function.certify_l2)
            return conf_p

        raise NotImplementedError("Sparse scheme is not implemented yet")

        