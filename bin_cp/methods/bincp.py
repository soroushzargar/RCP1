import torch
from typing import Literal
import math
from scipy.stats import norm
from bin_cp.methods.robust_cp import RobustCP
from bin_cp.robust.confidence import clopper_pearson_lower, clopper_pearson_upper
from sparse_smoothing.cert import regions_binary, compute_rho, binary_certificate_grid, compute_rho_for_many

class BinCP(RobustCP): 
    def __init__(self, smoothing_sigma=0.0, confidence_level=0.999, n_dcal=None, n_classes=None,
                 error_correction=True, lambda_base=None, p_base=None, scheme:Literal["guass", "sparse", "exact", "laplace-l1", "uniform-l1", "uniform-l2"]="guass",
                 **kwargs):
        self.smoothing_sigma = smoothing_sigma
        self.eta = 1 - confidence_level
        self.n_dcal = n_dcal
        self.n_classes = n_classes
        self.error_correction = error_correction

        self.lambda_base = lambda_base
        self.p_base = p_base
        self.p_conservative = p_base
        if self.lambda_base is not None and self.p_base is not None:
            raise ValueError("Both lambda_base and p_base cannot be set at the same time")
        if self.lambda_base is None and self.p_base is None:
            raise ValueError("Either lambda_base or p_base must be set")
        self.scheme = scheme

        super().__init__(stage="calibration", **kwargs)

    def correct_coverage_guarantee(self):
        if self.error_correction == False:
            self.final_coverage = self.nominal_coverage
            return
        self.final_coverage = self.nominal_coverage + self.eta

    def correct_coverage_guarantee(self):
        if self.error_correction == False:
            self.final_coverage = self.nominal_coverage
            return
        self.final_coverage = self.nominal_coverage + self.eta

    def compute_lower_bound_scores(self, S_sampled: torch.Tensor, y=None):
        if self.lambda_base is not None:
            # print("lambda")
            bins = (S_sampled >= self.lambda_base).sum(dim=-1)
            if self.error_correction:
                lower_probs = torch.tensor(clopper_pearson_lower(bins.cpu(), S_sampled.shape[-1], alpha=self.eta / (self.n_dcal + self.n_classes))).to(S_sampled.device)
            else:
                lower_probs = bins / S_sampled.shape[-1]
            return lower_probs

        if self.p_base is not None:
            # print("p")
            lower_lambdas = S_sampled.quantile(1 - self.p_base, dim=-1)
            if self.error_correction:
                self.p_conservative = clopper_pearson_lower(self.p_base * S_sampled.shape[-1], S_sampled.shape[-1], alpha=self.eta / (self.n_dcal + self.n_classes))
            else:
                self.p_conservative = self.p_base
            return lower_lambdas
        
    def predict_from_scores(self, S_sampled, return_scores=False):
        if self.p_base is not None:
            confidence_p = self.compute_threat_p(p=self.p_conservative, r=self.r, sigma=self.smoothing_sigma, scheme=self.scheme, type="lower")
            # print(f"base={self.p_base}, conf={confidence_p}")

            bins = (S_sampled >= self.conformal_threshold).sum(dim=-1)
            if self.error_correction:
                upper_probs = torch.tensor(clopper_pearson_upper(bins.cpu(), S_sampled.shape[-1], alpha=self.eta / (self.n_dcal + self.n_classes))).to(S_sampled.device)
            else:
                upper_probs = bins / S_sampled.shape[-1]

            pred_set = upper_probs >= confidence_p

        if self.lambda_base is not None:
            confidence_p = self.compute_threat_p(p=self.conformal_threshold, r=self.r, sigma=self.smoothing_sigma, scheme=self.scheme, type="lower")
            # print(f"base={self.conformal_threshold}, conf={confidence_p}")

            bins = (S_sampled >= self.lambda_base).sum(dim=-1)
            if self.error_correction:
                upper_probs = torch.tensor(clopper_pearson_upper(bins.cpu(), S_sampled.shape[-1], alpha=self.eta / (self.n_dcal + self.n_classes))).to(S_sampled.device)
            else:
                upper_probs = bins / S_sampled.shape[-1]
            
            pred_set = upper_probs >= confidence_p
        if return_scores:
            return pred_set, upper_probs
        return pred_set
    
    @staticmethod
    def compute_threat_p(p, r, sigma, scheme:Literal["guass", "sparse", "exact", "laplace-l1", "uniform-l1", "uniform-l2"]="guass", type:Literal["lower", "upper"]="lower", dataset_key=None):
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
            r_function = Laplace(dim=get_dim(dataset_key or "cifar"), sigma=sigma)
            if dataset_key is None:
                Warning("Dataset key is not provided, using CIFAR10 as default")
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
                conf_p = compute_lower_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            else:
                conf_p = compute_upper_p_from_r(torch.tensor(p), r, r_function.certify_l1)
            return conf_p

        raise NotImplementedError("Bounds are not yet implemented for this scheme is not implemented yet")
            

class SparseBinCP(BinCP):
    @staticmethod
    def compute_threat_p(p, r, sigma, scheme:Literal["guass", "sparse"]="guass", type:Literal["lower", "upper"]="lower"):
        if type == "lower":
            regs = regions_binary(ra=r[0], rd=r[1], pf_plus=sigma[0], pf_minus=sigma[1])
            p_conf = float(compute_rho(regions=regs, p_emp=p))
        else:
            raise NotImplementedError("Upper bound is not implemented yet")
        return p_conf
        