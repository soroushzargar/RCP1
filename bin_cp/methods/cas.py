import torch

from bin_cp.methods.robust_cp import RobustCP
from bin_cp.helpers.tensor import bound_tensor
from bin_cp.robust.bounds import CDF_bounds_l2
from bin_cp.robust.confidence import bernstein_bound, bernstein_vectorized
from bin_cp.robust.sparse_bounds import sparse_cdf_bound

class CAS(RobustCP):
    def __init__(self, smoothing_sigma=0.0, confidence_level=0.999, n_dcal=None, n_classes=None, error_correction=True, **kwargs):
        self.smoothing_sigma = smoothing_sigma
        self.eta = 1 - confidence_level
        self.n_dcal = n_dcal
        self.n_classes = n_classes
        self.error_correction = error_correction
        
        super().__init__(**kwargs)

    def correct_coverage_guarantee(self): 
        if self.error_correction == False:
            self.final_coverage = self.nominal_coverage
            return
        self.final_coverage = self.nominal_coverage + self.eta
    
    def compute_lower_bound_scores(self, S_sampled, y=None):
        if y is not None:
            S_forY = S_sampled[torch.arange(S_sampled.shape[0]), y]
        else:
            S_forY = S_sampled

        if self.error_correction == False:
            return bound_tensor(S_forY, CDF_bounds_l2, smoothing_sigma=self.smoothing_sigma, radius=self.r, 
                    confidence=(1 - self.eta/(self.n_dcal + self.n_classes)), type="lower", bonferroni_tasks=0)
        return bound_tensor(S_forY, CDF_bounds_l2, smoothing_sigma=self.smoothing_sigma, radius=self.r, 
                     confidence=(1 - self.eta/(self.n_dcal + self.n_classes)), type="lower", bonferroni_tasks=1)
        
    def compute_upper_bound_scores(self, S_sampled, y=None):
        if self.error_correction == False:
            return bound_tensor(S_sampled, CDF_bounds_l2, smoothing_sigma=self.smoothing_sigma, radius=self.r, 
                    confidence=(1 - self.eta/(self.n_dcal + self.n_classes)), type="upper", bonferroni_tasks=0)
        return bound_tensor(S_sampled, CDF_bounds_l2, smoothing_sigma=self.smoothing_sigma, radius=self.r, 
                     confidence=(1 - self.eta/(self.n_dcal + self.n_classes)), type="upper", bonferroni_tasks=1)
        
    def compute_vanilla_scores(self, S_sampled, y=None):
        if self.error_correction == False:
            return S_sampled.mean(dim=-1)
        return bernstein_vectorized(S_sampled, confidence=(1 - self.eta/(self.n_dcal + self.n_classes)), type="upper", bonferroni_tasks=1)
        return bound_tensor(S_sampled, bernstein_bound, confidence=(1 - self.eta/(self.n_dcal + self.n_classes)),
                            type="upper", bonferroni_tasks=1)
        
class SparseCAS(CAS):
    def compute_lower_bound_scores(self, S_sampled, y=None):
        if y is not None:
            S_forY = S_sampled[torch.arange(S_sampled.shape[0]), y]
        else:
            S_forY = S_sampled

        if self.error_correction == False:
            return bound_tensor(S_forY, sparse_cdf_bound, type="lower",
                pf_plus_att=self.smoothing_sigma[0], pf_minus_att=self.smoothing_sigma[1], 
                ra=self.r[0], rd=self.r[1], eta=self.eta, error_correction=False) 
        return bound_tensor(S_forY, sparse_cdf_bound, type="lower",
                pf_plus_att=self.smoothing_sigma[0], pf_minus_att=self.smoothing_sigma[1], 
                ra=self.r[0], rd=self.r[1], eta=self.eta/(self.n_dcal + self.n_classes), error_correction=True) 