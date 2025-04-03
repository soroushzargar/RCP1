from abc import ABC
from typing import Literal
from copy import deepcopy

import torch
import torch.nn.functional as F

from bin_cp.cp.core import ConformalClassifier as CP

class RobustCP(ABC): 
    def __init__(self, r=0.0, nominal_coverage=0.9, stage: Literal["calibration", "test", "Neither"] = "calibration", **kwargs):
        """ The base class for robust conformal prediction methods.
        Parameters
        ----------
        r : float
            The perturbation radius. The radius of the ball around the clean input that contains the adversarial examples.
        nominal_coverage : float
            The nominal coverage guarantee of the conformal classifier. Specified by the user at first place.
        stage : str
            The stage of the which the robustness is applied. Either "calibration" or "test".
        """

        super().__init__()
        self.r = r

        self.nominal_coverage = nominal_coverage
        self.final_coverage = nominal_coverage
        self.conformal_threshold = None
        self.stage = stage # calibration-time or test-time robustness
        
        self.correct_coverage_guarantee()
        self.internal_cp = CP(score_pipeline=[], coverage_guarantee=self.final_coverage)

    def set_nominal_coverage(self, nominal_coverage=0.9):
        """ Set the nominal coverage guarantee of the conformal classifier.
        This method is the endpoint with the user. The user should not directly change the nominal_coverage attribute. 
        This is since the coverage guarantee is corrected for the union error bound.

        Parameters
        ----------
        nominal_coverage : float
            The nominal coverage guarantee of the conformal classifier. Specified by the user at first place.
        """
        self.nominal_coverage = nominal_coverage
        self.correct_coverage_guarantee()
        self.internal_cp.coverage_guarantee = self.final_coverage
    
    def correct_coverage_guarantee(self): # if another nominal coverage is needed due to e.g. union error bound
        """ Overwrite in the child class to correct the coverage guarantee of the conformal classifier.
        Each class updates the final_coverage attribute according to the union error bound specific to the method.
        """
        self.final_coverage = deepcopy(self.nominal_coverage)

    def calibrate_from_scores(self, S_sampled, y, return_scores=False):
        if self.stage == "calibration":
            calibration_scores = self.compute_lower_bound_scores(S_sampled, y)
        
        if self.stage == "test":
            calibration_scores = self.compute_vanilla_scores(S_sampled)
        
        # if calibration_scores.ndim == 1:
        #     y_true_mask = torch.ones(size=(calibration_scores.shape[0], 1), dtype=torch.bool)
        #     self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores, y_true_mask)
        # else:
        #     y_true_mask = F.one_hot(y, num_classes=calibration_scores.shape[1]).bool()
        #     self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores, y_true_mask)

        self.conformal_threshold = self.calibrate_from_refined_scores(calibration_scores, y)

        if return_scores:
            return self.conformal_threshold, calibration_scores
        return self.conformal_threshold
    
    def predict_from_scores(self, S_sampled, return_scores=False):
        if self.stage == "calibration":
            test_scores = self.compute_vanilla_scores(S_sampled)
        
        if self.stage == "test":
            test_scores = self.compute_upper_bound_scores(S_sampled)
        
        # pred_sets = (test_scores >= self.conformal_threshold)
        pred_sets = self.predict_from_refined_scores(test_scores)
        if return_scores:
            return pred_sets, test_scores
        return pred_sets
        
    def compute_lower_bound_scores(self, S_sampled:torch.Tensor, y=None):
        raise NotImplementedError
    
    def compute_upper_bound_scores(self, S_sampled:torch.Tensor, y=None):
        raise NotImplementedError
    
    def compute_vanilla_scores(self, S_sampled:torch.Tensor):
        return S_sampled.mean(dim=-1)
    
    def pre_compute(self, S_sampled, y):
        self.precompute__calibration_scores = self.calibrate_from_scores(S_sampled, y, return_scores=True)[1]
        self.precompute__test_scores = self.predict_from_scores(S_sampled, return_scores=True)[1]
        self.precompute_y = y

    def pre_compute_calibrate(self, cal_mask):
        calibration_scores = self.precompute__calibration_scores[cal_mask]
        return self.calibrate_from_refined_scores(calibration_scores, self.precompute_y[cal_mask])

    def pre_compute_predict(self, test_mask):
        test_scores = self.precompute__test_scores[test_mask]
        return self.predict_from_refined_scores(test_scores)

    def calibrate_from_refined_scores(self, calibration_scores, y):
        if calibration_scores.ndim == 1:
            y_true_mask = torch.ones(size=(calibration_scores.shape[0], 1), dtype=torch.bool)
            self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores.reshape(-1, 1), y_true_mask)
        else:
            y_true_mask = F.one_hot(y, num_classes=calibration_scores.shape[1]).bool()
            self.conformal_threshold = self.internal_cp.calibrate_from_scores(calibration_scores, y_true_mask)
        return self.conformal_threshold

    def predict_from_refined_scores(self, test_scores):
        pred_sets = (test_scores >= self.conformal_threshold)
        return pred_sets
    
    

class VanillaSmoothCP(RobustCP):
    def compute_lower_bound_scores(self, S_sampled, y=None):
        return self.compute_vanilla_scores(S_sampled)
    
    def compute_upper_bound_scores(self, S_sampled, y=None):
        return self.compute_vanilla_scores(S_sampled)