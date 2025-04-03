from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd
# from gnn_cp.data.data_manager import GraphDataManager
from bin_cp.cp.scores import APSScore, TPSScore


class ConformalClassifier(object):
    def __init__(self, score_pipeline: list, model=None, coverage_guarantee=0.9, confidence_offset=0.0):
        """
        Initialize the VanillaCP object.

        Args:
            transformation_sequence (list): A list of transformations to be applied to the logits to get the scores.
            model: The machine learning model to be used for getting logits. If none, the object can work only with scores.
            coverage_guarantee (float): The desired coverage guarantee for the confidence intervals.
            confidence_offset (float): The offset to be added to the confidence guarantee in case of finite sample correction.

        Attributes:
            cached_scores: Cached scores for efficiency.
            score_quantile: The quantile of the scores used for calculating the confidence intervals.
        """
        self.score_pipeline = score_pipeline
        self.model = model
        self.coverage_guarantee = coverage_guarantee + confidence_offset
        self.cached_scores = None
        self.quantile_threshold = None


    @staticmethod
    def weighted_quantile(arr, q, weights=None, res_weight=1):
        """
        Compute the weighted quantile of a given array.

        Parameters:
        arr (torch.Tensor): The input array.
        q (float): The quantile value to compute.
        weights (torch.Tensor, optional): The weights associated with each element of the array. Defaults to all ones.
        res_weight (float, optional): The weight associated with the residual element x_{n+1}. Defaults to 1.

        Returns:
        float: The weighted quantile value.
        """
        
        if weights is None:
            weights = torch.ones_like(arr)

        weights = weights / weights.max()

        weights = torch.concat([weights, torch.tensor([res_weight]).to(weights.device)])

        val_sorted, arg_sort = torch.sort(arr)
        sorted_weights = weights[arg_sort] / (weights.sum())
        idx = (sorted_weights.cumsum(dim=0) < q).sum() - 1
        if idx < 0:
            idx = 0  
        return val_sorted[idx]

    @staticmethod
    def return_possible_alpha(in_val, q, weights=None):
        if weights is None:
            weights = torch.ones_like(in_val)
        
        # print(weights)
        weights = weights / weights.max()
        # print(weights)

        val_sorted, arg_sort = torch.sort(in_val)
        sorted_weights = weights[arg_sort] / (weights.sum() + 1)
        idx = (sorted_weights.cumsum(dim=0) < q).sum() - 1

        return sorted_weights[:idx+1].sum()

    # region Basic Functions:: Scoring
    def get_scores_from_logits(self, logits, test_idx=None):
        res = torch.clone(logits)
        for transformation_item in self.score_pipeline:
            res = transformation_item.pipe_transform(res)
        if test_idx is not None:
            res = res[test_idx]
        return res

    def get_scores(self, X, test_idx=None, model=None):
        if model is None:
            model = self.model
        logits = model.predict(X)
        result = self.get_scores_from_logits(logits)
        if test_idx is not None:
            return result[test_idx]
        return result

    # region Basic Functions:: Calibration
    def calibrate_from_logits(self, logits, y_true_mask):
        scores = self.get_scores_from_logits(logits)
        quantile_val = self.calibrate_from_scores(scores, y_true_mask)
        return quantile_val

    def calibrate_from_scores(self, scores, y_true_mask):
        score_points = scores[y_true_mask]
        # quantile_idx = self.get_quantile_idx(n_points=score_points.shape[0])
        # sorted_scores = torch.sort(score_points)[0]
        # self.cached_scores = sorted_scores
        # self.score_quantile = self.cached_scores[quantile_idx].item()

        n = score_points.shape[0]
        # alpha_q =  (1.0 - 1.0 / (n + 1)) * (1 - self.coverage_guarantee)
        alpha_q =  (1 - self.coverage_guarantee)
        self.quantile_threshold = self.weighted_quantile(score_points, alpha_q).item()
        self.cached_scores = score_points.clone()

        if alpha_q < 1 / n:
            self.quantile_threshold = -1000

        return self.quantile_threshold

    def weighted_calibrate_from_scores(self, scores, y_true_mask, weights=None, res_weight=1):
        score_points = scores[y_true_mask]
        n = score_points.shape[0]
        # alpha_q = (1.0 - 1.0 / (n + 1)) * (1 - self.coverage_guarantee)
        alpha_q = (1 - self.coverage_guarantee)
        # print(f"alpha q = {alpha_q}")
        self.weighted_q = self.weighted_quantile(score_points, alpha_q, weights=weights, res_weight=res_weight).item()
        self.quantile_threshold = self.weighted_q
        return self.weighted_q

    def calibrate(self, X, y_true_mask, test_idx=None, model=None, y_overall=False):
        if test_idx is not None and y_overall:
            true_mask = y_true_mask[test_idx]
        else:
            true_mask = y_true_mask
        scores = self.get_scores(X, test_idx, model)
        quantile_val = self.calibrate_from_scores(scores, true_mask)
        return quantile_val
    # endregion

    # region Basic FunctionsL:: Utils
    def change_coverage_guarantee(self, new_coverage_guarantee):
        self.coverage_guarantee = new_coverage_guarantee
        if self.cached_scores is not None:
            score_points = self.cached_scores
            n = score_points.shape[0]
            alpha_q = (1 - self.coverage_guarantee) - (1 / (n + 1)) * (1 - self.coverage_guarantee)
            self.quantile_threshold = torch.quantile(score_points, alpha_q).item()
            # quantile_idx = self.get_quantile_idx(n_points=self.cached_scores.shape[0])
            # self.score_quantile = self.cached_scores[quantile_idx].item

    def get_quantile_idx(self, n_points):
        alpha = 1 - self.coverage_guarantee
        q_idx = int((n_points - 1) * alpha)
        return q_idx
    # endregion

    # region Basic Functions:: Prediction
    def predict_from_scores(self, scores):
        result = scores > self.quantile_threshold
        return result

    def predict_from_logits(self, logits):
        scores = self.get_scores_from_logits(logits)
        return self.predict_from_scores(scores)
    # endregion

    # region builtin cps
    @classmethod
    def aps_graph_cp(cls, coverage_guarantee=0.9, model=None):
        return cls(transformation_sequence=[
            APSScore(softmax=True)
        ], coverage_guarantee=coverage_guarantee, model=model)
    @classmethod
    def tps_graph_cp(cls, coverage_guarantee=0.9, model=None):
        return cls(transformation_sequence=[
            TPSScore(softmax=True)
        ], coverage_guarantee=coverage_guarantee, model=model)
    # endregion

    # region Metric Functions
    @staticmethod
    def average_set_size(prediction_sets, count_empty=True):
        set_size_vals = prediction_sets.sum(axis=1)
        if count_empty:
            return set_size_vals.float().mean().item()
        result = set_size_vals[set_size_vals != 0].float().mean()
        return result.item()

    @staticmethod
    def coverage(prediction_sets, y_true_mask):
        cov = (prediction_sets[y_true_mask].sum() / y_true_mask.sum()).item()
        return cov

    @staticmethod
    def argmax_accuracy(scores, y_true):
        y_pred = scores.int().argmax(axis=1)
        y_true_idx = y_true.int().argmax(axis=1)
        res = accuracy_score(
            y_true=y_true_idx.cpu().numpy(),
            y_pred=y_pred.cpu().numpy()
        )
        return res
    # endregion

   