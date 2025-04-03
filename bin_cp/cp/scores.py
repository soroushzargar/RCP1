from abc import ABC

import torch
import torch.nn.functional as F
import numpy as np


class ScoreFunction(ABC):
    def __init__(self, **kwargs):
        self.redefined_args = kwargs

    def pipe_transform(self, logits):
        return self.transform(logits, **self.redefined_args)

    def transform(self, logits, **kwargs):
        return logits

class TPSScore(ScoreFunction):
    def transform(self, logits, **kwargs):
        res = F.softmax(logits, dim=1)
        return res
    
class LogitScore(ScoreFunction):
    def transform(self, logits, **kwargs):
        res = logits
        return res

class APSScore(ScoreFunction):
    def transform(self, logits, **kwargs):
        softmax_enabled = kwargs.get("softmax", True)
        shift_enabled = kwargs.get("shift", True) # shift to [0, 1] from [-1, 0]
        if softmax_enabled:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        ranks = torch.argsort(torch.argsort(softmaxes, axis=1), axis=1)
        cls_scores = []
        for cls_id in range(logits.shape[1]):
            y_rank = ranks[:, cls_id].reshape(-1, 1)
            larger_softmaxes = (softmaxes * ((ranks > y_rank).int())).sum(axis=1)
            u_vec = torch.rand_like(softmaxes[:, cls_id])
            # u_vec = torch.ones_like(softmaxes[:, cls_id])
            cls_result = softmaxes[:, cls_id] * u_vec + larger_softmaxes
            cls_scores.append(cls_result.reshape(-1, 1))
        result = torch.hstack(cls_scores) * -1
        if shift_enabled:
            result = result + 1
        return result

class MarginScore(ScoreFunction):
    def transform(self, logits, **kwargs):
        softmax_enabled = kwargs.get("softmax", True)
        if softmax_enabled:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits

        n_classes = softmaxes.shape[1]
        classes_scores = []
        class_idxs = np.arange(n_classes)
        for class_idx in range(n_classes):
            other_classes = np.delete(class_idxs, class_idx)
            y_embeddings = softmaxes[np.arange(softmaxes.shape[0]), class_idx]

            others_embeddings = softmaxes[:, other_classes]
            max_other_embeddings = others_embeddings.max(axis=1)[0]

            class_scores = y_embeddings - max_other_embeddings
            classes_scores.append(class_scores.reshape((-1, 1)))

        classes_scores = torch.hstack(classes_scores)
        return classes_scores

class RegularizerPenalty(ScoreFunction):
    def transform(self, logits, **kwargs):
        k_reg = kwargs.get("k_reg", 3)
        penalty = kwargs.get("penalty", 0.1)

        logits_ranks = torch.argsort(torch.argsort(logits, axis=1), axis=1)
        penalty_coef = logits_ranks - (logits.shape[1] - k_reg)
        penalty_coef[penalty_coef > 0] = 0
        res = penalty_coef * penalty
        result = logits + res
        return result

class LogVals(ScoreFunction):
    def transform(self, logits, **kwargs):
        error_val = kwargs.get("error_val", 1e-1)
        res = torch.log(logits + error_val)
        return res

class RowProbNormal(ScoreFunction):
    def transform(self, logits, **kwargs):
        mins = logits.min(axis=1)[0].reshape(-1, 1)
        sums = (logits - mins).sum(axis=1).reshape(-1, 1)
        result = (logits - mins) / sums
        return result