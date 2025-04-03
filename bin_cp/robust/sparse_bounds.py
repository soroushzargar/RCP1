# This functions adapted from https://github.com/abojchevski/sparse_smoothing
import torch
import numpy as np
import gmpy2
from tqdm import tqdm
# from statsmodels.stats.proportion import proportion_confint
import numpy as np

import torch_geometric
from torch_geometric.data import Data as GraphData
import cvxpy as convex


def regions_binary(ra, rd, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    ra: int
        Number of ones y has added to x
    rd : int
        Number of ones y has deleted from x
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under px and px_tilde,
    """

    pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
    with gmpy2.context(precision=precision):
        if pf_plus == 0:
            px = pf_minus ** rd
            px_tilde = pf_minus ** ra

            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0]
                             ])

        if pf_minus == 0:
            px = pf_plus ** ra
            px_tilde = pf_plus ** rd
            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0],
                             ])
        max_q = ra + rd
        i_vec = np.arange(0, max_q + 1)

        T = ra * ((pf_plus / (1 - pf_plus)) ** i_vec) + \
            rd * ((pf_minus / (1 - pf_minus)) ** i_vec)

        ratio = np.zeros_like(T)
        px = np.zeros_like(T)
        px[0] = 1

        for q in range(0, max_q + 1):
            ratio[q] = (pf_plus/(1-pf_minus)) ** (q - rd) * \
                (pf_minus/(1-pf_plus)) ** (q - ra)

            if q == 0:
                continue

            for i in range(1, q + 1):
                px[q] = px[q] + ((-1) ** (i + 1)) * T[i] * px[q - i]
            px[q] = px[q] / q

        scale = ((1-pf_plus) ** ra) * ((1-pf_minus) ** rd)

        px = px * scale

        regions = np.column_stack((px, px / ratio, ratio))
        if pf_plus+pf_minus > 1:
            # reverse the order to maintain decreasing sorting
            regions = regions[::-1]
        return regions


def sparse_mean_bound(randoms,  
                      pf_plus_att, pf_minus_att, ra=10, rd=1, type="lower",
                      eta=0.05, bonferroni_tasks=1, error_correction=True):
    if isinstance(randoms, torch.Tensor):
        randoms = randoms.cpu().numpy()
    
    if error_correction:
        eps = np.sqrt(np.log(2 * bonferroni_tasks/eta) / (2 * randoms.shape[-1]))
    else:
        eps = 0

    p_emp = randoms.mean() + eps

    reg = regions_binary(ra=ra, rd=rd, pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    a = 0.0
    b = 1

    h = convex.Variable(len(reg))
    if type == "lower":
        result = convex.Problem(convex.Minimize(convex.sum(h)), [h>=reg[:, 0]*a, h<=reg[:, 0]*b, h@reg[:, 2] == p_emp]).solve()
    elif type == "upper":
        result = convex.Problem(convex.Maximize(convex.sum(h)), [h>=reg[:, 1]*a, h<=reg[:, 1]*b, h@reg[:, 2] == p_emp]).solve()
    return result


def sparse_cdf_bound(randoms, 
                     pf_plus_att, pf_minus_att, ra=10, rd=1, type="lower",
                     eta=0.05, bonferroni_tasks=1, error_correction=True,
                     num_s=1000):
    if isinstance(randoms, torch.Tensor):
        randoms = randoms.cpu().numpy()
    
    if error_correction:
        eps = np.sqrt(np.log(2 * bonferroni_tasks/eta) / (2 * randoms.shape[-1]))
    else:
        eps = 0
    
    p_emp = randoms.mean() + eps

    reg = regions_binary(ra=ra, rd=rd, pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    a = 0
    b = 1

    # s = np.linspace(a, b, num_s)[1:-1]
    s = np.unique(np.concatenate([np.array([a, b]), randoms]))[1:-1]

    # CDF-based upper bound
    h = convex.Variable((len(s), len(reg)))

    if type == "upper":
        result = convex.Problem(convex.Maximize(s[0] + h@reg[:, 1] @ np.diff(list(s) + [b])[::-1]),
            [h>=0, h<=1, 
                h@reg[:, 0] ==  np.minimum((randoms[:, None] >= s[::-1]).mean(0) + eps, 1) ]).solve(solver='MOSEK')
    elif type == "lower":
        result = convex.Problem(convex.Minimize(a + h@reg[:, 1] @ np.diff([a] + list(s))[::-1]),
            [h>=0, h<=1, 
                h@reg[:, 0] ==  np.maximum((randoms[:, None] >= s[::-1]).mean(0) - eps, 0) ]).solve(solver='MOSEK')
    return result
