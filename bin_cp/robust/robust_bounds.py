import numpy as np
import torch

def compute_upper_p_from_r(p, r, certified_radius):
    r_max = certified_radius(1 - torch.tensor(p))
    
    r_residual = r_max - r

    p_min = 0
    p_max = 1

    while p_max - p_min > 1e-6:
        p_mid = (p_min + p_max) / 2
        r_mid = certified_radius(1 - torch.tensor(p_mid))
        # print(p_mid, r_mid, r_residual)
        if r_mid >= r_residual:
            p_min = p_mid
        else:
            p_max = p_mid
    solution = p_mid
    return solution

def compute_lower_p_from_r(p, r, certified_radius):
    r_max = certified_radius(1 - torch.tensor(p))
    
    r_residual = r_max + r

    p_min = 0
    p_max = 1

    while p_max - p_min > 1e-6:
        p_mid = (p_min + p_max) / 2
        r_mid = certified_radius(1 - torch.tensor(p_mid))
        # print(p_mid, r_mid, r_residual)
        if r_mid >= r_residual:
            p_min = p_mid
        else:
            p_max = p_mid
    solution = p_mid
    return solution