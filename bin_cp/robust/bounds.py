import torch
from scipy.stats import norm
import numpy as np

from bin_cp.robust.confidence import hoefding_bound, bernstein_bound, dvoretzky_keifer_wolfowitz_bound


def mean_bound_from_cdf(cdf, bins, type="lower"):
    binwidth = bins[1:] - bins[:-1]

    if type == "lower" or type == "both":
        lower_bound = bins[-2] - torch.sum((cdf[1:-1]) * binwidth[:-1])
    if type == "upper" or type == "both":
        upper_bound = bins[-1] - torch.sum((cdf[1:-1]) * binwidth[1:])

    if type == "lower":
        return lower_bound
    elif type == "upper":
        return upper_bound

def mean_bounds_l2(randoms, smoothing_sigma, radius, confidence=0.95, type="lower", mc_correction="hoef", bonferroni_tasks=1):
    if type == "upper":
        if bonferroni_tasks == 0:
            corrected_upperbound_mean = randoms.mean()
        elif mc_correction == "hoef":
            corrected_upperbound_mean = torch.minimum(hoefding_bound(randoms, confidence, type="upper", bonferroni_tasks=bonferroni_tasks), torch.tensor(1))
        elif mc_correction == "bern":
            corrected_upperbound_mean = torch.minimum(bernstein_bound(randoms, confidence, type="upper", bonferroni_tasks=bonferroni_tasks), torch.tensor(1))
        elif mc_correction == "dkw":
            corrected_upperbound_mean = torch.minimum(dvoretzky_keifer_wolfowitz_bound(randoms, confidence, type="upper", bonferroni_tasks=bonferroni_tasks, bins="adaptive", n_bins=-1), torch.tensor(1))
        result = norm.cdf(
            norm.ppf(corrected_upperbound_mean.cpu(), scale=smoothing_sigma) + radius,
            scale=smoothing_sigma)
    
    elif type == "lower":
        if bonferroni_tasks == 0:
            corrected_lowerbound_mean = randoms.mean()
        elif mc_correction == "hoef":
            corrected_lowerbound_mean = torch.maximum(hoefding_bound(randoms, confidence, type="lower", bonferroni_tasks=bonferroni_tasks), torch.tensor(0))
        elif mc_correction == "bern":
            corrected_lowerbound_mean = torch.maximum(bernstein_bound(randoms, confidence, type="lower", bonferroni_tasks=bonferroni_tasks), torch.tensor(0))
        elif mc_correction == "dkw":
            corrected_lowerbound_mean = torch.maximum(dvoretzky_keifer_wolfowitz_bound(randoms, confidence, type="lower", bonferroni_tasks=bonferroni_tasks, bins="adaptive", n_bins=-1), torch.tensor(0))
        result = norm.cdf(
            norm.ppf(corrected_lowerbound_mean.cpu(), scale=smoothing_sigma) - radius,
            scale=smoothing_sigma)
        
    return result
    
def CDF_bounds_l2(randoms, smoothing_sigma, radius, confidence=0.95, type="lower", bonferroni_tasks=1, n_bins=-1):
    if type == "upper":
        eta = 1 - confidence
        if bonferroni_tasks == 0:
            error = 0
        else:
            error = np.sqrt(np.log(1 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))

        bins = randoms[:n_bins].clone()
        bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device)]).unique()

        m_empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
        m_empi_cdf = torch.clamp(m_empi_cdf - error, 0, 1)
        m_empi_cdf_upper = torch.tensor(norm.cdf(norm.ppf(m_empi_cdf.cpu(), scale=smoothing_sigma) - radius, scale=smoothing_sigma)).to(randoms.device)
        upper_bound = mean_bound_from_cdf(m_empi_cdf_upper, bins, type="upper")
        return upper_bound
    
    elif type == "lower":
        eta = 1 - confidence
        if bonferroni_tasks == 0:
            error = 0
        else:
            error = np.sqrt(np.log(1 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))

        bins = randoms[:n_bins].clone()
        bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device)]).unique()

        m_empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
        m_empi_cdf = torch.clamp(m_empi_cdf + error, 0, 1)
        m_empi_cdf_lower = torch.tensor(norm.cdf(norm.ppf(m_empi_cdf.cpu(), scale=smoothing_sigma) + radius, scale=smoothing_sigma)).to(randoms.device)
        lower_bound = mean_bound_from_cdf(m_empi_cdf_lower, bins, type="lower")
        return lower_bound
    
def CDF_adv_l2(randoms, smoothing_sigma, radius, confidence=0.95, type="lower", bonferroni_tasks=1, n_bins=-1, added_bins=[]):
    if type == "lower":
        eta = 1 - confidence
        if bonferroni_tasks == 0:
            error = 0
        else:
            error = np.sqrt(np.log(1 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))

        bins = randoms[:n_bins].clone()
        bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device), torch.tensor(added_bins).to(randoms.device)]).unique()

        m_empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
        m_empi_cdf = torch.clamp(m_empi_cdf - error, 0, 1)
        m_empi_cdf_upper = torch.tensor(norm.cdf(norm.ppf(m_empi_cdf.cpu(), scale=smoothing_sigma) - radius, scale=smoothing_sigma)).to(randoms.device)
        return m_empi_cdf_upper, bins
        # upper_bound = mean_bound_from_cdf(m_empi_cdf_upper, bins, type="upper")
        # return upper_bound
    
    elif type == "upper":
        eta = 1 - confidence
        if bonferroni_tasks == 0:
            error = 0
        else:
            error = np.sqrt(np.log(1 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))

        bins = randoms[:n_bins].clone()
        bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device), torch.tensor(added_bins).to(randoms.device)]).unique()

        m_empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
        m_empi_cdf = torch.clamp(m_empi_cdf + error, 0, 1)
        m_empi_cdf_lower = torch.tensor(norm.cdf(norm.ppf(m_empi_cdf.cpu(), scale=smoothing_sigma) + radius, scale=smoothing_sigma)).to(randoms.device)
        return m_empi_cdf_lower, bins
        # lower_bound = mean_bound_from_cdf(m_empi_cdf_lower, bins, type="lower")
        # return lower_bound

def empirical_cdf(randoms, n_bins=-1):
    bins = randoms[:n_bins].clone()
    bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device)]).unique()

    m_empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
    m_empi_cdf = torch.clamp(m_empi_cdf, 0, 1)
    return m_empi_cdf, bins
