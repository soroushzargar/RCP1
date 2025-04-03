import torch
import numpy as np
from statsmodels.stats.proportion import proportion_confint

def hoefding_bound(randoms, confidence=0.95, type="lower", bonferroni_tasks=1):
    eta = 1 - confidence
    if type == "lower":
        return randoms.mean() - np.sqrt(np.log((1 * bonferroni_tasks)/eta) / (2 * randoms.shape[0]))
    elif type == "upper":
        return randoms.mean() + np.sqrt(np.log((1 * bonferroni_tasks)/eta) / (2 * randoms.shape[0]))
    elif type == "both":
        return randoms.mean() - np.sqrt(np.log((2 * bonferroni_tasks)/eta) / (2 * randoms.shape[0])), randoms.mean() + np.sqrt(np.log((2 * bonferroni_tasks)/eta) / (2 * randoms.shape[0]))
    
def bernstein_bound(randoms, confidence=0.95, type="lower", bonferroni_tasks=1):
    eta = 1 - confidence
    if type == "lower":
        lower = randoms.mean() - np.sqrt(2 * np.var(randoms.cpu().numpy()) * np.log((2 * bonferroni_tasks)/eta) / (randoms.cpu().numpy().shape[0])) - (7 * np.log((2 * bonferroni_tasks)/eta) / (3 * (randoms.shape[0] - 1)))
        return lower
    elif type == "upper":
        upper = randoms.mean() + np.sqrt(2 * np.var(randoms.cpu().numpy()) * np.log((2 * bonferroni_tasks)/eta) / (randoms.cpu().numpy().shape[0])) + (7 * np.log((2 * bonferroni_tasks)/eta) / (3 * (randoms.shape[0] - 1)))
        return upper
    elif type == "both":
        lower = randoms.mean() - np.sqrt(2 * np.var(randoms.cpu().numpy()) * np.log((4 * bonferroni_tasks)/eta) / (randoms.cpu().numpy().shape[0])) - (7 * np.log((4 * bonferroni_tasks)/eta) / (3 * (randoms.shape[0] - 1)))
        upper = randoms.mean() + np.sqrt(2 * np.var(randoms.cpu().numpy()) * np.log((4 * bonferroni_tasks)/eta) / (randoms.cpu().numpy().shape[0])) + (7 * np.log((4 * bonferroni_tasks)/eta) / (3 * (randoms.shape[0] - 1)))
        return lower, upper
    
def bernstein_vectorized(randoms, confidence=0.95, type="lower", bonferroni_tasks=1):
    mean = randoms.mean(dim=-1)
    var = randoms.var(dim=-1)
    eta = 1 - confidence
    if type == "lower":
        bonferroni_eta_values = torch.log(torch.tensor((2 * bonferroni_tasks)/eta))
        lower = mean - torch.sqrt(2 * var * bonferroni_eta_values / (randoms.shape[-1])) - (7 * bonferroni_eta_values / (3 * (randoms.shape[-1] - 1)))
        return lower
    elif type == "upper":
        bonferroni_eta_values = torch.log(torch.tensor((2 * bonferroni_tasks)/eta))
        upper = mean + torch.sqrt(2 * var * bonferroni_eta_values / (randoms.shape[-1])) + (7 * bonferroni_eta_values / (3 * (randoms.shape[-1] - 1)))
        return upper
    elif type == "both":
        bonferroni_eta_values = torch.log(torch.tensor((4 * bonferroni_tasks)/eta))
        lower = mean - torch.sqrt(2 * var * bonferroni_eta_values / (randoms.shape[-1])) - (7 * bonferroni_eta_values / (3 * (randoms.shape[-1] - 1)))
        upper = mean + torch.sqrt(2 * var * bonferroni_eta_values / (randoms.shape[-1])) + (7 * bonferroni_eta_values / (3 * (randoms.shape[-1] - 1)))
        return lower, upper

def dvoretzky_keifer_wolfowitz_cdf_correction(randoms, confidence=0.95, type="lower", bins=None, n_bins=1000, bonferroni_tasks=1):
    eta = 1 - confidence
    if bins is None:
        bins = torch.linspace(0, 1, n_bins)
    if bins == "adaptive":
        bins = randoms[:n_bins].clone()
        bins = torch.cat([torch.tensor([0]).to(randoms.device), bins, torch.tensor([1]).to(randoms.device)]).unique()

    if type == "lower" or type == "upper":
        error = np.sqrt(np.log(1 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))
    elif type == "both":
        error = np.sqrt(np.log(2 * bonferroni_tasks/eta) / (2 * randoms.shape[0]))

    empi_cdf = ((randoms.view(-1, 1) <= bins.to(randoms.device)).sum(dim=0) / randoms.shape[0])
    empi_lower = torch.maximum(empi_cdf - error, torch.zeros_like(empi_cdf))
    empi_upper = torch.minimum(empi_cdf + error, torch.ones_like(empi_cdf))

    if type == "lower":
        return empi_lower, bins
    elif type == "upper":
        return empi_upper, bins
    elif type == "both":
        return empi_lower, empi_upper, bins
    
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
    
def dvoretzky_keifer_wolfowitz_bound(randoms, confidence=0.95, type="lower", bins=None, n_bins=1000, bonferroni_tasks=1):
    eta = 1 - confidence
    if type == "lower" or type == "both":
        cdf_upper, bins = dvoretzky_keifer_wolfowitz_cdf_correction(randoms, confidence=confidence, type="upper", bins=bins, n_bins=n_bins, bonferroni_tasks=bonferroni_tasks)
        lower_bound = mean_bound_from_cdf(cdf_upper, bins, type="lower")
    if type == "upper" or type == "both":
        cdf_lower, bins = dvoretzky_keifer_wolfowitz_cdf_correction(randoms, confidence=confidence, type="lower", bins=bins, n_bins=n_bins, bonferroni_tasks=bonferroni_tasks)
        upper_bound = mean_bound_from_cdf(cdf_lower, bins, type="upper")
    
    if type == "lower":
        return lower_bound
    elif type == "upper":
        return upper_bound
    elif type == "both":
        return lower_bound, upper_bound
    

def anderson_ci(x, alpha, type="upper"):
    n = len(x)
    i = np.arange(1, n + 1)
    if type == "upper" or type == "lower":
        u_DKW = np.maximum(0, i / n - np.sqrt(np.log(1 / alpha) / (2 * n)))
    elif type == "both":
        u_DKW = np.maximum(0, i / n - np.sqrt(np.log(2 / alpha) / (2 * n)))

    zu_i = np.sort(x)
    zu_iplus1 = np.append(zu_i, 1)[1:]
    zl_i = np.flip(1 - zu_i)
    zl_iplus1 = np.append(zl_i, 1)[1:]
    u = 1 - np.sum(u_DKW * (zu_iplus1 - zu_i))
    l = np.sum(u_DKW * (zl_iplus1 - zl_i))
    if type == "upper":
        return u
    elif type == "lower":
        return l
    return l, u


def clopper_pearson_lower(n_bins, n_samples, alpha=0.05):
    p_lower = proportion_confint(
        n_bins, n_samples, alpha=alpha, method="beta")[0]
    return p_lower

def clopper_pearson_upper(n_bins, n_samples, alpha=0.05):
    p_lower = proportion_confint(
        n_bins, n_samples, alpha=alpha, method="beta")[1]
    return p_lower

def dkw_cdf(randoms, confidence=0.95, type="lower", bonferroni_tasks=1, n_bins=-1, added_bins=[]):
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
        
        return m_empi_cdf, bins
    
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
        return m_empi_cdf, bins