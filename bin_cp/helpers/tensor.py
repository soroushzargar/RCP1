import torch

def get_smooth_scores(smooth_logits, cp, mean=False):
    results = torch.stack([cp.get_scores_from_logits(smooth_logits[:, i, :]) for i in range(smooth_logits.shape[1])])
    results = results.permute(1, 2, 0)
    if mean:
        results = results.mean(dim=-1)
    return results

def get_cal_mask(vals_tensor, fraction=0.1):
    perm = torch.randperm(vals_tensor.shape[0])
    mask = torch.zeros((vals_tensor.shape[0]), dtype=bool)
    cutoff_index = int(vals_tensor.shape[0] * fraction)
    mask[perm[:cutoff_index]] = True
    return mask

def quantization_pdf(sampled_scores, bins, dim=-1, return_type='density'):
    # Calculate comparison matrix
    comp = (torch.moveaxis(sampled_scores, dim, -1).unsqueeze(-1) < bins[None, :])
    
    # Sum over the last dimension, and divide by the shape of the last dimension
    # to get the quantile CDF
    qcdf = comp.sum(dim=-2) 
    if return_type == 'density':
        qcdf = qcdf / sampled_scores.shape[dim]
    
    # Calculate the PDF from the CDF
    qpdf = qcdf[..., 1:] - qcdf[..., :-1]
    
    # If needed, move the axes back to their original positions
    # qpdf = torch.moveaxis(qpdf, -1, dim)
    
    return qpdf

def bound_tensor(input_tensor, func, **kwargs):
    if input_tensor.ndim == 3:    
        result = torch.zeros_like(input_tensor[:, :, 0])
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                result[i, j] = func(input_tensor[i, j, :], **kwargs)
    elif input_tensor.ndim == 2:
        result = torch.zeros_like(input_tensor[:, 0])
        for i in range(input_tensor.shape[0]):
            result[i] = func(input_tensor[i, :], **kwargs)
    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")
    return result