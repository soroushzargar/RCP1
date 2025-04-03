import torch

def standard_l2_norm(inputs, sigma=0.5):
    """ Forms an input
    """
    noise = torch.normal(0, sigma, inputs.shape).to(inputs.device)
    noisy_inputs = inputs + noise
    return noisy_inputs

