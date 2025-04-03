import os
import yaml
from ml_collections import ConfigDict

import torch

from bin_cp.helpers.lightner import Output


def smooth_prediction_filename(dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, r=0):
    logits_file_name = f"{dataset_name}-clean-npoints_{n_datapoints}-modelsigma_{str(model_sigma).replace('.', '_')}-smoothing_{str(smoothing_sigma).replace('.', '_')}-samples_{n_samples}"
    if r > 0:
        logits_file_name += f"-r_{str(r)}"
    logits_file_name += ".pth"
    return logits_file_name

def load_smooth_prediction(dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, r=0, 
                           models_dir=None, dataset_dir=None, logits_dir=None, config_file=None):
    """Load the smooth predictions from the file
    Args:
        dataset_name (str): Name of the dataset
        model_sigma (float): Sigma of the model
        n_datapoints (int): Number of datapoints
        smoothing_sigma (float): Sigma of the smoothing
        n_samples (int): Number of samples
        r (float, optional): Radius. Defaults to 0.
        models_dir (str, optional): Directory of the models. Defaults to None.
        dataset_dir (str, optional): Directory of the dataset. Defaults to None.
        logits_dir (str, optional): Directory of the logits. Defaults to None.
        config_file (str, optional): Configuration file. Defaults to None.
    Returns:
        Output: The prediction object which contains three componenets:
            y_pred: The predicted labels
            logits: The logits
            y_true: The true labels
    """
    if models_dir is None or dataset_dir is None or logits_dir is None:
        general_config = yaml.safe_load(open("../conf/general.yaml", "r"))
        conf = ConfigDict(general_config["general"])
    else:
        conf = ConfigDict(
            {
                "models_dir": models_dir,
                "dataset_dir": dataset_dir,
                "logits_dir": logits_dir,
            }
        )
    
    logits_file_name = smooth_prediction_filename(dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples, r)
    clean_d = torch.load(os.path.join(conf.logits_dir, logits_file_name))
    y_pred = clean_d["y_pred"]
    logits = clean_d["logits"]
    y_true = clean_d["y_true"]
    prediction = Output(y_pred=y_pred, logits=logits, y_true=y_true)
    return prediction