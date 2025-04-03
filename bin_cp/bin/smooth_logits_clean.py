import os

import yaml
from ml_collections import ConfigDict

import torch
from torch.utils.data import DataLoader, Subset

from bin_cp.experiments.image_utils.cifar_resnet import ResNet
from bin_cp.experiments.image_utils.architectures import get_architecture
from bin_cp.experiments.image_utils.image_datasets import get_dataset
from bin_cp.helpers.lightner import ModelManager, Output
from bin_cp.robust.smoothing import standard_l2_norm
from bin_cp.helpers.storage import smooth_prediction_filename

from sacred import Experiment
ex = Experiment('SmoothPredictions')

@ex.config
def config():
    dataset_name = "cifar10"
    model_sigma = 0.25
    n_datapoints = 2048
    smoothing_sigma = 0.25
    n_samples = 10000

@ex.automain
def run(dataset_name, model_sigma, n_datapoints, smoothing_sigma, n_samples):
    # Loading and processing configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    general_config = yaml.safe_load(open("../conf/general.yaml", "r"))
    conf = ConfigDict(general_config["general"])
    default_models = general_config["models"]
    model_name = default_models[dataset_name]

    # Loading model
    model_file = os.path.join(conf.models_dir, dataset_name, model_name, f"noise_{model_sigma}", "checkpoint.pth.tar")
    model_dict = torch.load(model_file)
    model = get_architecture(model_dict["arch"], dataset_name)
    model.load_state_dict(model_dict["state_dict"])
    model_obj = ModelManager(model, device=device)

    # Loading dataset
    dataset = get_dataset('cifar10', 'test', root=conf.dataset_dir)
    print(f"dataset size = {len(dataset)}")
    subset_indices = list(range(0, n_datapoints, ))
    dataset = Subset(dataset, subset_indices)
    print(f"dataset size = {len(dataset)}")

    test_dataset = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    # Creating or loading logits file
    logits_file_name = smooth_prediction_filename(dataset_name=dataset_name,
        model_sigma=model_sigma,
        n_datapoints=n_datapoints,
        smoothing_sigma=smoothing_sigma,
        n_samples=n_samples)

    try:
        clean_d = torch.load(os.path.join(conf.logits_dir, logits_file_name))
        y_pred = clean_d["y_pred"]
        logits = clean_d["logits"]
        y_true = clean_d["y_true"]
        prediction = Output(y_pred=y_pred, logits=logits, y_true=y_true)
        print("Loaded logits from file")
    except Exception as e:
        print(f"Error loading logits from file: {e}")
        print("Computing logits")
        prediction = model_obj.smooth_predict(test_dataset, n_samples=n_samples, smoothing_function=lambda x: standard_l2_norm(x, sigma=smoothing_sigma))
        torch.save({
            "y_pred":prediction.y_pred, "logits": prediction.logits, "y_true": prediction.y_true
            }, os.path.join(conf.logits_dir, logits_file_name))
    acc = (prediction.y_pred == prediction.y_true).float().mean().item()
    print(f"Accuracy = {acc:.3f}")
