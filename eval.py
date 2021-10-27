import argparse
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.metrics import auc, roc_curve

from cutpaste import CPmodel, data


def eval(data_path, data_cls, batch_size, ckpt_path=None, gpu=None, model=None):

    device = torch.device('cuda:{:d}'.format(gpu))

    train_dataset = data.Dataset(data_path, data_cls, "test", "train_data_noaug")
    test_dataset = data.Dataset(data_path, data_cls, "test", "test_data_noaug")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    if model is None:
        print("loading models...")
        ckpt = torch.load(ckpt_path)
        model = CPmodel.Model(ckpt["model_name"], ckpt["aug_mode"], False)
        new_state_dict = OrderedDict()
        for k, v in ckpt["check_point"].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.to(device)

    print("computing embeddings of training and testing images...")
    model.eval()
    train_embedding = list()
    with torch.no_grad():
        for input in train_dataloader:
            input = input.to(device)
            embed, _ = model(input)
            train_embedding.append(embed)
        train_embedding = torch.cat(train_embedding, 0).to("cpu")  # too big to store on gpu

    test_embedding = list()
    test_label = list()
    with torch.no_grad():
        for input, label in test_dataloader:
            input = input.to(device)
            embed, _ = model(input)
            test_embedding.append(embed)
            test_label.append(label)
    test_embedding = torch.cat(test_embedding, 0).to("cpu")
    test_label = torch.cat(test_label, 0)
    train_embedding = torch.nn.functional.normalize(train_embedding, p=2, dim=1)
    test_embedding = torch.nn.functional.normalize(test_embedding, p=2, dim=1)
    # claculate mean
    mean = torch.mean(train_embedding, axis=0)
    inv_cov = torch.Tensor(LedoitWolf().fit(train_embedding).precision_, device="cpu")

    distances = mahalanobis_distance(test_embedding, mean, inv_cov)
    # TODO: set threshold on mahalanobis distances and use "real" probabilities

    roc_auc = plot_roc(test_label, distances * (-1))
    # print(roc_auc)
    return roc_auc


def mahalanobis_distance(values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor) -> torch.Tensor:
    """Compute the batched mahalanobis distance.
    values is a batch of feature vectors.
    mean is either the mean of the distribution to compare, or a second
    batch of feature vectors.
    inv_covariance is the inverse covariance of the target distribution.
    """
    assert values.dim() == 2
    assert 1 <= mean.dim() <= 2
    assert len(inv_covariance.shape) == 2
    assert values.shape[1] == mean.shape[-1]
    assert mean.shape[-1] == inv_covariance.shape[0]
    assert inv_covariance.shape[0] == inv_covariance.shape[1]

    if mean.dim() == 1:  # Distribution mean.
        mean = mean.unsqueeze(0)
    x_mu = values - mean  # batch x features
    # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
    dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
    return dist.sqrt()


def plot_roc(labels, scores, filename="", modelname="", save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver operating characteristic {modelname}")
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/mxh/anomaly_data/")
    parser.add_argument("--class", default="bottle", dest="cls")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--ckpt-path", default="test.pth")
    args = parser.parse_args()
    eval(args.data, args.cls, args.batch_size, args.ckpt_path, args.gpu)
