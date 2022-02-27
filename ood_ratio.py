import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

from model import PixelCNN
from utils import discretized_mix_logistic_prob

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local-model-path", type=str, default='./models/pcnn_lr:0.00020_nr-resnet5_nr-filters160_19.pth', help="pre-trained model path")
parser.add_argument("-g", "--global-model-path", type=str, default='./models/pcnn_lr:0.00020_nr-resnet5_nr-filters160_429.pth', help="pre-trained model path")
args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToTensor(),
    rescaling])

cifar_val = torchvision.datasets.CIFAR10('./Data',
                                          train=False, 
                                          download=True, 
                                          transform=transform)
cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

svhn_val = torchvision.datasets.SVHN('./Data',
                                     split='test',
                                     download=True,
                                     transform=transform)
svhn_val.data = svhn_val.data[:10000]
svhn_loader = data.DataLoader(svhn_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

def get_model(model_path, remove_prefix=False):
    model = PixelCNN(nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10).to(device)

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if remove_prefix:
            name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

local_model = get_model(args.local_model_path)
full_model = get_model(args.global_model_path)

with torch.no_grad():
    cifar_scores_full  = []
    cifar_scores_local = []
    cifar_scores = []
    for x, y in tqdm(cifar_loader):
        x = x.to(device)
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        full_log_probs  = discretized_mix_logistic_prob(x, full_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)
        full_log_prob   = torch.sum(torch.sum(full_log_probs,  dim=2), dim=1)
        score = full_log_prob - local_log_prob
        cifar_scores.append(score)
        cifar_scores_full.append(full_log_prob)
        cifar_scores_local.append(local_log_prob)
    cifar_score = torch.cat(cifar_scores)
    cifar_score_full  = torch.cat(cifar_scores_full)
    cifar_score_local = torch.cat(cifar_scores_local)

    svhn_scores_full  = []
    svhn_scores_local = []
    svhn_scores = []
    for x, y in tqdm(svhn_loader):
        x = x.to(device)
        local_log_probs = discretized_mix_logistic_prob(x, local_model(x))
        full_log_probs  = discretized_mix_logistic_prob(x, full_model(x))
        local_log_prob  = torch.sum(torch.sum(local_log_probs, dim=2), dim=1)
        full_log_prob   = torch.sum(torch.sum(full_log_probs,  dim=2), dim=1)
        score = full_log_prob - local_log_prob
        svhn_scores.append(score)
        svhn_scores_full.append(full_log_prob)
        svhn_scores_local.append(local_log_prob)
    svhn_score = torch.cat(svhn_scores)
    svhn_score_full  = torch.cat(svhn_scores_full)
    svhn_score_local = torch.cat(svhn_scores_local)

    print("AUROC:")
    labels = torch.cat((torch.ones(10000), torch.zeros(10000))).numpy()
    scores = torch.cat((cifar_score, svhn_score)).cpu().detach().numpy()
    print(labels.shape)
    print(scores.shape)
    print(roc_auc_score(labels, scores))

    plt.hist(cifar_score.cpu().detach().numpy(), bins=200)
    plt.hist(svhn_score.cpu().detach().numpy(), bins=200)
    plt.savefig("hist.png")

    full_mean_likelihood = cifar_score_full.cpu().detach().numpy().mean()
    full_mean_likelihood_svhn = svhn_score_full.cpu().detach().numpy().mean()
    full_mean_bpd = -full_mean_likelihood * np.log2(np.e) / (32 * 32 * 3)
    full_mean_bpd_svhn = -full_mean_likelihood_svhn * np.log2(np.e) / (32 * 32 * 3)

    local_mean_likelihood = cifar_score_local.cpu().detach().numpy().mean()
    local_mean_likelihood_svhn = svhn_score_local.cpu().detach().numpy().mean()
    local_mean_bpd = -local_mean_likelihood * np.log2(np.e) / (32 * 32 * 3)
    local_mean_bpd_svhn = -local_mean_likelihood_svhn * np.log2(np.e) / (32 * 32 * 3)

    print(f"full mean bpd: {full_mean_bpd}")
    print(f"full mean bpd on svhn: {full_mean_bpd_svhn}")

    print(f"local mean bpd: {local_mean_bpd}")
    print(f"local mean bpd on svhn: {local_mean_bpd_svhn}")
