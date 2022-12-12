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
parser.add_argument("-m", "--model-path", type=str, default='./models/pcnn_lr.0.00040_nr-resnet5_nr-filters160_319.pth', help="pre-trained model path")
args = parser.parse_args()
print(args)

model_path  = args.model_path

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToTensor(),
    rescaling])

blur_transform = transforms.Compose([transforms.GaussianBlur(7, sigma=(0.1, 2.0)), transforms.ToTensor(), rescaling])

cifar_val = torchvision.datasets.CIFAR10('./Data',
                                          train=False, 
                                          download=True, 
                                          transform=transform)
cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

blur_cifar_val = torchvision.datasets.CIFAR10('./Data',
                                          train=False, 
                                          download=True, 
                                          transform=blur_transform)
blur_cifar_loader = data.DataLoader(blur_cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

model = PixelCNN(nr_resnet=5, nr_filters=160,
            input_channels=3, nr_logistic_mix=10).to(device)

# model.load_state_dict(torch.load(model_path), strict=False)

state_dict = torch.load(model_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

with torch.no_grad():
    cifar_scores = []
    for x, y in tqdm(cifar_loader):
        x = x.to(device)
        cifar_scores.append(torch.sum(torch.sum(discretized_mix_logistic_prob(x, model(x)), dim=2), dim=1))
    cifar_score = torch.cat(cifar_scores)

    blur_cifar_scores = []
    for x, y in tqdm(blur_cifar_loader):
        x = x.to(device)
        blur_cifar_scores.append(torch.sum(torch.sum(discretized_mix_logistic_prob(x, model(x)), dim=2), dim=1))
    blur_cifar_score = torch.cat(blur_cifar_scores)

    print("AUROC:")
    labels = torch.cat((torch.ones(10000), torch.zeros(10000))).numpy()
    scores = torch.cat((cifar_score, blur_cifar_score)).cpu().detach().numpy()
    print(labels.shape)
    print(scores.shape)
    print(roc_auc_score(labels, scores))

    plt.hist(cifar_score.cpu().detach().numpy(), bins=200)
    plt.hist(blur_cifar_score.cpu().detach().numpy(), bins=200)
    plt.savefig("hist_blur.png")

    mean_likelihood = cifar_score.cpu().detach().numpy().mean()
    mean_likelihood_blur_cifar = blur_cifar_score.cpu().detach().numpy().mean()
    mean_bpd = -mean_likelihood * np.log2(np.e) / (32 * 32 * 3)
    mean_bpd_blur_cifar = -mean_likelihood_blur_cifar * np.log2(np.e) / (32 * 32 * 3)
    print(f"mean bpd: {mean_bpd}")
    print(f"mean bpd on blur cifar: {mean_bpd_blur_cifar}")
