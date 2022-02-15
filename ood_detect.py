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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path  = './models/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth'
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

    svhn_scores = []
    for x, y in tqdm(svhn_loader):
        x = x.to(device)
        svhn_scores.append(torch.sum(torch.sum(discretized_mix_logistic_prob(x, model(x)), dim=2), dim=1))
    svhn_score = torch.cat(svhn_scores)

    print("AUROC:")
    labels = torch.cat((torch.ones(10000), torch.zeros(10000))).numpy()
    scores = torch.cat((cifar_score, svhn_score)).cpu().detach().numpy()
    print(labels.shape)
    print(scores.shape)
    print(roc_auc_score(labels, scores))

    plt.hist(cifar_score.cpu().detach().numpy(), bins=200)
    plt.hist(svhn_score.cpu().detach().numpy(), bins=200)
    plt.savefig("hist.png")

    mean_likelihood = cifar_score.cpu().detach().numpy().mean()
    mean_bpd = -mean_likelihood * np.log2(np.e) / (32 * 32 * 3)
    print(f"mean bpd: {mean_bpd}")
