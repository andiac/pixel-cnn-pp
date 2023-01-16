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
parser.add_argument("-m", "--model-path", type=str, default='./models/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth', help="pre-trained model path")
parser.add_argument("-p", "--pre-trained", default=False, action='store_true')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|fashion')
args = parser.parse_args()
print(args)

model_path  = args.model_path

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToTensor(),
    rescaling])

if "cifar" in args.dataset:
    cifar_val = torchvision.datasets.CIFAR10('./data',
                                              train=False, 
                                              download=True, 
                                              transform=transform)
    cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    svhn_val = torchvision.datasets.SVHN('./data',
                                         split='test',
                                         download=True,
                                         transform=transform)
    svhn_val.data = svhn_val.data[:10000]
    svhn_loader = data.DataLoader(svhn_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    model = PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=3, nr_logistic_mix=10).to(device)

if "fashion" in args.dataset:
    cifar_val = torchvision.datasets.FashionMNIST('./data',
                                              train=False, 
                                              download=True, 
                                              transform=transform)
    cifar_loader = data.DataLoader(cifar_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    svhn_val = torchvision.datasets.MNIST('./data',
                                         split='test',
                                         download=True,
                                         transform=transform)
    svhn_val.data = svhn_val.data[:10000]
    svhn_loader = data.DataLoader(svhn_val, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    # TODO...
    model = PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=3, nr_logistic_mix=10).to(device)


# model.load_state_dict(torch.load(model_path), strict=False)

if args.pre_trained:
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(torch.load(model_path), strict=False)
    
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
    mean_likelihood_svhn = svhn_score.cpu().detach().numpy().mean()
    mean_bpd = -mean_likelihood * np.log2(np.e) / (32 * 32 * 3)
    mean_bpd_svhn = -mean_likelihood_svhn * np.log2(np.e) / (32 * 32 * 3)
    print(f"mean bpd: {mean_bpd}")
    print(f"mean bpd on svhn: {mean_bpd_svhn}")

    cifar_score_np = cifar_score.cpu().detach().numpy()  * np.log2(np.e) / (32 * 32 * 3)
    svhn_score_np  = svhn_score.cpu().detach().numpy() * np.log2(np.e) / (32 * 32 * 3)
    hist_range_min = np.concatenate((cifar_score_np, svhn_score_np)).min()
    hist_range_max = np.concatenate((cifar_score_np, svhn_score_np)).max()
    cifary, cifarx = np.histogram(cifar_score_np, bins=22, range=(hist_range_min, hist_range_max))
    svhny, svhnx   = np.histogram(svhn_score_np, bins=22, range=(hist_range_min, hist_range_max))
    print("cifar hist:")
    for y, x in zip(cifary, cifarx):
        print(f"({x}, {y})")
    print(f"({cifarx[-1]}, 0)")

    print("svhn hist:")
    for y, x in zip(svhny, svhnx):
        print(f"({x}, {y})")
    print(f"({svhnx[-1]}, 0)")


