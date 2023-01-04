# Refer to https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

import os
import torch
import argparse
import torchvision
from collections import OrderedDict

from model import PixelCNN
from utils import discretized_mix_logistic_loss_1d, discretized_mix_logistic_loss, sample_from_discretized_mix_logistic_1d, sample_from_discretized_mix_logistic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', type=int,
                        default=2048, help='number of steps of langevin dynamics')
    parser.add_argument('-ss', '--step-size', type=int,
                        default=10, help='Learning rate nu in langevin dynamics')
    parser.add_argument('-d', '--dataset', type=str,
                        default='mnist', help='Can be either cifar|mnist')
    parser.add_argument("-r", "--row", type=int, default=10, help="number of rows in sampled image")
    parser.add_argument("-c", "--col", type=int, default=10, help="number of cols in sampled image")
    parser.add_argument("-m", "--model-path", type=str, default='./models/pcnn_lr_0.00040_nr-resnet5_nr-filters160_249.pth', help="pre-trained model path")
    parser.add_argument("-p", "--pre-trained", default=False, action='store_true')
    parser.add_argument('-nlm', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    args = parser.parse_args()
    print(args)

    if 'mnist' in args.dataset : 
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

    elif 'cifar' in args.dataset : 
        loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
    else :
        raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

    rescaling_inv = lambda x : .5 * x  + .5

    obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
    if not os.path.exists("./mcmc_sample"):
        os.makedirs("./mcmc_sample")

    model_path  = args.model_path
    device = torch.device("cuda:0")
    model = PixelCNN(nr_resnet=5, nr_filters=160,
                    input_channels=obs[0], nr_logistic_mix=args.nr_logistic_mix).to(device)
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
    for p in model.parameters():
        p.requires_grad = False

    inp_imgs = torch.rand((args.row * args.col,) + obs) * 2 - 1 # [-1, 1)
    inp_imgs = inp_imgs.to(device)
    inp_imgs.requires_grad = True

    torch.set_grad_enabled(True)

    noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
    # List for storing generations at each step (for later analysis)
    imgs_per_step = []

    # Loop over K (steps)
    for i in range(args.steps):
        # Part 1: Add noise to the input.
        noise.normal_(0, 0.005)
        inp_imgs.data.add_(noise.data)
        inp_imgs.data.clamp_(min=-1.0, max=1.0)

        # Part 2: calculate gradients for the current input.
        out_imgs = model(inp_imgs)
        loss = 3e-3 * (loss_op(inp_imgs, out_imgs))
        loss.backward()
        # inp_imgs.grad.mul_(0.0001)
        inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients
        print(inp_imgs.grad.data)

        # Apply gradients to our current samples
        inp_imgs.data.add_(-args.step_size * inp_imgs.grad.data)
        inp_imgs.grad.detach_()
        inp_imgs.grad.zero_()
        inp_imgs.data.clamp_(min=-1.0, max=1.0)

        imgs_per_step.append(inp_imgs.clone().detach())
        torchvision.utils.save_image(inp_imgs.clone().detach(), f'mcmc_sample/{i}.png', nrow=args.row, padding=0)


