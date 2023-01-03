import os
import torch
import argparse
import torchvision
from collections import OrderedDict

from model import PixelCNN
from utils import sample_from_discretized_mix_logistic_1d

sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, 10)
rescaling_inv = lambda x : .5 * x  + .5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str,
                        default='mnist', help='Can be either cifar|mnist')
    parser.add_argument("-r", "--row", type=int, default=10, help="number of rows in sampled image")
    parser.add_argument("-c", "--col", type=int, default=10, help="number of cols in sampled image")
    parser.add_argument("-m", "--model-path", type=str, default='./models/pcnn_lr_0.00040_nr-resnet5_nr-filters160_249.pth', help="pre-trained model path")
    parser.add_argument("-p", "--pre-trained", default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
    if not os.path.exists("./sampled_images"):
        os.makedirs("./sampled_images")

    model_path  = args.model_path
    device = torch.device("cuda:0")
    model = PixelCNN(nr_resnet=5, nr_filters=160,
                input_channels=obs[0], nr_logistic_mix=10).to(device)
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
        data = torch.zeros(args.row * args.col, obs[0], obs[1], obs[2])
        data = data.cuda()
        for i in range(obs[1]):
            for j in range(obs[2]):
                # data_v = Variable(data, volatile=True)
                data_v = data
                out   = model(data_v, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]

    model_name = os.path.split(model_path)[1][:-4]
    data = rescaling_inv(data)
    torchvision.utils.save_image(data, f'sampled_images/{model_name}.png', nrow=args.row, padding=0)

