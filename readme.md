## PixelCNN++

A Pytorch Implementation of [PixelCNN++.](https://arxiv.org/pdf/1701.05517.pdf)

Main work taken from the [official implementation](https://github.com/openai/pixel-cnn)

Pre-trained models are available [here](https://mega.nz/#F!W7IhST7R!PV7Pbet8Q07GxVLGnmQrZg)

I kept the code structure to facilitate comparison with the official code. 

The code achieves **2.95** BPD on test set, compared to **2.92** BPD on the official tensorflow implementation. 
<p align="center">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_143.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_122.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_137.png">
<img src="https://github.com/pclucas14/pixel-cnn-pp/blob/master/images/pcnn_lr:0.00020_nr-resnet5_nr-filters160_101.png">
</p>

### Training the model
```
python main.py
```

### Training the model by blurred training set
```
python blur_train.py
```

### Explanation of the scirpts

`ood_detect.py`: train on CIFAR10, test on CIFAR10 and SVHN. Plot the histgram and output mean BPDs.

`blur_detect.py`: train on CIFAR10, test on CIFAR10 and blurred CIFAR10, the generated histogram shows that blurred CIFAR10 has higher dentisty then original CIFAR10.

### Differences with official implementation
1. No data dependant weight initialization 
2. No exponential moving average of past models for test set evalutation

### Contact
This repository is no longer maintained. Feel free to file an issue if need be, however response may be slow. 

### Out-of-distribution detection
| Epoch | BPD    | SVHN\_BPD | AUROC  |
|-------|--------|----------|--------|
| 319   | 2.9418 | 2.0881   | 0.1493 |
| 489   | 2.9409 | 2.0940   | 0.1518 |
| 589   | 2.9361 | 2.0942   | 0.1535 |
| 789   | 2.9295 | 2.0969   | 0.1570 |
| 889   | 2.9292 | 2.0912   | 0.1558 |

