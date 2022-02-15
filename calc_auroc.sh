#!/bin/bash

for epoch in 319 489 589 789 889
do
    python ood_detect.py -m ./models/pcnn_lr.0.00040_nr-resnet5_nr-filters160_$epoch.pth
done
