ShuffleNetV2 Person Re-Identification Strong Baseline
=====================================================

Files
-----
config.py   Shared settings
data.py     Market-1501 datasets, augmentation and P x K batch sampler
model.py    ShuffleNetV2 x1.0, last-stride-one adaptation and BNNeck
losses.py   Batch-hard triplet loss and center loss
metrics.py  Market-1501 Rank-1/5/10/20, mAP, parameter and FLOP estimates
train.py    Training, checkpoint resume, evaluation and CSV logging

Important
---------
This implementation includes the six main training tricks from
"A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification":
1. Warmup and step-decay learning rate
2. Random erasing augmentation
3. Label smoothing
4. Last-stride-one adaptation for the ShuffleNetV2 final downsampling block
5. BNNeck
6. Center loss

It also adds batch-hard triplet loss as requested.

The code is configured for 100 epochs and 128x64 images. It uses P=4 identities
and K=4 images per identity per batch (batch size 16), which is important for
batch-hard triplet mining.

The previous plain ShuffleNetV2 checkpoint is NOT compatible with the changed
architecture/training setup. This code uses a separate checkpoint:
checkpoints/shufflenetv2_reid_strong_baseline.pth
It will start from scratch the first time, then resume this new checkpoint
on subsequent runs. It does not overwrite checkpoints/shufflenetv2_market1501.pth.

Run from the project directory containing Market-1501:
    python train.py

Expected dataset folders:
    Market-1501/bounding_box_train
    Market-1501/query
    Market-1501/bounding_box_test

Results are appended to:
    outputs/reid_evaluation_results.csv

Open that CSV in Excel. It contains retrieval metrics, training loss components,
parameter counts, MACs, FLOPs, learning rate, and training time.

Parameter/FLOP reporting
------------------------
FLOPs are estimated for a single 1x3x128x64 image and use the convention
1 multiply-accumulate (MAC) = 2 FLOPs. The estimate counts convolution and
linear operations; it does not count every BN, activation, pooling, or memory
operation, so treat it as an approximate model-compute figure.

The published 94.5% Rank-1 and 85.9% mAP from the paper were obtained with
ResNet-50 and its experimental setup. They are not expected results or a
guarantee for this ShuffleNetV2 adaptation.

Requirements
------------
Install compatible PyTorch and torchvision versions for your Python environment.
No third-party FLOP package is needed.
