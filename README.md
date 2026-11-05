# CIFAR-100 Out-of-Distribution Challenge

My solution to a [Kaggle class challenge](https://www.kaggle.com/competitions/cs-541-challenge-spring-2026/leaderboard) (Spring 2026): train image classifiers on CIFAR-100, then predict labels for an out-of-distribution (OOD) test set of corrupted images. The test set covers 20 distortion types at 5 severity levels, 10,000 images each, for 1,000,000 predictions in total.

## Results

| Part | Model | Approach | Clean val accuracy | Kaggle OOD accuracy |
|---|---|---|---|---|
| 1 | Wide ResNet-28-10 (36.5M params) | Trained from scratch at 32×32 | 81.92% | 69.06% |
| 2 | ConvNeXt-Tiny (27.9M params) | ImageNet-pretrained, fine-tuned at 224×224 | 88.40% | 78.75% |
| 3 | EVA-02-Large/14 (304.2M params) | Masked-image-modeling pretrained ViT, two-phase fine-tune at 448×448 | 95.42% | **88.28%** |

The submitted predictions come from Part 3. Clean validation accuracy is measured on a held-out 10% of the CIFAR-100 training set.

## Approach

**Part 1: Wide ResNet-28-10 from scratch.** Random crops, horizontal flips, MixUp (alpha 0.2) and label smoothing (0.1), trained for 100 epochs with SGD (Nesterov momentum) and cosine learning-rate decay.

**Part 2: ConvNeXt-Tiny, fine-tuned.** ImageNet weights from `timm`. A small adapter takes the challenge's CIFAR-normalized 32×32 inputs, upsamples them to 224×224 and re-normalizes them for ImageNet, so the required submission pipeline works unchanged. AdamW, label smoothing and cosine decay over 30 epochs.

**Part 3: EVA-02-Large at 448×448** (`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`), fine-tuned in two phases:

- Phase 1: 3 epochs training only the classifier head (learning rate 1e-3).
- Phase 2: 12 epochs of full fine-tuning (backbone 1e-5, head 5e-5) with a 2-epoch warmup and cosine decay, MixUp and CutMix, gradient clipping and mixed precision.
- Robustness augmentation: CIFAR-C-style corruptions (noise, blur, fog, brightness, contrast, JPEG compression, pixelation, saturation, grayscale, spatter) applied with probability 0.5 at severities 1 to 3, plus AugMix and random erasing.

## Files

- `cifar100_ood.ipynb`: the notebook with all three parts and its saved training output
- `cifar100_ood.py`: the same code as a Python script

## Running it

Set `MODE` in the configuration cell, then run all cells:

- `demo`: builds the architectures and runs one-batch sanity checks, with no training
- `inference`: downloads trained checkpoints from `CHECKPOINTS_URL` and writes the submission CSV
- `full`: trains all three parts end to end and writes `submission_ood.csv`

Dependencies: `torch`, `torchvision`, `timm`, `huggingface_hub`, `numpy`, `pandas`, `matplotlib`, `tqdm` and `pillow` (the notebook installs any that are missing). The OOD test set downloads automatically from Hugging Face. Full training needs a large GPU; these results were produced on an NVIDIA H200 NVL.
