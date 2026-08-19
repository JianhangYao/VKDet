<h1 align="center">VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection</h1>

<div align="center">
<p>
  <a href="https://arxiv.org/abs/2511.18075"><img src="https://img.shields.io/badge/Paper-arxiv%3A2511.18075-blue" alt="Paper"/></a>
  <a href="https://huggingface.co/papers/2511.18075"><img src="https://img.shields.io/badge/Daily%20Paper-huggingface-yellow" alt="HF Paper"/></a>
  <a href="https://github.com/ChenDelong1999/RemoteCLIP"><img src="https://img.shields.io/badge/VLM-RemoteCLIP-green" alt="RemoteCLIP"/></a>
</p>
</div>

VK-Det is an open-vocabulary aerial object detector built around three components:

| Component | Full name | Role |
|---|---|---|
| ASKD | Adaptive Selective Knowledge Distillation | Selects informative image regions and distills VLM knowledge |
| PAPL | Prototype-Aware Pseudo-Labeling | Clusters region embeddings and generates novel-class pseudo-labels |
| SMI | Synthetic Matching Inference | Fuses distillation, prototype, and localization scores |

## Installation

The tested H800 environment uses Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, and MMCV-Full 1.7.2.

```bash
virtualenv -p python3.10 vkdet
source vkdet/bin/activate

pip install torch==2.1.2 torchvision==0.16.2

# Set CUDA_HOME to the toolkit installed on your machine.
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" \
  pip install --no-build-isolation mmcv-full==1.7.2

pip install -r requirements.txt
pip install -e .
```

`requirements_h800.txt` records the fully pinned reference environment. It is useful for comparison, but a fresh installation should follow the shorter commands above because CUDA wheels and compiler toolchains vary between machines.

## Required directory layout

Large assets are intentionally excluded from Git.

```text
vk-det/
├── configs/dior/
├── data/DIOR/
│   ├── JPEGImages-trainval/
│   ├── Annotations/
│   │   ├── Horizontal Bounding Boxes/
│   │   └── Filtered_Annotations/
│   └── ImageSets/Main/
│       ├── filtered_train.txt
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── embeddings/dior/
│   └── ovd_dior_text_embedding.pth
├── proposals/dior/
│   ├── invalid_box_filter/
│   │   └── filtered_train_proposals.pkl
│   ├── whole_train/
│   │   └── whole_train_proposals.pkl
│   └── val/
│       └── train_filtered_val_proposals.pkl
├── weights/
│   ├── current_mmdetection_Head.pth
│   ├── RemoteCLIP-ViT-B-32.pt
│   └── ViT-B-32.pt
├── askd_cache/                 # generated; ignored by Git
└── workdirs/                   # generated; ignored by Git
```

### Pretrained weights

| File | Source | Purpose |
|---|---|---|
| `current_mmdetection_Head.pth` | [LP-OVOD](https://github.com/VinAIResearch/LP-OVOD) | SoCo-pretrained detector |
| `RemoteCLIP-ViT-B-32.pt` | [RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) | VLM used by ASKD/PAPL |
| `ViT-B-32.pt` | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | Offline OpenAI CLIP initialization |

OLN proposals are not redistributed by this repository. Generate them with [OLN](https://github.com/mcahny/object_localization_network), preserve the split-file ordering, and place them at the paths shown above. Each proposal file must be a `list[ndarray]`; each array has shape `[N, 5]` and stores `(x1, y1, x2, y2, score)`.

## DIOR preparation

Place the official DIOR images, annotations, and split files under `data/DIOR`, then run:

```bash
python dataset/prepare_dior.py --dior_root data/DIOR
```

This creates `filtered_train.txt` and `Annotations/Filtered_Annotations`, removing novel-class annotations from Stage 1 supervision. Use `--force` to rebuild the filtered XML directory safely.

Generate DIOR text embeddings with:

```bash
python text_prepare/dior_utils.py
```

## Training

The scripts use `torch.distributed.run` and default to one visible GPU. For multiple GPUs, set both `CUDA_VISIBLE_DEVICES` and `GPUS`.

### Stage 1: ASKD pretraining

```bash
bash scripts/vkdet_train_stage1.sh
```

Stage 1 automatically runs ASKD or reuses matching files under `askd_cache/`, then trains for 20 epochs. Its default output is `workdirs/dior/stage1`.

### Stage 2: PAPL fine-tuning

```bash
bash scripts/vkdet_train_stage2.sh
```

Stage 2 loads `workdirs/dior/stage1/epoch_20.pth`, runs ASKD/PAPL preprocessing, persists cluster centers and pseudo-label JSON, and trains for 12 epochs. Its default output is `workdirs/dior/stage2`.

Multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash scripts/vkdet_train_stage1.sh
```

## Evaluation

```bash
bash scripts/vkdet_test_stage2.sh workdirs/dior/stage2/epoch_12.pth
```

Evaluation uses `configs/dior/vkdet_dior_stage2_test.py` and loads the persisted cluster centers from `askd_cache/papl_dior_csd500/centers/30_cluster_centers.pth`. It does not rerun ASKD/PAPL.

## Citation

```bibtex
@inproceedings{yao2026vkdet,
  title     = {VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection},
  author    = {Yao, Jianhang and Zheng, Yongbin and Lu, Siqi and Xu, Wanying and Sun, Peng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  eprint    = {2511.18075},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2511.18075}
}
```

## Acknowledgements

This project builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [LP-OVOD](https://github.com/VinAIResearch/LP-OVOD), [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild), [CastDet](https://github.com/Li-Qingyun/CastDet), [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP), and [OLN](https://github.com/mcahny/object_localization_network). Please follow the licenses and dataset terms of the upstream projects when redistributing code, data, proposals, or weights.
