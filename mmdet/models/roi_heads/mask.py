# Author: Yao
# CreatTime: 2024/12/31
# FileName: mask
# Description: simple introduction of the code

import random
import math
import numpy as np
from PIL import Image
import torch

# Image size, patch size, aspect-ratio settings
log_aspect_ratio = tuple(map(lambda x: math.log(x), (0.3, 1 / 0.3)))
pred_shape = 'block'


def get_pred_ratio(pred_ratio=None, pred_ratio_var=None):
    """Return predicted mask ratios under different conditions."""
    if isinstance(pred_ratio, list):
        pred_ratio_final = []
        for prm, prv in zip(pred_ratio, pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio_final.append(pr)
        pred_ratio_final = random.choice(pred_ratio_final)
    else:
        assert pred_ratio >= pred_ratio_var
        pred_ratio_final = random.uniform(pred_ratio - pred_ratio_var,
                                          pred_ratio + pred_ratio_var) if pred_ratio_var > 0 else pred_ratio

    return pred_ratio_final


def apply_mask_to_bboxes(features, pred_ratio=[0.0, 0.3], pred_ratio_var=[0.0, 0.2]):
    """
    Mask a feature tensor of shape [B, H, W, C], where each [7, 7] block
    is a bbox's feature map (e.g. region features of shape [bboxes, 7, 7, 512]).

    Args:
    - features (np.ndarray or torch.Tensor): of shape [B, 7, 7, 512].
    - pred_ratio (tuple): mask ratio range (min, max).
    - pred_ratio_var (tuple): mask ratio variance range (min, max).

    Returns:
    - masked_features (np.ndarray or torch.Tensor): masked features.
    - masks (list): mask per bbox.
    """
    B, H, W, C = features.shape
    masked_features = features
    masks = []

    for b in range(B):
        # independent mask per bbox
        pred_ratio_final = get_pred_ratio(pred_ratio, pred_ratio_var)
        mask_count_target = int(pred_ratio_final * H * W)
        mask = np.zeros((H, W), dtype=bool)
        mask_count = 0

        while mask_count < mask_count_target:
            max_mask_patches = mask_count_target - mask_count
            delta = 0
            for attempt in range(10):
                low = (min(H, W) // 3) ** 2  # low = 1
                target_area = random.uniform(low, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < W and h < H:
                    top = random.randint(0, H - h)
                    left = random.randint(0, W - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1
                if delta > 0:
                    break
            if delta == 0:
                break
            else:
                mask_count += delta

        masks.append(mask)

        for i in range(H):
            for j in range(W):
                if mask[i, j]:
                    masked_features[b, i, j, :] = 0

    return masked_features, masks




