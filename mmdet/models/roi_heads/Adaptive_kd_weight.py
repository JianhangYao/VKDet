# Author: Yao
# CreatTime: 2025/10/13
# FileName: Adaptive_kd_weight
# Description: simple introduction of the code

import torch
import pickle
import xml.etree.ElementTree as ET
import os
import numpy as np
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def up_attn(attn_img, shapen_temp):
    # H, W = img_metas['img_shape'][0], img_metas['img_shape'][1]
    attn = attn_img  / shapen_temp  # / temp_s
    h, w = attn.shape
    attn_flat = attn.view(-1) # attn
    def norm_attn(mask_flat):
        mu = mask_flat.mean()
        delta = 1 - mu
        min_val = mask_flat.min()
        delta_star = max(delta, 0)
        A_star = mask_flat + delta_star
        return A_star

    mask_flat = torch.sigmoid(attn_flat.to(device))
    norm_mask = norm_attn(mask_flat)

    mask = norm_mask.view(1, h, w) # norm_mask.view(h, w).cpu().numpy()
    return mask

def attn_align(data_root, bboxes, clip_whole_image_attns, img_name):
    """
    Compute per-bbox attention-weighted features from the whole-image attention maps.
    :param bboxes: list[array] [N,5], (x1, y1, x2, y2, score) in image coords
    :param clip_whole_image_attns: List[Tensor], each of shape [batch, H, W]
    :param img_metas: List[dict], image meta info (for sizes)
    :return:
       weights: Tensor [N], attention weight per bbox
       region_features: Tensor [N, D], region features (optional)
    """
    # ========== 1. normalize coordinates ==========
    # if DOTA:
    # if DIOR:
    xml_path = os.path.join(
        data_root,
        'Horizontal Bounding Boxes',
        f"{img_name}.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)


    weights = torch.zeros(len(bboxes), device=device)


    # attention map for this sample [H, W]
    attn_map = clip_whole_image_attns.squeeze()
    # bboxes of this sample [K,5] -> [K,4]

    batch_boxes = torch.tensor([bbox[:4] for bbox in bboxes], device=device)



    # ========== 2. coordinate transform ==========
    # scale factors (assumes attn map is image-sized)
    # adjust if not, e.g. attn_h = img_h // stride
    scale_factor = torch.tensor([attn_map.shape[1] / img_w,
                                 attn_map.shape[0] / img_h,
                                 attn_map.shape[1] / img_w,
                                 attn_map.shape[0] / img_h], device=device)

    # to attention-map coords [K,4]
    feat_boxes = batch_boxes * scale_factor

    # integer coords (use RoI Align for precise alignment)
    x1 = torch.floor(feat_boxes[:, 0]).long().clamp(min=0, max=attn_map.shape[1] - 1)
    y1 = torch.floor(feat_boxes[:, 1]).long().clamp(min=0, max=attn_map.shape[0] - 1)
    x2 = torch.ceil(feat_boxes[:, 2]).long().clamp(min=0, max=attn_map.shape[1])
    y2 = torch.ceil(feat_boxes[:, 3]).long().clamp(min=0, max=attn_map.shape[0])

    # ========== 3. region feature extraction ==========
    # avg pooling (simple & fast)
    region_attns = []
    num_regions = int(0)
    for i in range(batch_boxes.shape[0]):
        attn_roi = attn_map[y1[i]:y2[i], x1[i]:x2[i]] # attn_roi.cpu().numpy()
        region_mean = attn_roi.mean() if attn_roi.numel() > 0 else 0.0
        # w = batch_boxes[i][2] - batch_boxes[i][0]
        # h = batch_boxes[i][3] - batch_boxes[i][1]
        if region_mean >= 1:
            region = True
            # if area < float(32 * 32):
            num_regions = num_regions + 1
        else:
            region = False

        region_attns.append(region)

    weights = region_attns # scaled.cpu().numpy()

    return weights


def augment_box(box, augment_type, image_size=800,
                box_jitter_var=0.1, reg_jitter_var=0.05, alpha = 1):
    """
    Augment a remote-sensing proposal box with one of four strategies:
    1. translate  2. scale  3. box jitter  4. regression jitter

    Args:
        box: [x1, y1, x2, y2, score] box coords and confidence
        image_size: image size (square)
        box_jitter_var: box-jitter variance (jitter strength)
        reg_jitter_var: regression-jitter variance (perturbation strength)

    Returns:
        augmented box [new_x1, new_y1, new_x2, new_y2, score]
    """
    x1, y1, x2, y2, score = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # random strategy (0: short-side jitter, 1: long-side jitter)

    # if augment_type == 0:
    #     delta_x1 = random.uniform(-0.1, 0.1) * w
    #     delta_y1 = random.uniform(-0.1, 0.1) * h
    #     delta_x2 = random.uniform(-0.1, 0.1) * w
    #     delta_y2 = random.uniform(-0.1, 0.1) * h
    #
    #     new_x1 = np.clip(x1 + delta_x1, 0, image_size - 1)
    #     new_y1 = np.clip(y1 + delta_y1, 0, image_size - 1)
    #     new_x2 = np.clip(x2 + delta_x2, new_x1 + 1, image_size)
    #     new_y2 = np.clip(y2 + delta_y2, new_y1 + 1, image_size)
    #
    # elif augment_type == 1:
    #     scale_factor = random.uniform(0.7, 1.3)
    #     new_w = w * scale_factor
    #     new_h = h * scale_factor
    #
    #     new_x1 = cx - new_w / 2
    #     new_y1 = cy - new_h / 2
    #     new_x2 = cx + new_w / 2
    #     new_y2 = cy + new_h / 2
    #
    #     new_x1 = np.clip(new_x1, 0, image_size - 1)
    #     new_y1 = np.clip(new_y1, 0, image_size - 1)
    #     new_x2 = np.clip(new_x2, new_x1 + 1, image_size)
    #     new_y2 = np.clip(new_y2, new_y1 + 1, image_size)

    # 3. short-side jitter (scale-invariant)
    if augment_type == 0:
        min_edge = min(w, h)
        short_jitter_x = box_jitter_var * w * random.gauss(0, 1)
        short_jitter_y = box_jitter_var * h * random.gauss(0, 1)

        new_x1 = np.clip(cx - min_edge/2 + short_jitter_x , 0, image_size - 1)
        new_y1 = np.clip(cy - min_edge/2 + short_jitter_y, 0, image_size - 1)
        new_x2 = np.clip(cx + min_edge/2 + short_jitter_x, new_x1 + 1, image_size)
        new_y2 = np.clip(cy + min_edge/2 + short_jitter_y, new_y1 + 1, image_size)

    # 4. long-side jitter
    else:
        # regression perturbation (center shift + scale)
        max_edge = max(w, h)
        long_jitter_x = reg_jitter_var * w * random.gauss(0, 1)
        long_jitter_y = reg_jitter_var * h * random.gauss(0, 1)

        # enforce min size
        scale_x = np.clip(long_jitter_x, -0.3, 0.3)
        scale_y = np.clip(long_jitter_y, -0.3, 0.3)

        new_x1 = np.clip(cx - max_edge/2 + scale_x , 0, image_size - 1)
        new_y1 = np.clip(cy - max_edge/2 + scale_y, 0, image_size - 1)
        new_x2 = np.clip(cx + max_edge/2 + scale_x, new_x1 + 1, image_size)
        new_y2 = np.clip(cy + max_edge/2 + scale_y, new_y1 + 1, image_size)

    return torch.tensor([new_x1, new_y1, new_x2, new_y2, score])