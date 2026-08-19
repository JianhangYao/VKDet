# Author: Yao
# CreatTime: 2025/10/17
# FileName: base_filter
# Description: Following RandBox(CVPR2023)'s anchor position conditions
import numpy as np
import random
import torch
import pickle
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from mmdet.models.roi_heads.Adaptive_kd_weight import up_attn, attn_align, augment_box
from mmdet.our_utils.hungarian_matcher import HungarianMatcherDynamicK
from mmdet.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou, generalized_box_iou
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def filter_and_match_annotations(clip_whole_image_attns, i, filtered_lst, data_root):
    """Filter and match annotations for a single image."""

    xml_filename = f"{clip_whole_image_attns['img_names'][i]}.xml"

    xml_path = os.path.join(data_root, f"Annotations/Filtered_Annotations/{xml_filename}")

    if not os.path.exists(xml_path):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        gt_bboxes = []

        for obj in root.findall("object"):
            # class & difficulty
            name = obj.find("name").text
            difficult_elem = obj.find("difficult")
            difficult = 0 if difficult_elem is None else int(difficult_elem.text)

            # skip difficult objects
            if difficult == 1:
                continue

            bnd_box = obj.find("bndbox")
            bbox = [
                float(bnd_box.find("xmin").text),
                float(bnd_box.find("ymin").text),
                float(bnd_box.find("xmax").text),
                float(bnd_box.find("ymax").text)
            ]
            gt_bboxes.append(bbox)

        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)

    except ET.ParseError as e:
        print(f"XML parse error: {xml_path}, {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing annotation: {xml_path}, {str(e)}")
        return None

    # skip if no valid objects
    if len(gt_bboxes) == 0:
        return None

    # candidate proposals
    filtered_lst_tensors = []
    for t in filtered_lst:
        filtered_lst_tensor = t.clone().detach()
        filtered_lst_tensors.append(filtered_lst_tensor)
    candidate_boxes = torch.stack(filtered_lst_tensors, dim=0)[:, :4]  # [M, 4]

    matcher = HungarianMatcherDynamicK()

    # in-box / in-center flags
    is_in_boxes_anchor, is_in_boxes_and_center = matcher.get_in_boxes_info(
        box_xyxy_to_cxcywh(candidate_boxes),
        box_xyxy_to_cxcywh(gt_bboxes),
        expanded_strides=32
    )
    """
    is_in_boxes_anchor: bool tensor [M] 
        - True: anchor inside the gt box or its center region
        - False: anchor in the background

    is_in_boxes_and_center: bool tensor [M, K, 2]
        - dim 1: anchor index
        - dim 2: gt index
        - dim 3: [in_box, in_center]
    """

    # pairwise IoU matrix
    pair_wise_ious = ops.box_iou(candidate_boxes, gt_bboxes)  # [M, K]

    iou_threshold = 0.5 # (pair_wise_ious > iou_threshold)
    instance_index = is_in_boxes_anchor.unsqueeze(1) | is_in_boxes_and_center


    return {
        # 'gt_bboxes': gt_bboxes,
        # 'is_in_boxes_anchor': is_in_boxes_anchor,
        # 'is_in_boxes_and_center': is_in_boxes_and_center,
        # 'pair_wise_ious': pair_wise_ious,
        'gt_instance_index': instance_index
    }
