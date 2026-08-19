# Author: Yao
# CreatTime: 2025/10/18
# FileName: filter_thr_p
# Description: simple introduction of the code

"""
Filter degenerate boxes (w/h below a threshold) from proposals.
author: your_name
"""

import pickle
import numpy as np
from typing import List, Tuple, Dict


def filter_proposals(
    proposals: List[np.ndarray],
    min_wh: float = 5,
    min_area: float = None,
    min_width: float = None,
    min_height: float = None,
) -> Tuple[List[np.ndarray], Dict]:
    """Filter proposals by minimum width, height, and optional area.

    ``min_wh`` is kept for backward compatibility and is used for both width
    and height unless ``min_width`` or ``min_height`` is explicitly supplied.
    A proposal is retained only when all geometry constraints are satisfied.
    """
    if min_width is None:
        min_width = min_wh
    if min_height is None:
        min_height = min_wh

    proposals_filtered = []
    invalid_images = []
    invalid_boxes_count = 0

    for img_idx, img_boxes in enumerate(proposals):
        boxes_array = np.asarray(img_boxes)
        if boxes_array.size == 0:
            proposals_filtered.append(boxes_array)
            continue
        if boxes_array.ndim != 2 or boxes_array.shape[1] < 4:
            raise ValueError(
                f'Expected proposal array [N, >=4], got shape '
                f'{boxes_array.shape} for image index {img_idx}'
            )

        boxes = boxes_array[:, :4].astype(np.float32, copy=False)
        wh = boxes[:, 2:4] - boxes[:, :2]
        areas = wh[:, 0] * wh[:, 1]
        valid_mask = (
            (wh[:, 0] >= float(min_width))
            & (wh[:, 1] >= float(min_height))
        )
        if min_area is not None:
            valid_mask &= areas >= float(min_area)

        invalid_count = int((~valid_mask).sum())
        if invalid_count:
            invalid_images.append(img_idx)
            invalid_boxes_count += invalid_count

        proposals_filtered.append(boxes_array[valid_mask])

    stats = {
        "original": sum(len(x) for x in proposals),
        "filtered": sum(len(x) for x in proposals_filtered),
        "deleted": invalid_boxes_count,
        "invalid_images": len(invalid_images),
    }
    return proposals_filtered, stats



def filter_proposals_file(src_pkl: str,
                          dst_pkl: str,
                          min_wh: float = 0.5) -> None:
    """
    Filter a pkl file and save the result to a new file.
    """
    with open(src_pkl, "rb") as f:
        proposals = pickle.load(f)

    proposals_filtered, stats = filter_proposals(proposals, min_wh)

    with open(dst_pkl, "wb") as f:
        pickle.dump(proposals_filtered, f)

    print("Processing summary:")
    print(f"Total original proposals: {stats['original']}")
    print(f"Total filtered proposals: {stats['filtered']}")
    print(f"Total invalid boxes removed: {stats['deleted']}")
    print(f"Images with invalid boxes: {stats['invalid_images']}")
    print(f"New file saved to: {dst_pkl}")
