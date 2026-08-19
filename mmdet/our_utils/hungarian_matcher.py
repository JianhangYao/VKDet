# HungarianMatcherDynamicK — SimOTA-style dynamic-k matcher
# Adapted from:
#   - OTA: Optimal Transport Assignment for Object Detection (CVPR 2021)
#     https://arxiv.org/abs/2103.14259
#   - DINO: DETR with Improved DeNoising Anchor Boxes (ICLR 2023),
#     dynamic-k variant of its HungarianMatcher
#     https://arxiv.org/abs/2203.03605
#   - UVO Challenge winner adaptive center sampling
#     https://github.com/dulucas/UVO_Challenge
# Migrated into VK-Det from mmdet/models/new_modules/new_loss.py, keeping only
# the parts used by base_filter.py. Unused helpers (loss_nc_labels,
# loss_unknown, ...) were dropped during the migration.
import torch
import torch.nn as nn
import torchvision.ops as ops

from mmdet.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou

class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg: dict = None, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1,
                 use_focal: bool = True):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cfg = cfg
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = False
        self.ota_k = 5
        self.forward_k = 10
        if self.use_focal:
            alpha = 2.0
            gamma = 0.25
            self.focal_loss_alpha = alpha
            self.focal_loss_gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]
            else:
                out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]

            indices = []
            matched_ids = []
            unknown_labels = []
            ow_indices = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue

                bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                    expanded_strides=32
                )

                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy) # pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids].cpu().numpy()

                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())

                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids] # focal-style classification cost incl. background class
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes

                bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

                bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1) # L1 cost

                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy) # GIoU cost: handles non-overlapping boxes; range [-1, 1]

                # Final cost matrix
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (
                    ~is_in_boxes_and_center)
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                #                 ==========================================================
                forward_score = torch.sum(bz_out_prob[:, :16], dim=1) # NOTE: background class excluded from the similarity sum (fixed)
                _, forward_index = torch.topk(forward_score, self.forward_k, largest=True, sorted=True)
                forward_index_list = forward_index.cpu().numpy().tolist()
                unknown_label = []
                for each in forward_index_list:
                    if each not in matched_qidx.cpu().numpy().tolist():
                        unknown_label.append(each)
                unknown_indices_batchi_a = copy.deepcopy(indices_batchi[0]) # gt_lables
                unknown_indices_batchi_b = torch.tensor([0] * len(unknown_label), dtype=torch.long,
                                                        device=unknown_indices_batchi_a.device) # unknown_label = 17; known labels 0-15; 16 = background
                for each in range(unknown_indices_batchi_a.shape[0]):
                    if each in unknown_label:
                        unknown_indices_batchi_a[each] = True
                    else:
                        unknown_indices_batchi_a[each] = False

                unknown_labels.append(unknown_label)
                ow_indices.append((unknown_indices_batchi_a, unknown_indices_batchi_b))
                #                 ==========================================================

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        #         ==================================================
        unknown_targets = []
        for i, unknown_label in enumerate(unknown_labels):
            unity_label = [17] * len(unknown_label) # fc_unknown_cls
            unity_boxes = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_boxes_xyxy = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_image_size_xyxy_tgt = targets[i]['image_size_xyxy'].unsqueeze(0)
            unity_area = targets[i]['image_size_xyxy'].unsqueeze(0)
            unknown_target = {'labels': torch.tensor(unity_label, dtype=torch.long, device=targets[0]['labels'].device),
                              'boxes': unity_boxes.repeat(len(unknown_label), 1),
                              'image_size_xyxy': targets[i]['image_size_xyxy'],
                              'boxes_xyxy': unity_boxes_xyxy.repeat(len(unknown_label), 1),
                              'image_size_xyxy_tgt': unity_image_size_xyxy_tgt.repeat(len(unknown_label), 1),
                              'area': unity_area.repeat(len(unknown_label), 1)}
            unknown_targets.append(unknown_target)
        #         ==================================================

        return indices, matched_ids, ow_indices, unknown_targets

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (
                    target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (
                    target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (
                    target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (
                    target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # anchor inside box OR center region
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)  # anchor inside both box AND center region

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0) # ious_in_boxes_matrix.cpu().numpy()
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

