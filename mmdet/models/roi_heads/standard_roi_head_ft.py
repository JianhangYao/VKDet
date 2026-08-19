import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class StandardRoIHeadFinetune(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, test_temp=0.07, neg_pos_ub=20, centers_path = None, **kwargs):
        super(StandardRoIHeadFinetune, self).__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.text_features_for_classes = self.text_features_for_classes.float()
        self.mapping_label = {label: i for i, label in enumerate(self.unknown_label_ids)}
        self.map_unknown_label_ids = list( i for i, label in enumerate(self.unknown_label_ids))
        self.mapping_label.update({len(self.OVD_Unknown_CLS): len(self.unknown_label_ids)}) # 31

        # Original prototype config
        self.fc_cls = nn.Linear(512, len(self.unknown_label_ids) + 1, bias=True)
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)

        # Cluster centers
        from mmdet.our_utils import prep_cache
        _cached_centers = prep_cache.get('papl_centers')
        if _cached_centers is not None:
            _c = _cached_centers
        elif centers_path is not None:
            if not os.path.isfile(centers_path):
                raise FileNotFoundError(
                    f'Cluster-center file not found: {centers_path}. '
                    'Run PAPL preprocessing or provide a valid centers_path.'
                )
            _c = torch.load(centers_path, map_location='cpu')
        else:
            raise RuntimeError(
                'StandardRoIHeadFinetune requires PAPL cluster centers, '
                'but neither prep_cache["papl_centers"] nor centers_path is available.'
            )

        if not torch.is_tensor(_c):
            _c = torch.as_tensor(_c)
        if _c.dim() > 2 and _c.size(0) == 1:
            _c = _c.squeeze(0)
        if _c.dim() != 2:
            raise ValueError(f'Expected cluster centers with shape [K, D], got {_c.shape}')
        # Register as a buffer so the centers move with the model and are
        # included in state_dict/checkpoints for independent test processes.
        self.register_buffer('cluster_center', _c.detach().float().cpu())

        self.fix_bg = False
        self.test_temp = test_temp
        if self.bbox_sampler is not None:
            self.bbox_sampler.neg_pos_ub = neg_pos_ub

        if len(self.map_unknown_label_ids) != 4:
            self.fine_tune_cls = True


    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.bbox_head.eval()
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results["loss_bbox"])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""

        # -------------Classification loss---------------
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results, region_embeddings,_ = self._bbox_forward(x, rois)

        region_embeddings = self.projection(region_embeddings)

        region_embeddings = F.normalize(region_embeddings, p=2, dim=1)
        novel_weight = F.normalize(self.fc_cls.weight, p=2, dim=1)

        cls_score_text = region_embeddings @ novel_weight.T
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg, self.fine_tune_cls, len(self.OVD_Unknown_CLS))
        labels, _, _, _ = bbox_targets
        labels = labels.cpu().numpy()
        labels = torch.tensor([self.mapping_label[label] for label in labels]).long().to(self.device)

        pos_inds = (labels >= 0) & (labels < len(self.unknown_label_ids))
        pos_labels = labels[pos_inds]

        num_pos_bboxes = sum([res.pos_bboxes.size(0) for res in sampling_results])
        cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction="mean")

        loss_bbox = dict()

        loss_bbox.update(cls_loss=cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, **kwargs):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # Get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)

        # Text embeddings
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = F.normalize(bg_class_embedding, p=2, dim=1)
            normalized_weight = F.normalize(self.text_features_for_classes, p=2, dim=1)
            text_features = torch.cat([normalized_weight, bg_class_embedding], dim=0) # [21, 512]
        else:
            text_features = self.text_features_for_classes
        objectness = kwargs.get("objectness", None)

        # Score for the first head (fine-tuning)
        bbox_results, region_embeddings,_ = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)

        region_embeddings_norm = F.normalize(region_embeddings, p=2, dim=-1)

        cls_score_text = torch.matmul(region_embeddings_norm, text_features.T).float()
        cls_score_text = F.softmax(cls_score_text / self.temperature_test, dim=-1)

        novel_text_embeddings = self.text_features_for_classes[-self.novel_num:, :].detach()
        unknown_weights = self.fc_cls.weight.detach()
        unknown_weights = F.normalize(unknown_weights, dim=-1, p=2)
        sim = self.text_features_for_classes @ self.cluster_center.T
        mask, topk_i = self.filter_prototype(novel_text_embeddings, unknown_weights, self.cluster_center)
        topk_i = topk_i.to(self.device)

        novel_score_text = F.softmax((region_embeddings_norm @ unknown_weights.T) / 0.1, dim=-1)

        mask  = torch.cat([mask, torch.tensor([False], device=mask.device)])
        novel_score_text_norm = novel_score_text[..., mask]
        scale = novel_score_text_norm.size(0)
        out = torch.zeros(scale, self.novel_num, device='cuda')
        out.scatter_add_(dim=1, index=topk_i.expand(scale, -1), src=novel_score_text_norm)
        cls_score_text[..., self.novel_label_ids] = out

        # Ignore background
        cls_score_text_full = cls_score_text[:,:-1]

        # Score for the second head (distillation)
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = F.normalize(region_embeddings_image, p=2, dim=1)
            # norm_text_features = F.normalize(self.text_features_for_classes, p=2, dim=1)
            cls_score_image = region_embeddings_image @ text_features.T.float()
            cls_score_image = F.softmax((cls_score_image / self.temperature_test).float(), dim=-1)
            # Ignore background
            cls_score_image = cls_score_image[:,:-1]

        # Ensemble two heads (default setting)
        if self.ensemble:
            cls_score = torch.where(
                self.novel_index,
                cls_score_image ** (1 - self.beta) * cls_score_text_full**self.beta, # novel_classes
                cls_score_text_full ** (1 - self.alpha) * cls_score_image**self.alpha, # base_classess
            )
        else:
            cls_score = cls_score_text_full

        if objectness is not None:
            cls_score = (cls_score * objectness.unsqueeze(1)) ** 0.5

        # add score for background class (compatible with mmdet nms)
        cls_score = torch.cat([cls_score, torch.zeros(cls_score.size(0), 1, device=self.device)], dim=1)

        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels


    def filter_prototype(self, new_text_embeddings, unknown_weights, center_embeddings):
        # Noise clusters are of two kinds: low-similarity with everything
        # (background or unlabeled) and high-similarity with everything (ambiguous)
        similarity_matrix = new_text_embeddings @ center_embeddings.T
        diff = similarity_matrix.max(dim=0)[0] - similarity_matrix.min(dim=0)[0]
        mask = diff >= 2
        sub_mat = similarity_matrix[:, mask]
        _, top1_row = sub_mat.max(dim=0)
        topk_indices = torch.tensor([[int(r)] for r in top1_row]).squeeze(1)
        return mask, topk_indices

    # def filter_prototype(self, new_text_embeddings, unknown_weights, center_embeddings):
    #     similarity_matrix = new_text_embeddings @ center_embeddings.T
    #     num_prototypes = similarity_matrix.size(1)

    #     if num_prototypes < 2:
    #         raise ValueError(
    #             f"At least 2 prototypes are required, got {num_prototypes}"
    #         )

    #     _, selected_prototypes = torch.topk(
    #         similarity_matrix,
    #         k=2,
    #         dim=1,
    #         largest=True,
    #         sorted=True
    #     )

    #     mask = torch.zeros(
    #         num_prototypes,
    #         dtype=torch.bool,
    #         device=similarity_matrix.device
    #     )
    #     mask.scatter_(0, selected_prototypes.reshape(-1), True)

    #     selected_indices = mask.nonzero(as_tuple=False).squeeze(1)
    #     selected_scores = similarity_matrix[:, selected_indices]
    #     topk_indices = selected_scores.argmax(dim=0).long()

    #     return mask, topk_indices