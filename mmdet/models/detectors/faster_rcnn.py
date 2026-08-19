from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import random
import math
import numpy as np

@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, *args, backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=None, pretrained=None, **kwargs):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )



    # def crop_proposals(self, img, img_metas, gt_bboxes, gt_labels):
    #     return None




    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs
    ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """



        # patch_size = 16
        # pred_ratio =  [0.0, 0.3]
        # pred_ratio_var = [0.0, 0.2]
        # pred_aspect_ratio = (0.3, 1 / 0.3)
        # pred_shape = 'block'
        # pred_start_epoch=0



        # #====================init parameter===================
        # psz = patch_size
        #
        # pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and len(pred_ratio) == 1 else pred_ratio
        # pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1 else pred_ratio_var
        # if isinstance(pred_ratio) and not isinstance(pred_ratio_var, list):
        #     pred_ratio_var = [pred_ratio_var] * len(pred_ratio)
        # log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        #
        #
        #
        #
        #
        #
        #
        # def get_pred_ratio(self):
        #     if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
        #         return 0
        #     if isinstance(self.pred_ratio, list):
        #         pred_ratio = []
        #         for prm, prv in zip(pred_ratio, pred_ratio_var):
        #             assert prm >= prv
        #             pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
        #             pred_ratio.append(pr)
        #         pred_ratio = random.choice(pred_ratio)
        #     else:
        #         assert self.pred_ratio >= self.pred_ratio_var
        #         pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        #     return pred_ratio
        #
        #
        # def set_epoch(self, epoch):
        #     self.epoch = epoch
        #
        #
        #
        # def __getitem__(self, index):
        #     proposals = crop_proposals(img, img_metas, gt_bboxes, gt_labels)
        #     masks = []
        #     for proposal in proposals:
        #         try:
        #             H, W = proposal.shape[1] // psz, proposal.shape[2] // psz
        #         except:
        #             continue
        #
        #         high = get_pred_ratio() * H * W















        x = self.extract_feat(img, img_metas)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, proposals, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs
        )
        losses.update(roi_losses)

        return losses
