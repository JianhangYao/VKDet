
import mmcv
import numpy as np
from PIL import Image
from .api_wrappers import COCO
import os.path as osp
import xml.etree.ElementTree as ET
from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict
from mmdet.core import eval_map, eval_recalls
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
@DATASETS.register_module()




class XMLDataset_DOTA(CustomDataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """
    CLASSES = (
        'plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'roundabout', 'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool'
    )

    # 4 class names in order, obtained from load_coco_json() function
    UNSEEN_CLS = ['tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']

    # 16 class names in order, obtained from load_coco_json() function
    SEEN_CLS = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'roundabout']

    # 20 class names in order, obtained from load_coco_json() function
    OVD_ALL_CLS = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'roundabout', 'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']

    OVD_Unknown_CLS = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'roundabout', 'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool',
                   'Unknown Class1', 'Unknown Class2', 'Unknown Class3', 'Unknown Class4', 'Unknown Class5', 'Unknown Class6',
                    'Unknown Class7', 'Unknown Class8', 'Unknown Class9', 'Unknown Class10', 'Unknown Class11', 'Unknown Class12',
                    'Unknown Class13', 'Unknown Class14', 'Unknown Class15', 'Unknown Class16', 'Unknown Class17', 'Unknown Class18', 'Unknown Class19', 'Unknown Class20',
                   'Unknown Class21', 'Unknown Class22', 'Unknown Class23', 'Unknown Class24', 'Unknown Class25', 'Unknown Class26', 'Unknown Class27', 'Unknown Class28',
                   'Unknown Class29', 'Unknown Class30'
                       ]

    Unknown_CLS = ['Unknown Class1', 'Unknown Class2', 'Unknown Class3', 'Unknown Class4', 'Unknown Class5', 'Unknown Class6',
                    'Unknown Class7', 'Unknown Class8', 'Unknown Class9', 'Unknown Class10', 'Unknown Class11', 'Unknown Class12',
                    'Unknown Class13', 'Unknown Class14', 'Unknown Class15', 'Unknown Class16', 'Unknown Class17', 'Unknown Class18', 'Unknown Class19', 'Unknown Class20',
                   'Unknown Class21', 'Unknown Class22', 'Unknown Class23', 'Unknown Class24', 'Unknown Class25',
                   'Unknown Class26', 'Unknown Class27', 'Unknown Class28', 'Unknown Class29', 'Unknown Class30'
                   ]

    type = 'val'



    def __init__(self, min_size=None,
                 is_class_agnostic=False,
                 **kwargs):
        self.is_class_agnostic = is_class_agnostic

        self.min_size = min_size

        # assert self.CLASSES or kwargs.get("classes", None), "CLASSES in `XMLDataset_DOTA` can not be None."
        super(XMLDataset_DOTA, self).__init__(**kwargs)


    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of the XML (or txt list) file.
        Returns:
            list[dict]: Annotation info from XML file.
        """
        if ann_file.endswith('.txt'):
            if self.is_class_agnostic:
                self.cat2label = {cls: 0 for cat_id, cls in enumerate(self.CLASSES)}
            else:
                self.cat2label = {
                    cls: cat_id for cat_id, cls in enumerate(self.CLASSES)}

            # assign numeric ids to classes
            self.class_to_code = {category: index for index, category in enumerate(self.CLASSES)}
            self.train_cat_ids = [self.class_to_code[cls] for cls in self.SEEN_CLS]
            self.eval_cat_ids = [self.class_to_code[cls] for cls in self.UNSEEN_CLS]

            data_infos = []
            img_ids = mmcv.list_from_file(ann_file)
            img_ids = [p.replace('.xml', '') for p in img_ids]
            for img_id in img_ids:

                if self.type == 'filter_train':
                    filename = f"train/JPEGImages-train/{img_id}.jpg"
                    xml_path = osp.join(self.img_prefix, "Annotations/Filtered_Annotations", f"{img_id}.xml")
                elif self.type == 'whole_train':
                    filename = f"train/JPEGImages-train/{img_id}.jpg"
                    xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/train_with_objects", f"{img_id}.xml")
                elif self.type == 'val':
                    filename = f"val/JPEGImages-val/{img_id}.jpg"
                    xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/val_with_objects", f"{img_id}.xml")
                else:
                    print('Path not found')

                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find("size")
                if size is not None:
                    width = int(size.find("width").text)
                    height = int(size.find("height").text)
                else:
                    img_path = osp.join(self.img_prefix, "JPEGImages", "{}.jpg".format(img_id))
                    img = Image.open(img_path)
                    width, height = img.size
                data_infos.append(dict(id=img_id, filename=filename, width=width, height=height))
            return data_infos
        # Load fine-tuning data
        elif ann_file.endswith('.json'):
            self.dota = COCO(ann_file)
            # The order of returned `cat_ids` will not
            # change with the order of the CLASSES
            self.cat_ids = self.dota.get_cat_ids(cat_names=self.OVD_Unknown_CLS)
            # self.cat_ids_base = self.dota.get_cat_ids(cat_names=self.SEEN_CLS)
            # self.cat_ids_novel = self.dota.get_cat_ids(cat_names=self.UNSEEN_CLS)

            self.cat2label = {cls: i for i, cls in enumerate(self.OVD_Unknown_CLS)} # used in evaluation (with CLASSES)
            self.img_ids = self.dota.get_img_ids()
            # print(self.img_ids)
            data_infos = []
            total_ann_ids = []
            for i in self.img_ids:
                info = self.dota.load_imgs([i])[0]
                info["filename"] = info["file_name"]
                data_infos.append(info)
                ann_ids = self.dota.get_ann_ids(img_ids=[i])
                total_ann_ids.extend(ann_ids)
            assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
            return data_infos
        else:
            raise ValueError(f"Unsupported annotation format: {ann_file}")


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info["width"], img_info["height"]) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = str(img_info["id"]).zfill(5)
                xml_path = osp.join(self.img_prefix, "Annotations/Filtered_Annotations", f"{img_id}.xml")
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]["id"]
        # img_id = str(img_id).zfill(5)

        if self.type == 'filter_train':
            xml_path = osp.join(self.img_prefix, "Annotations/Filtered_Annotations", f"{img_id}.xml")
        elif self.type == 'whole_train':
            xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/train_with_objects",
                                f"{img_id}.xml")
        elif self.type == 'val':
            xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/val_with_objects",
                                f"{img_id}.xml")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.CLASSES:
                continue

            label = self.cat2label[name]
            difficult = obj.find("difficult")
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find("bndbox")
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
        )

        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]["id"]
        if self.type == 'filter_train':
            xml_path = osp.join(self.img_prefix, "Annotations/Filtered_Annotations", f"{img_id}.xml")
        elif self.type == 'whole_train':
            xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/train_with_objects",
                                f"{img_id}.xml")
        elif self.type == 'val':
            xml_path = osp.join(self.img_prefix, "Annotations/Horizontal Bounding Boxes/val_with_objects",
                                f"{img_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def evaluate(
        self, results, metric="mAP", logger=None, proposal_nums=(100, 300, 1000), iou_thr=0.5, scale_ranges=None
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """


        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["mAP", "recall"]
        if metric not in allowed_metrics:
            raise KeyError(f"metric {metric} is not supported")
        annotations = [self.get_ann_info(i) for i in range(len(self))]



        eval_results = OrderedDict()
        if metric == "mAP":
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results, annotations, scale_ranges=scale_ranges, iou_thr=iou_thr, dataset=self.CLASSES, logger=logger
            )
            eval_results["mAP"] = mean_ap

        elif metric == "recall":
            gt_bboxes = [ann["bboxes"] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f"AR@{num}"] = ar[i]
        return eval_results
