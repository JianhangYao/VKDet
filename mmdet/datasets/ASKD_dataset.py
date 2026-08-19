# Author: Yao
# CreatTime: 2025/10/14
# FileName: ASKD_dataset
# Description: simple introduction of the code
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from mmdet.datasets import XMLDataset, XMLDataset_DOTA
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import os
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

# Preprocessing defaults
n_px = 224
clip_transform = Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

class DIORCroppedProposals(XMLDataset):

    def __init__(
            self,
            pipeline = [],
            cls_mode="CSD",
            split_mode="train",
            data_root = None,
            img_prefix="JPEGImages-trainval",  # DIOR image directory
            enhance_proposals=None,
            test_mode=False,
            filter_empty_gt=True,  # DIOR usually filters empty annotations
            img_suffix='.jpg',
    ):
        self.data_root = data_root
        if cls_mode == "CSD":
            if split_mode == "train":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/train.txt")
            elif split_mode == "val":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/val.txt")
            else:
                raise ValueError(f"Unsupported ann_mode: {split_mode}. Expected 'train' or 'val'")
        elif cls_mode == "OVD":
            self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/filtered_train.txt")
        else:
            raise ValueError(f"Unsupported split_mode: {cls_mode}. Expected 'CSD' or 'OVD'")
        self.pipeline = pipeline
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.img_suffix = img_suffix

        with open(self.annotation_file, 'r') as f:
            self.original_img_names = [line.strip().replace('.xml', '') for line in f]

        self.data_infos = self._load_annotations()


        self.proposals = enhance_proposals
        self.num_proposals_per_img = list([len(p) for p in self.proposals])
        self.num_proposals_per_img.insert(0, 0)
        self.num_proposals_per_img_cum = np.cumsum(self.num_proposals_per_img)

    def _load_annotations(self):
        data_infos = []
        valid_indices = []
        for idx, img_name in enumerate(self.original_img_names):
            xml_path = os.path.join(
                self.data_root,
                 'Horizontal Bounding Boxes',
                f'{img_name}.xml'
            )
            if not os.path.exists(xml_path):
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objs = root.findall('object')
            if self.filter_empty_gt and len(objs) == 0:
                continue
            data_infos.append(dict(
                id=idx,
                img_name=img_name,
                width=width,
                height=height,
                xml_path=xml_path
            ))
        self.valid_indices = valid_indices
        return data_infos

    def get_bboxes_and_bboxes15(self, bboxes, h, w):
        def box_coord_2int(box):
            box = (
                np.floor(box[0] - 0.001),
                np.floor(box[1] - 0.001),
                np.ceil(box[2] + 0.001),
                np.ceil(box[3] + 0.001),
            )
            return (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], w),
                min(box[3], h),
            )

        assert len(bboxes) == 4
        bboxes15 = (
            1.25 * bboxes[0] - 0.25 * bboxes[2],
            1.25 * bboxes[1] - 0.25 * bboxes[3],
            1.25 * bboxes[2] - 0.25 * bboxes[0],
            1.25 * bboxes[3] - 0.25 * bboxes[1],
        )
        bboxes = box_coord_2int(bboxes)
        bboxes15 = box_coord_2int(bboxes15)
        bboxes = (bboxes[0], bboxes[1], max(bboxes[2], bboxes[0] + 1), max(bboxes[3], bboxes[1] + 1))
        bboxes15 = (bboxes15[0], bboxes15[1], max(bboxes15[2], bboxes15[0] + 1), max(bboxes15[3], bboxes15[1] + 1))
        return bboxes, bboxes15

    def __len__(self):
        return self.num_proposals_per_img_cum[-1]

    def __getitem__(self, idx):
        if len(self.num_proposals_per_img_cum) == 0:
            raise IndexError("Empty dataset")

        img_id = np.searchsorted(self.num_proposals_per_img_cum, idx, side="right") - 1
        try:
            proposal_id = idx - self.num_proposals_per_img_cum[img_id]
        except img_id as x:
            ValueError(f'Invalid img_id {x}, idx {idx}')

        img_info = self.data_infos[img_id]
        img_name = img_info['img_name']
        img_path = os.path.join(self.data_root, self.img_prefix, img_name + self.img_suffix)
        image = Image.open(img_path)
        proposal = self.proposals[img_id][proposal_id][:4]
        bboxes, bboxes15 = self.get_bboxes_and_bboxes15(proposal, img_info["height"], img_info["width"])
        img, img15 = image.crop(bboxes), image.crop(bboxes15)
        img = clip_transform(img)
        img15 = clip_transform(img15)

        return img, img15, idx


    def _set_group_flag(self):
        pass

class DIORImage(XMLDataset):

    def __init__(self,
                 pipeline=[],
                 cls_mode = "CSD",
                 split_mode = "train",
                 data_root = None,
                 img_prefix = "JPEGImages-trainval",
                 test_mode = False,
                 filter_empty_gt = True,
                 img_suffix='.jpg'
                 ):

        self.data_root = data_root
        if cls_mode == "CSD":
            if split_mode == "train":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/train.txt")
            elif split_mode == "val":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/val.txt")
            else:
                raise ValueError(f"Unsupported ann_mode: {split_mode}. Expected 'train' or 'val'")
        elif cls_mode == "OVD":
            self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/filtered_train.txt")
        else:
            raise ValueError(f"Unsupported split_mode: {cls_mode}. Expected 'CSD' or 'OVD'")

        self.pipeline = pipeline
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.img_suffix = img_suffix

        with open(self.annotation_file, 'r') as f:
            self.original_img_names = [line.strip().replace('.xml', '') for line in f]

        self.data_infos = self._load_annotations()

    def _load_annotations(self):
        data_infos = []
        valid_indices = []
        for idx, img_name in enumerate(self.original_img_names):
            xml_path = os.path.join(
                self.data_root,
                'Horizontal Bounding Boxes',
                f'{img_name}.xml'
            )
            if not os.path.exists(xml_path):
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objs = root.findall('object')
            if self.filter_empty_gt and len(objs) == 0:
                continue
            data_infos.append(dict(
                id=idx,
                img_name=img_name,
                width=width,
                height=height,
                xml_path=xml_path
            ))
        self.valid_indices = valid_indices
        return data_infos

    def get_bboxes_and_bboxes15(self, bboxes, h, w):
        def box_coord_2int(box):
            box = (
                np.floor(box[0] - 0.001),
                np.floor(box[1] - 0.001),
                np.ceil(box[2] + 0.001),
                np.ceil(box[3] + 0.001),
            )
            return (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], w),
                min(box[3], h),
            )

        assert len(bboxes) == 4
        bboxes15 = (
            1.25 * bboxes[0] - 0.25 * bboxes[2],
            1.25 * bboxes[1] - 0.25 * bboxes[3],
            1.25 * bboxes[2] - 0.25 * bboxes[0],
            1.25 * bboxes[3] - 0.25 * bboxes[1],
        )
        bboxes = box_coord_2int(bboxes)
        bboxes15 = box_coord_2int(bboxes15)
        bboxes = (bboxes[0], bboxes[1], max(bboxes[2], bboxes[0] + 1), max(bboxes[3], bboxes[1] + 1))
        bboxes15 = (bboxes15[0], bboxes15[1], max(bboxes15[2], bboxes15[0] + 1), max(bboxes15[3], bboxes15[1] + 1))
        return bboxes, bboxes15

    def __getitem__(self, idx):

        img_id = idx
        img_info = self.data_infos[img_id]
        img_name = img_info['img_name']
        img_path = os.path.join(self.data_root, self.img_prefix, img_name + self.img_suffix)
        image = Image.open(img_path)
        image = clip_transform(image)

        return image, idx, img_name

    def _set_group_flag(self):
        pass

class DOTAImage(XMLDataset_DOTA):

    def __init__(self,
                 annotation_file = "",
                 pipeline = [],
                 data_root = None,
                 img_prefix = "JPEGImages-trainval",
                 test_mode = False,
                 filter_empty_gt = True,
                 img_suffix='.jpg'
                 ):

        self.annotation_file = annotation_file
        self.pipeline = pipeline
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.img_suffix = img_suffix

        # DOTA loads image info from the image list
        with open(annotation_file, 'r') as f:
            self.original_img_names = [line.strip().replace('.xml', '') for line in f]

        # Load and filter annotations
        self.data_infos = self._load_annotations()

    def _load_annotations(self):
        data_infos = []
        valid_indices = []
        for idx, img_name in enumerate(self.original_img_names):
            xml_path = os.path.join(
                self.data_root,
                'Annotations/Horizontal Bounding Boxes/train_with_objects',
                f'{img_name}.xml'
            )
            if not os.path.exists(xml_path):
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objs = root.findall('object')
            if self.filter_empty_gt and len(objs) == 0:
                continue
            data_infos.append(dict(
                id=idx,
                img_name=img_name,
                width=width,
                height=height,
                xml_path=xml_path
            ))
        self.valid_indices = valid_indices
        return data_infos

    def __getitem__(self, idx):

        img_id = idx
        img_info = self.data_infos[img_id]
        img_name = img_info['img_name']
        img_path = os.path.join(self.data_root, self.img_prefix, img_name + self.img_suffix)
        image = Image.open(img_path)

        image = clip_transform(image)

        return image, idx, img_name

    def _set_group_flag(self):
        pass

class DOTACroppedProposals(XMLDataset_DOTA):
    n_px = 224
    clip_transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    scale = [1.0, 1.5]

    def __init__(
            self,
            pipeline = [],
            cls_mode="CSD",
            split_mode="train",
            data_root = None,
            img_prefix="JPEGImages-trainval",  # DIOR image directory
            enhance_proposals=None,
            test_mode=False,
            filter_empty_gt=True,  # DIOR usually filters empty annotations
            img_suffix='.jpg',
    ):
        self.data_root = data_root
        if cls_mode == "CSD":
            if split_mode == "train":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/cropped_images_train_list_with_objects.txt")
            elif split_mode == "val":
                self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/cropped_images_val_list_with_objects.txt")
            else:
                raise ValueError(f"Unsupported ann_mode: {split_mode}. Expected 'train' or 'val'")
        elif cls_mode == "OVD":
            self.annotation_file = os.path.join(self.data_root, "ImageSets/Main/cropped_images_train_filter_list.txt")
        else:
            raise ValueError(f"Unsupported split_mode: {cls_mode}. Expected 'CSD' or 'OVD'")
        self.pipeline = pipeline
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.img_suffix = img_suffix

        with open(self.annotation_file, 'r') as f:
            self.original_img_names = [line.strip().replace('.xml', '') for line in f]

        # Load and filter annotations
        self.data_infos = self._load_annotations()


        self.proposals = enhance_proposals
        self.num_proposals_per_img = list([len(p) for p in self.proposals])
        self.num_proposals_per_img.insert(0, 0)
        self.num_proposals_per_img_cum = np.cumsum(self.num_proposals_per_img)


    def _load_annotations(self):
        data_infos = []
        valid_indices = []
        for idx, img_name in enumerate(self.original_img_names):
            xml_path = os.path.join(
                self.data_root,
                 'Annotations/Horizontal Bounding Boxes/train_with_objects',
                f'{img_name}.xml'
            )
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objs = root.findall('object')
            if self.filter_empty_gt and len(objs) == 0:
                continue
            data_infos.append(dict(
                id=idx,
                img_name=img_name,
                width=width,
                height=height,
                xml_path=xml_path
            ))
        self.valid_indices = valid_indices
        return data_infos

    def get_bboxes_and_bboxes15(self, bboxes, h, w):
        def box_coord_2int(box):
            box = (
                np.floor(box[0] - 0.001),
                np.floor(box[1] - 0.001),
                np.ceil(box[2] + 0.001),
                np.ceil(box[3] + 0.001),
            )
            return (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], w),
                min(box[3], h),
            )  # clip boxes to image boundary

        assert len(bboxes) == 4
        bboxes15 = (
            1.25 * bboxes[0] - 0.25 * bboxes[2],
            1.25 * bboxes[1] - 0.25 * bboxes[3],
            1.25 * bboxes[2] - 0.25 * bboxes[0],
            1.25 * bboxes[3] - 0.25 * bboxes[1],
        )
        bboxes = box_coord_2int(bboxes)
        bboxes15 = box_coord_2int(bboxes15)
        bboxes = (bboxes[0], bboxes[1], max(bboxes[2], bboxes[0] + 1), max(bboxes[3], bboxes[1] + 1))
        bboxes15 = (bboxes15[0], bboxes15[1], max(bboxes15[2], bboxes15[0] + 1), max(bboxes15[3], bboxes15[1] + 1))
        return bboxes, bboxes15   # Receptive-field boxes

    def __len__(self):
        return self.num_proposals_per_img_cum[-1]

    def __getitem__(self, idx):
        if len(self.num_proposals_per_img_cum) == 0:
            raise IndexError("Empty dataset")

        img_id = np.searchsorted(self.num_proposals_per_img_cum, idx, side="right") - 1
        try:
            proposal_id = idx - self.num_proposals_per_img_cum[img_id]
        except img_id as x:
            ValueError(f'Invalid img_id {x}, idx {idx}')

        img_info = self.data_infos[img_id]
        img_name = img_info['img_name']
        img_path = os.path.join(self.data_root, self.img_prefix, img_name + self.img_suffix)
        image = Image.open(img_path)
        proposal = self.proposals[img_id][proposal_id][:4]
        bboxes, bboxes15 = self.get_bboxes_and_bboxes15(proposal, img_info["height"], img_info["width"])# Receptive-field boxes
        img, img15 = image.crop(bboxes), image.crop(bboxes15)
        img = self.clip_transform(img)
        img15 = self.clip_transform(img15)

        return img, img15, idx

    def _set_group_flag(self):
        pass