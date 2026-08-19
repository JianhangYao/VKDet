# Author: Yao
# CreatTime: 2025/10/17
# FileName: gt_annotation_generate
# Description: simple introduction of the code
from __future__ import annotations
import os
import xml.etree.ElementTree as ET

def generate_gt_annotations_from_xml(xml_dir, class_names=None):
    gt_annotations = {}

    # Build class-name -> id mapping if a class list is provided
    class_name_to_id = {}
    if class_names:
        class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        # Image id from filename
        image_id = os.path.splitext(xml_file)[0]

        if image_id not in gt_annotations:
            gt_annotations[image_id] = {
                'boxes': [],
                'labels': []
            }

        # Extract objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text

            # Map class name to id if a class list is provided
            if class_names:
                if class_name in class_name_to_id:
                    label = class_name_to_id[class_name]
                else:
                    # Skip classes not in the list (or use a sentinel value)
                    continue
            else:
                # Otherwise use the raw class name
                label = class_name

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            gt_annotations[image_id]['boxes'].append([xmin, ymin, xmax, ymax])
            gt_annotations[image_id]['labels'].append(label)

    return gt_annotations