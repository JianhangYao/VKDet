#!/usr/bin/env python
# One-click DIOR dataset preprocessing for open-vocabulary detection (OVD).
#
# Input : raw DIOR release (JPEGImages-trainval/ + Annotations/ + ImageSets/).
# Output: the filtered splits and annotations required by VK-Det training.
#
# Pipeline:
#   1. filtered_train.txt     — train.txt images with >= 1 base-class object (4,933)
#   2. Filtered_Annotations/  — matching XMLs with all novel-class objects removed
#
# Class split (paper Appendix D, following the DescReg protocol):
#   Base  (16): airplane, baseballfield, bridge, chimney, dam,
#               Expressway-Service-area, Expressway-toll-station, golffield,
#               harbor, overpass, ship, stadium, storagetank, tenniscourt,
#               trainstation, vehicle
#   Novel (4):  airport, basketballcourt, groundtrackfield, windmill
#
# Usage:
#   python dataset/prepare_dior.py --dior_root data/DIOR [--force]
#
# Notes:
#   --dior_root points to the DIOR root with JPEGImages-trainval/,
#   Annotations/ and ImageSets/Main/. Idempotent; --force rebuilds annotations.
import argparse
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

NOVEL_CLASSES = {"airport", "basketballcourt", "groundtrackfield", "windmill"}


def gen_filtered_train(dior_root, out_dir):
    """train.txt -> filtered_train.txt (keep images with >= 1 base-class object)"""
    ann_dir = os.path.join(dior_root, "Annotations/Horizontal Bounding Boxes")
    with open(os.path.join(out_dir, "train.txt")) as f:
        train_ids = [l.strip() for l in f if l.strip()]

    kept, no_ann = [], 0
    for img_id in train_ids:
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            no_ann += 1
            continue
        root = ET.parse(xml_path).getroot()
        names = {o.find("name").text for o in root.findall("object")}
        if names - NOVEL_CLASSES:          # >= 1 base class
            kept.append(img_id)

    out = os.path.join(out_dir, "filtered_train.txt")
    with open(out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"  [gen ] filtered_train.txt: {len(kept)}/{len(train_ids)} images kept"
          + (f" ({no_ann} missing xml skipped)" if no_ann else ""))
    return kept


def gen_filtered_annotations(dior_root, img_ids, force=False):
    """Copy XMLs to Filtered_Annotations/ with novel-class <object> nodes removed."""
    src_dir = os.path.join(dior_root, "Annotations/Horizontal Bounding Boxes")
    dst_dir = os.path.join(dior_root, "Annotations/Filtered_Annotations")
    if os.path.isdir(dst_dir) and not force:
        n = sum(name.endswith(".xml") for name in os.listdir(dst_dir))
        if n == len(img_ids):
            print(f"  [skip] Filtered_Annotations/ already has {n} XMLs (use --force to rebuild)")
            return

    removed = 0
    parent_dir = os.path.dirname(dst_dir)
    os.makedirs(parent_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="Filtered_Annotations.", dir=parent_dir)
    try:
        for img_id in img_ids:
            tree = ET.parse(os.path.join(src_dir, f"{img_id}.xml"))
            root = tree.getroot()
            for obj in list(root.findall("object")):
                if obj.find("name").text in NOVEL_CLASSES:
                    root.remove(obj)
                    removed += 1
            tree.write(os.path.join(tmp_dir, f"{img_id}.xml"))

        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.replace(tmp_dir, dst_dir)
        tmp_dir = None
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    print(f"  [gen ] Filtered_Annotations/: {len(img_ids)} XMLs, {removed} novel objects removed")


def main():
    ap = argparse.ArgumentParser(description="DIOR OVAD dataset preparation for VK-Det")
    ap.add_argument("--dior_root", default="data/DIOR", help="DIOR root directory")
    ap.add_argument("--force", action="store_true", help="force rebuilding Filtered_Annotations")
    args = ap.parse_args()

    root = os.path.abspath(args.dior_root)
    hbb = os.path.join(root, "Annotations/Horizontal Bounding Boxes")
    ms = os.path.join(root, "ImageSets/Main")
    for path, desc in [(hbb, "Annotations/Horizontal Bounding Boxes"),
                       (os.path.join(ms, "train.txt"), "ImageSets/Main/train.txt")]:
        if not os.path.exists(path):
            sys.exit(f"[error] missing {desc}: {path}\n"
                     f"       place the raw DIOR release here (Annotations/ImageSets/JPEGImages from the official zips)")

    print(f"[prepare_dior] root = {root}")
    print("[1/2] Generate filtered_train.txt (base-only image list)")
    img_ids = gen_filtered_train(root, ms)

    print("[2/2] Generate Filtered_Annotations/ (drop novel-class annotations)")
    gen_filtered_annotations(root, img_ids, force=args.force)


    print("[done] Dataset preparation complete:")
    print(f"  - {ms}/filtered_train.txt  ({len(img_ids)} lines)")
    print(f"  - {hbb}/../Filtered_Annotations/  ({len(img_ids)} XML)")
    print("\nNext: place OLN proposals under proposals/dior/ (see README), then start training")


if __name__ == "__main__":
    main()
