# Dataset Preparation

Dataset preparation tools for VK-Det (DIOR).

## Files

| File | Purpose |
|---|---|
| `prepare_dior.py` | One-click preprocessing: generates `filtered_train.txt` + `Filtered_Annotations/` |
| `filter_proposals.py` | Applies VK-Det's shared width/height/area filter to OLN proposals |

## One-click preparation

```bash
# Prerequisite: raw DIOR unpacked under data/DIOR/ (Annotations/, ImageSets/, JPEGImages-trainval/)
python dataset/prepare_dior.py --dior_root data/DIOR
```

Outputs (idempotent, safe to re-run):

1. `ImageSets/Main/filtered_train.txt` — 4,933 base-class training images
2. `Annotations/Filtered_Annotations/` — matching XMLs with novel-class objects (airport / basketballcourt / groundtrackfield / windmill) removed

Add `--force` to rebuild filtered annotations.

## Proposal filtering (optional)

Filter degenerate boxes from OLN-generated proposals before use:

```bash
python dataset/filter_proposals.py \
    --src raw_train_proposals.pkl \
    --dst proposals/dior/invalid_box_filter/filtered_train_proposals.pkl \
    --min-width 32 --min-height 32 --min-area 1024
```
