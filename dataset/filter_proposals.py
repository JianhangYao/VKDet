#!/usr/bin/env python
"""Filter OLN proposals with VK-Det's shared geometry filter."""

import argparse
import pickle

from mmdet.our_utils.filter_thr_p import filter_proposals


def main():
    parser = argparse.ArgumentParser(
        description="Filter proposal boxes by width, height, and area"
    )
    parser.add_argument("--src", required=True, help="input proposal pkl")
    parser.add_argument("--dst", required=True, help="output filtered pkl")
    parser.add_argument("--min-width", type=float, default=32.0)
    parser.add_argument("--min-height", type=float, default=32.0)
    parser.add_argument("--min-area", type=float, default=1024.0)
    args = parser.parse_args()

    with open(args.src, "rb") as file:
        proposals = pickle.load(file)

    filtered, stats = filter_proposals(
        proposals,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
    )

    with open(args.dst, "wb") as file:
        pickle.dump(filtered, file)

    print(
        "Proposal filtering: "
        f"{stats['original']} -> {stats['filtered']} "
        f"(deleted {stats['deleted']} from "
        f"{stats['invalid_images']} images)"
    )
    print(f"Saved to: {args.dst}")


if __name__ == "__main__":
    main()
