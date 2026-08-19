# Author: Yao
# CreatTime: 2025/10/17
# FileName: visual_proposal
# Description: simple introduction of the code
import os
from tqdm import tqdm
import cv2
def visualize_from_json(json_data, image_base_dir, output_dir, font_path=cv2.FONT_HERSHEY_SIMPLEX, max_per_class=100):
    """
     Render proposal visualization directly from a JSON file.
     Args:
         json_path (str): path to the pseudo-label JSON (e.g. dior_vild_proposal100.json)
         image_base_dir (str): DIOR image root (e.g. /path/to/DIOR/)
         output_dir (str): output directory
         font_path (str): font file path
         max_per_class (int): max samples to visualize per class
     """
    print("\n======================== Visualizing pseudo-label proposals ========================\n")
    id_to_image = {img["id"]: img for img in json_data["images"]}
    id_to_category = {cat["id"]: cat["name"] for cat in json_data["categories"]}

    # group annotations by class
    category_anns = {}
    for ann in json_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in category_anns:
            category_anns[cat_id] = []
        category_anns[cat_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    # drawing params consistent with imshow_det_bboxes
    BG_COLOR = (0, 0, 0)
    BG_ALPHA = 0.7
    TEXT_PADDING = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    bbox_color = (0, 255, 0)
    text_color = (200, 200, 200)

    for cat_id, anns in tqdm(category_anns.items()):
        category_name = f"unknown_{cat_id}"

        class_dir = os.path.join(output_dir, f"{category_name}_id{cat_id}")
        os.makedirs(class_dir, exist_ok=True)

        # cap per-class count
        processed = 0

        # group annotations by image id
        img_anns = {}
        for ann in anns:
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)
        for img_id, img_anns_list in img_anns.items():
            if processed >= max_per_class:
                break

            img_info = id_to_image.get(img_id)
            if not img_info:
                continue

            img_path = os.path.join(image_base_dir, img_info["file_name"])

            # load with OpenCV instead of PIL
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f"Failed to load image {img_path}")
                    continue
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue

            # semi-transparent label per annotation
            for ann in img_anns_list:
                bbox = ann["bbox"]
                cos = ann["dis"]
                x, y, w, h = bbox
                bbox_int = [int(x), int(y), int(x + w), int(y + h)]

                cv2.rectangle(
                    cv_img,
                    (bbox_int[0], bbox_int[1]),
                    (bbox_int[2], bbox_int[3]),
                    bbox_color,
                    thickness
                )

                label_text = f"{category_name}: {cos:.2f}"

                text_size, _ = cv2.getTextSize(label_text, FONT, font_scale, 1)
                text_width, text_height = text_size

                # label background at top-left inside the box
                bg_top = bbox_int[1]
                bg_left = bbox_int[0]
                bg_bottom = bg_top + text_height + TEXT_PADDING * 2
                bg_right = bg_left + text_width + TEXT_PADDING * 2

                overlay = cv_img.copy()
                cv2.rectangle(
                    overlay,
                    (bg_left, bg_top),
                    (bg_right, bg_bottom),
                    BG_COLOR,
                    -1
                )

                cv_img = cv2.addWeighted(overlay, BG_ALPHA, cv_img, 1 - BG_ALPHA, 0)

                text_x = bg_left + TEXT_PADDING
                text_y = bg_top + text_height + TEXT_PADDING
                cv2.putText(
                    cv_img,
                    label_text,
                    (text_x, text_y),
                    FONT,
                    font_scale,
                    text_color,
                    1
                )

            # save with image-id + annotation-count naming

            save_path = os.path.join(
                class_dir,
                f"img_{img_id}_anns{len(img_anns_list)}.jpg"
            )
            cv2.imwrite(save_path, cv_img)
            processed += 1
    print(f"Pseudo-label visualization saved to: {save_path}")

# Example usage
if __name__ == "__main__":

    visualize_from_json(
        json_data="data/dior_vild_proposal500.json",
        image_base_dir="data/DIOR/",
        output_dir="workdirs/visualization_results",
        font_path=cv2.FONT_HERSHEY_SIMPLEX, # cv2.FONT_HERSHEY_SIMPLEX "arial.ttf"
        max_per_class=100
    )
