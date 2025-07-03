import os
import pickle
import argparse
import logging
import sys
import time
from pathlib import Path
import csv  # For CSV output

import numpy as np
from PIL import Image
import cv2  # For visualization
from tqdm import tqdm

# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Custom Type for Boolean Argument ---
def str_to_bool(value):
    """Converts string 'true'/'false' to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got '{value}'")


# --- Core Functions (No changes here from previous version, as they are independent of argument parsing) ---

def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union (IoU) between two boolean masks."""
    if mask1.shape != mask2.shape:
        return 0.0

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def find_best_sam_subset_for_gt(sam_masks_list, gt_mask, target_img_shape):
    """
    Finds the subset of SAM masks that maximizes IoU with the gt_mask using a greedy approach.
    Returns: best_iou, selected_indices_original, current_combined_mask
    """
    current_combined_mask = np.zeros(target_img_shape, dtype=bool)
    if gt_mask is None or gt_mask.shape != target_img_shape:
        return 0.0, [], current_combined_mask

    current_best_iou = calculate_iou(current_combined_mask, gt_mask)
    selected_indices_original = []

    if not sam_masks_list:
        return current_best_iou, [], current_combined_mask

    valid_sam_masks_with_indices = []
    for i, m_dict in enumerate(sam_masks_list):
        if 'segmentation' not in m_dict or 'area' not in m_dict or m_dict['area'] <= 0:
            continue
        sam_mask_np = m_dict['segmentation']
        if sam_mask_np.shape != target_img_shape:
            continue
        valid_sam_masks_with_indices.append({'original_index': i, 'mask': sam_mask_np.astype(bool)})

    if not valid_sam_masks_with_indices:
        return current_best_iou, [], current_combined_mask

    candidate_pool_indices = list(range(len(valid_sam_masks_with_indices)))

    while candidate_pool_indices:
        iou_of_best_next_state = -1.0
        best_candidate_local_idx_to_add = -1
        potential_combined_mask_for_best_candidate = None

        for local_idx_in_pool in candidate_pool_indices:
            sam_mask_candidate_data = valid_sam_masks_with_indices[local_idx_in_pool]
            sam_mask_candidate = sam_mask_candidate_data['mask']

            potential_new_combined_mask = np.logical_or(current_combined_mask, sam_mask_candidate)
            iou_with_candidate_added = calculate_iou(potential_new_combined_mask, gt_mask)

            if iou_with_candidate_added > iou_of_best_next_state:
                iou_of_best_next_state = iou_with_candidate_added
                best_candidate_local_idx_to_add = local_idx_in_pool
                potential_combined_mask_for_best_candidate = potential_new_combined_mask

        if best_candidate_local_idx_to_add != -1 and iou_of_best_next_state > current_best_iou:
            current_best_iou = iou_of_best_next_state
            current_combined_mask = potential_combined_mask_for_best_candidate

            original_idx = valid_sam_masks_with_indices[best_candidate_local_idx_to_add]['original_index']
            selected_indices_original.append(original_idx)

            candidate_pool_indices.remove(best_candidate_local_idx_to_add)
        else:
            break

    return current_best_iou, selected_indices_original, current_combined_mask


def generate_and_save_visualizations(
        original_image_pil, gt_segment_mask, combined_sam_mask,  # combined_sam_mask is the oracle mask
        image_id, segment_label_val, iou_score,
        target_img_shape, output_dir_for_visuals,
        all_precomputed_sam_masks_list  # <--- NEW PARAMETER: This is the full list of SAM masks for the image
):
    """Generates and saves visualizations for low IoU cases, including all precomputed SAM masks."""
    if original_image_pil is None:
        logger.warning(
            f"Cannot generate visualization for {image_id}_seg{segment_label_val}: Original image not loaded.")
        return

    try:
        output_dir_for_visuals.mkdir(parents=True, exist_ok=True)

        img_pil_resized = original_image_pil.resize(target_img_shape[::-1], Image.LANCZOS)
        img_np_rgb = np.array(img_pil_resized)
        if img_np_rgb.ndim == 2:
            img_np_rgb = cv2.cvtColor(img_np_rgb, cv2.COLOR_GRAY2RGB)
        elif img_np_rgb.shape[2] == 4:
            img_np_rgb = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        filename_prefix = output_dir_for_visuals / f"{image_id}_seg{segment_label_val}_iou{iou_score:.2f}"

        # 1. Original image overlaid with SAM oracle mask (selected subset)
        sam_overlay_color = np.zeros_like(img_bgr)
        sam_overlay_color[combined_sam_mask] = [255, 0, 0]  # Blue for SAM oracle
        blended_sam_on_image = cv2.addWeighted(img_bgr, 0.7, sam_overlay_color, 0.3, 0)
        cv2.imwrite(f"{filename_prefix}_overlay_sam_oracle_on_image.png", blended_sam_on_image)

        # 2. Standalone colored SAM oracle mask (selected subset)
        sam_mask_standalone_colored = np.zeros((*target_img_shape, 3), dtype=np.uint8)
        sam_mask_standalone_colored[combined_sam_mask] = [255, 165, 0]  # Orange
        cv2.imwrite(f"{filename_prefix}_sam_oracle_mask_colored.png", sam_mask_standalone_colored)

        # 3. (Bonus) Original image overlaid with GT mask
        gt_overlay_color = np.zeros_like(img_bgr)
        gt_overlay_color[gt_segment_mask] = [0, 255, 0]  # Green for GT
        blended_gt_on_image = cv2.addWeighted(img_bgr, 0.7, gt_overlay_color, 0.3, 0)
        cv2.imwrite(f"{filename_prefix}_overlay_gt_on_image.png", blended_gt_on_image)

        # 4. (Bonus) Original image with GT, SAM Oracle, and Overlap
        comparison_overlay = np.copy(img_bgr)
        comparison_overlay[np.logical_and(gt_segment_mask, ~combined_sam_mask)] = \
            (0.3 * img_bgr[np.logical_and(gt_segment_mask, ~combined_sam_mask)] + 0.7 * np.array([0, 255, 0])).astype(
                np.uint8)
        comparison_overlay[np.logical_and(~gt_segment_mask, combined_sam_mask)] = \
            (0.3 * img_bgr[np.logical_and(~gt_segment_mask, combined_sam_mask)] + 0.7 * np.array([255, 0, 0])).astype(
                np.uint8)
        intersection_mask = np.logical_and(gt_segment_mask, combined_sam_mask)
        comparison_overlay[intersection_mask] = \
            (0.3 * img_bgr[intersection_mask] + 0.7 * np.array([0, 255, 255])).astype(np.uint8)
        cv2.imwrite(f"{filename_prefix}_overlay_oracle_comparison.png", comparison_overlay)

        # --- NEW VISUALIZATION: All precomputed SAM masks combined ---
        all_sam_combined_mask = np.zeros(target_img_shape, dtype=bool)
        if all_precomputed_sam_masks_list:
            for m_dict in all_precomputed_sam_masks_list:
                # Ensure mask is valid and correct shape before combining
                if 'segmentation' in m_dict and m_dict['area'] > 0 and m_dict['segmentation'].shape == target_img_shape:
                    all_sam_combined_mask = np.logical_or(all_sam_combined_mask, m_dict['segmentation'].astype(bool))

        if all_sam_combined_mask.any():  # Only save if there's actually something to show
            # 5. Original image overlaid with ALL precomputed SAM masks
            all_sam_overlay_color = np.zeros_like(img_bgr)
            all_sam_overlay_color[all_sam_combined_mask] = [255, 255, 0]  # Cyan for ALL SAM masks
            blended_all_sam_on_image = cv2.addWeighted(img_bgr, 0.7, all_sam_overlay_color, 0.3, 0)
            cv2.imwrite(f"{filename_prefix}_overlay_all_sam_on_image.png", blended_all_sam_on_image)

            # 6. Standalone colored mask of ALL precomputed SAM masks
            all_sam_standalone_colored = np.zeros((*target_img_shape, 3), dtype=np.uint8)
            all_sam_standalone_colored[all_sam_combined_mask] = [255, 255, 0]  # Cyan
            cv2.imwrite(f"{filename_prefix}_all_sam_masks_combined_colored.png", all_sam_standalone_colored)

    except Exception as e:
        logger.error(f"Error generating visualization for {image_id}_seg{segment_label_val}: {e}", exc_info=False)


# --- Main Evaluation Script ---
def evaluate_sam_oracle_generic(args):
    precomputed_sam_masks_dir = Path(args.sam_masks_dir)
    pascal_gt_dir = Path(args.pascal_gt_dir)
    pascal_jpeg_dir = Path(args.pascal_jpeg_dir) if args.pascal_jpeg_dir else None
    target_img_shape = (args.img_size, args.img_size)
    output_csv_path = Path(args.output_csv) if args.output_csv else None

    enable_visualizations = args.visualize
    low_iou_output_dir_path = Path(args.low_iou_output_dir) if enable_visualizations else None

    if not precomputed_sam_masks_dir.is_dir():
        logger.error(f"SAM precomputed masks directory not found: {precomputed_sam_masks_dir}")
        return
    if not pascal_gt_dir.is_dir():
        logger.error(f"PASCAL VOC GT directory not found: {pascal_gt_dir}")
        return

    if enable_visualizations:
        if not pascal_jpeg_dir:
            logger.warning(
                "Visualizations enabled, but --pascal_jpeg_dir is not provided. Visualizations will be skipped.")
            enable_visualizations = False
        elif not pascal_jpeg_dir.is_dir():
            logger.warning(f"--pascal_jpeg_dir {pascal_jpeg_dir} not found. Visualizations will be skipped.")
            enable_visualizations = False
        else:
            logger.info(
                f"Visualizations enabled. Low IoU threshold: {args.low_iou_threshold}. Output to: {low_iou_output_dir_path}")

    sam_mask_files = sorted(list(precomputed_sam_masks_dir.glob('*_sam_hybrid_masks.pkl')))
    if not sam_mask_files:
        logger.error(f"No SAM mask .pkl files found in {precomputed_sam_masks_dir}")
        return

    logger.info(f"Found {len(sam_mask_files)} SAM mask .pkl files to process.")

    all_max_ious_for_segments = []
    image_level_segment_ious = {}
    detailed_results_for_csv = []

    images_with_errors = 0
    total_gt_segments_evaluated = 0

    csv_file = None
    csv_writer = None
    if output_csv_path:
        try:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(output_csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['image_id', 'segment_label', 'max_iou'])
        except IOError as e:
            logger.error(f"Could not open CSV file {output_csv_path} for writing: {e}")
            csv_writer = None

    pbar = tqdm(sam_mask_files, desc="Evaluating SAM Oracle")
    for sam_mask_file_path in pbar:
        image_id = sam_mask_file_path.stem.replace('_sam_hybrid_masks', '')
        gt_annotation_path = pascal_gt_dir / f"{image_id}.png"
        original_image_path = pascal_jpeg_dir / f"{image_id}.jpg" if pascal_jpeg_dir else None

        original_image_pil = None
        if enable_visualizations and original_image_path and original_image_path.exists():
            try:
                original_image_pil = Image.open(original_image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load original image {original_image_path} for {image_id}: {e}")

        if not gt_annotation_path.exists():
            logger.debug(f"GT annotation file missing for {image_id} ({gt_annotation_path}). Skipping.")
            images_with_errors += 1
            continue

        try:
            with open(sam_mask_file_path, 'rb') as f:
                sam_masks_for_image = pickle.load(f)  # <-- This is the full list of SAM masks for the image
        except Exception as e:
            logger.error(f"Error loading SAM masks for {image_id} from {sam_mask_file_path}: {e}")
            images_with_errors += 1
            continue

        if not isinstance(sam_masks_for_image, list):
            logger.warning(f"SAM mask file {sam_mask_file_path} does not contain a list. Skipping.")
            images_with_errors += 1
            continue

        try:
            gt_full_pil = Image.open(gt_annotation_path)
            gt_full_pil_resized = gt_full_pil.resize(target_img_shape[::-1], Image.NEAREST)
            gt_full_np = np.array(gt_full_pil_resized)
        except Exception as e:
            logger.error(f"Error loading or resizing GT for {image_id} from {gt_annotation_path}: {e}")
            images_with_errors += 1
            continue

        unique_gt_labels = np.unique(gt_full_np)

        current_image_segment_ious = []
        for label_val in unique_gt_labels:
            if label_val == 0: continue
            if label_val == 255: continue

            gt_mask_for_segment = (gt_full_np == label_val).astype(bool)
            if not gt_mask_for_segment.any(): continue

            best_iou, _, combined_sam_mask = find_best_sam_subset_for_gt(
                sam_masks_for_image, gt_mask_for_segment, target_img_shape
            )

            all_max_ious_for_segments.append(best_iou)
            current_image_segment_ious.append(best_iou)
            if csv_writer:
                detailed_results_for_csv.append([image_id, int(label_val), float(best_iou)])

            total_gt_segments_evaluated += 1

            if enable_visualizations and best_iou < args.low_iou_threshold:
                if original_image_pil:
                    generate_and_save_visualizations(
                        original_image_pil, gt_mask_for_segment, combined_sam_mask,
                        image_id, int(label_val), best_iou,
                        target_img_shape, low_iou_output_dir_path,
                        sam_masks_for_image  # <--- PASS THE FULL LIST OF SAM MASKS HERE
                    )
                else:
                    logger.debug(f"Skipping visualization for {image_id}_seg{label_val} due to missing original image.")

        if current_image_segment_ious:
            image_level_segment_ious[image_id] = current_image_segment_ious

        if all_max_ious_for_segments:
            running_mean_iou = np.mean(all_max_ious_for_segments)
            pbar.set_postfix_str(f"Running Mean Max Segment IoU: {running_mean_iou:.4f}")

    if csv_writer and csv_file:
        csv_writer.writerows(detailed_results_for_csv)
        csv_file.close()
        logger.info(f"Detailed segment IoUs saved to: {output_csv_path}")

    logger.info("\n" + "=" * 30 + " SAM Oracle Mean Max IoU Summary " + "=" * 30)
    logger.info(f"Total SAM mask files processed (or attempted): {len(sam_mask_files)}")
    logger.info(f"Images skipped due to errors/missing files: {images_with_errors}")
    logger.info(f"Total individual ground truth segments evaluated: {total_gt_segments_evaluated}")

    if all_max_ious_for_segments:
        overall_mean_max_iou = np.mean(all_max_ious_for_segments)
        logger.info(f"Overall Mean of Maximum Achievable IoUs (across all GT segments): {overall_mean_max_iou:.4f}")
    else:
        logger.warning("No ground truth segments were evaluated. Cannot compute mean max IoU.")

    logger.info("\n--- Image-level Average Max Segment IoUs ---")
    image_average_ious_list = []
    for img_id, seg_ious in image_level_segment_ious.items():
        if seg_ious:
            avg_iou_for_img = np.mean(seg_ious)
            image_average_ious_list.append({'id': img_id, 'avg_iou': avg_iou_for_img})

    if image_average_ious_list:
        image_average_ious_list.sort(key=lambda x: x['avg_iou'])
        logger.info(f"\nTop 5 WORST performing images (by avg segment IoU):")
        for item in image_average_ious_list[:5]:
            logger.info(f"  Image '{item['id']}': Average Max Segment IoU = {item['avg_iou']:.4f}")

        logger.info(f"\nTop 5 BEST performing images (by avg segment IoU):")
        for item in image_average_ious_list[-5:]:
            logger.info(f"  Image '{item['id']}': Average Max Segment IoU = {item['avg_iou']:.4f}")

        overall_mean_of_image_avg_ious = np.mean([item['avg_iou'] for item in image_average_ious_list])
        logger.info(f"\nMean of Image-Average Max Segment IoUs: {overall_mean_of_image_avg_ious:.4f}")
    else:
        logger.info("No image-level average IoUs to report.")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM Oracle Performance with enhanced analysis.")
    parser.add_argument('--sam_masks_dir', type=str, default=r"C:\Users\Khan\PycharmProjects\FSS-Research-\pascal\pascal_sam_precomputed_masks_hybrid\JPEGImages",
                        help="Directory containing the pre-computed SAM mask .pkl files.")
    parser.add_argument('--pascal_gt_dir', type=str, default=r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw\SegmentationClassAug",
                        help="Path to the PASCAL VOC ground truth SegmentationClass directory.")
    parser.add_argument('--pascal_jpeg_dir', type=str, default=r'C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw\JPEGImages',  # Example path, adjust as needed,
                        help="(Optional) Path to PASCAL VOC JPEGImages for visualization.")
    parser.add_argument('--img_size', type=int, default=256,
                        help="Image size (square) for SAM masks and GT resizing.")
    parser.add_argument('--output_csv', type=str, default="sam_oracle_segment_ious.csv",
                        help="Path to save the detailed CSV output of segment IoUs.")

    parser.add_argument('--visualize', type=str_to_bool, default=True,
                        help="Enable or disable saving visualizations for segments with IoU below threshold. "
                             "Set to 'True' or 'False'. Default: False.")

    parser.add_argument('--low_iou_threshold', type=float, default=0.5,
                        help="IoU threshold below which visualizations are saved (if --visualize is True).")
    parser.add_argument('--low_iou_output_dir', type=str, default="low_iou_visualizations",
                        help="Directory to save low IoU visualizations (if --visualize is True).")

    args = parser.parse_args()

    start_time = time.time()
    evaluate_sam_oracle_generic(args)
    end_time = time.time()
    logger.info(f"Total evaluation time: {end_time - start_time:.2f} seconds.")
