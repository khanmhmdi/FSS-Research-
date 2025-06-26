import os
import pickle
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- Core Functions ---

def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union (IoU) between two boolean masks."""
    if mask1.shape != mask2.shape:
        logger.error(f"Mask shape mismatch: {mask1.shape} vs {mask2.shape}. Cannot calculate IoU.")
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

    Args:
        sam_masks_list (list): List of SAM mask dicts (each with 'segmentation' and 'area').
        gt_mask (np.array): Boolean ground truth mask (already resized to target_img_shape).
        target_img_shape (tuple): Expected (H, W) shape for all masks.

    Returns:
        tuple: (best_iou, list_of_selected_original_indices, combined_mask_of_selected)
    """
    if not sam_masks_list:
        return 0.0, [], np.zeros_like(gt_mask, dtype=bool)

    if gt_mask.shape != target_img_shape:
        logger.warning(f"GT mask shape {gt_mask.shape} differs from target {target_img_shape}.")
        # Fallback: try to resize GT mask, though it should be handled upstream
        try:
            gt_mask_pil = Image.fromarray(gt_mask.astype(np.uint8) * 255)
            gt_mask_pil_resized = gt_mask_pil.resize(target_img_shape[::-1], Image.NEAREST)
            gt_mask = np.array(gt_mask_pil_resized).astype(bool)
            if gt_mask.shape != target_img_shape:  # Check again
                logger.error("GT Mask resizing failed to match target shape. Returning 0 IoU.")
                return 0.0, [], np.zeros_like(target_img_shape, dtype=bool)  # Placeholder empty mask
        except Exception as e:
            logger.error(f"Error resizing GT mask: {e}. Returning 0 IoU.")
            return 0.0, [], np.zeros(target_img_shape, dtype=bool)

    valid_sam_masks_with_indices = []
    for i, m_dict in enumerate(sam_masks_list):
        if 'segmentation' not in m_dict or 'area' not in m_dict:
            logger.debug(f"SAM mask dict {i} is malformed. Skipping.")
            continue

        if m_dict['area'] <= 0:
            continue

        sam_mask_np = m_dict['segmentation']
        if sam_mask_np.shape != target_img_shape:
            logger.debug(f"SAM mask {i} shape {sam_mask_np.shape} differs from target {target_img_shape}. Skipping.")
            continue

        valid_sam_masks_with_indices.append({'original_index': i, 'mask': sam_mask_np.astype(bool)})

    if not valid_sam_masks_with_indices:
        initial_iou = calculate_iou(np.zeros_like(gt_mask, dtype=bool), gt_mask)
        return initial_iou, [], np.zeros_like(gt_mask, dtype=bool)

    selected_indices_original = []
    current_combined_mask = np.zeros_like(gt_mask, dtype=bool)
    current_best_iou = calculate_iou(current_combined_mask, gt_mask)

    candidate_pool_indices = list(range(len(valid_sam_masks_with_indices)))

    while candidate_pool_indices:
        iou_of_best_next_state = current_best_iou  # Must improve over the current best
        best_candidate_local_idx_to_add = -1
        potential_combined_mask_for_best_candidate = None

        for local_idx_in_pool in candidate_pool_indices:  # local_idx_in_pool is an index for valid_sam_masks_with_indices
            sam_mask_candidate_data = valid_sam_masks_with_indices[local_idx_in_pool]
            sam_mask_candidate = sam_mask_candidate_data['mask']

            potential_new_combined_mask = np.logical_or(current_combined_mask, sam_mask_candidate)
            iou_with_candidate_added = calculate_iou(potential_new_combined_mask, gt_mask)

            if iou_with_candidate_added > iou_of_best_next_state:
                iou_of_best_next_state = iou_with_candidate_added
                best_candidate_local_idx_to_add = local_idx_in_pool
                potential_combined_mask_for_best_candidate = potential_new_combined_mask

        if best_candidate_local_idx_to_add != -1:  # Check if any candidate was found (even if no improvement)
            # Only update if there's an actual improvement over current_best_iou
            if iou_of_best_next_state > current_best_iou:
                current_best_iou = iou_of_best_next_state
                current_combined_mask = potential_combined_mask_for_best_candidate

                original_idx = valid_sam_masks_with_indices[best_candidate_local_idx_to_add]['original_index']
                selected_indices_original.append(original_idx)

                candidate_pool_indices.remove(best_candidate_local_idx_to_add)
            else:
                # No candidate improved the IoU
                break
        else:
            # No candidate was found in the pool (pool might be empty or logic error)
            break

    return current_best_iou, selected_indices_original, current_combined_mask


# --- Main Evaluation Script ---

def evaluate_sam_oracle_generic(args):
    # The SAM masks are expected to be directly in sam_masks_dir, not in a JPEGImages subfolder
    precomputed_sam_masks_dir = Path(args.sam_masks_dir)
    pascal_gt_dir = Path(args.pascal_gt_dir)  # GTs are usually in SegmentationClass
    target_img_shape = (args.img_size, args.img_size)

    if not precomputed_sam_masks_dir.is_dir():
        logger.error(f"SAM precomputed masks directory not found: {precomputed_sam_masks_dir}")
        return
    if not pascal_gt_dir.is_dir():
        logger.error(f"PASCAL VOC GT directory not found: {pascal_gt_dir}")
        return

    sam_mask_files = sorted(list(precomputed_sam_masks_dir.glob('*_sam_hybrid_masks.pkl')))
    if not sam_mask_files:
        logger.error(f"No SAM mask .pkl files found in {precomputed_sam_masks_dir}")
        return

    logger.info(f"Found {len(sam_mask_files)} SAM mask .pkl files to process.")

    all_max_ious_for_segments = []
    images_with_errors = 0
    total_gt_segments_evaluated = 0

    for sam_mask_file_path in tqdm(sam_mask_files, desc="Evaluating SAM Oracle Performance"):
        image_id = sam_mask_file_path.stem.replace('_sam_hybrid_masks', '')
        gt_annotation_path = pascal_gt_dir / f"{image_id}.png"

        if not gt_annotation_path.exists():
            logger.warning(f"GT annotation file missing for {image_id} ({gt_annotation_path}). Skipping.")
            images_with_errors += 1
            continue

        try:
            with open(sam_mask_file_path, 'rb') as f:
                sam_masks_for_image = pickle.load(f)
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
            # Resize GT to the same size SAM processed the image at. Use NEAREST for label maps.
            gt_full_pil_resized = gt_full_pil.resize(target_img_shape[::-1], Image.NEAREST)  # PIL resize takes (W,H)
            gt_full_np = np.array(gt_full_pil_resized)
        except Exception as e:
            logger.error(f"Error loading or resizing GT for {image_id} from {gt_annotation_path}: {e}")
            images_with_errors += 1
            continue

        unique_gt_labels = np.unique(gt_full_np)

        image_had_evaluable_segments = False
        for label_val in unique_gt_labels:
            if label_val == 0: continue  # Skip background
            if label_val == 255: continue  # Skip void/border

            # Create binary mask for this specific GT segment/object instance
            gt_mask_for_segment = (gt_full_np == label_val).astype(bool)

            if not gt_mask_for_segment.any():  # Should not happen if label_val was from unique() unless it was only border
                continue

            best_iou, _, _ = find_best_sam_subset_for_gt(
                sam_masks_for_image, gt_mask_for_segment, target_img_shape
            )
            all_max_ious_for_segments.append(best_iou)
            total_gt_segments_evaluated += 1
            image_had_evaluable_segments = True

        if not image_had_evaluable_segments and len(
                sam_mask_files) == 1:  # only log if it's a single file and no segments
            logger.info(f"Image {image_id} had no evaluable foreground segments (only background/void).")

    logger.info("\n--- SAM Oracle Mean Max IoU Summary ---")
    logger.info(f"Total SAM mask files processed (or attempted): {len(sam_mask_files)}")
    logger.info(f"Images skipped due to errors/missing files: {images_with_errors}")
    logger.info(f"Total individual ground truth segments evaluated: {total_gt_segments_evaluated}")

    if all_max_ious_for_segments:
        overall_mean_max_iou = np.mean(all_max_ious_for_segments)
        logger.info(f"\nOverall Mean of Maximum Achievable IoUs (across all GT segments): {overall_mean_max_iou:.4f}")
    else:
        logger.warning("No ground truth segments were evaluated. Cannot compute mean max IoU.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SAM Oracle Performance directly from SAM mask files and GT files.")
    parser.add_argument('--sam_masks_dir', type=str, default=r"C:\Users\Khan\PycharmProjects\FSS-Research-\pascal\pascal_sam_precomputed_masks_hybrid\JPEGImages",
                        help="Directory containing the pre-computed SAM mask .pkl files (e.g., 'C:\\Users\\Khan\\...\\pascal_sam_precomputed_masks_hybrid').")
    parser.add_argument('--pascal_gt_dir', type=str, default=r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw\SegmentationClassAug",
                        help="Path to the PASCAL VOC ground truth SegmentationClass directory (e.g., './VOCdevkit/VOC2012/SegmentationClass').")
    parser.add_argument('--img_size', type=int, default=256,
                        help="Image size (square) that SAM masks were precomputed for and GTs will be resized to.")

    args = parser.parse_args()

    start_time = time.time()
    evaluate_sam_oracle_generic(args)
    end_time = time.time()
    logger.info(f"Total evaluation time: {end_time - start_time:.2f} seconds.")