import os
import sys
import argparse
import logging
import time
import pickle # Using pickle to save the list of mask dictionaries
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2 # Used for connected components and potential color conversions
from tqdm import tqdm
from torchvision import transforms

# --- SAM Import ---
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
except ImportError:
    print("ERROR: 'segment_anything' library not found.")
    print("Please install it using: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Configuration Defaults (Adapt from your main script) ---
# MODIFIED: Default path for PASCAL VOC dataset
DEFAULT_PASCAL_PATH = r'../data/pascal/raw'  # Example path, adjust as needed
DEFAULT_SAM_CHECKPOINT = r'C:\Users\Khan\PycharmProjects\FSS-Research-\model\sam_vit_h_4b8939.pth' # Keep your checkpoint path
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_IMG_SIZE = 512
# MODIFIED: Output directory name for clarity
DEFAULT_OUTPUT_DIR = 'pascal_sam_precomputed_masks_hybrid'  # Directory to save masks
DEFAULT_MIN_GAP_AREA = 2000 # Minimum number of contiguous uncovered pixels to prompt

# --- SAM Initialization Function (Modified) ---
def initialize_sam(model_type, checkpoint_path, device):
    """Initializes SAM model, predictor, and automatic mask generator."""
    logger.info(f"Initializing SAM: Type='{model_type}', Checkpoint='{checkpoint_path}'")
    if not os.path.exists(checkpoint_path):
        logger.error(f"SAM checkpoint MISSING: {checkpoint_path}")
        raise FileNotFoundError(f"SAM checkpoint needed: {checkpoint_path}")
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        sam.eval()

        # Initialize Predictor
        predictor = SamPredictor(sam)
        logger.info(f"SAM Predictor initialized on '{device}'.")

        # Initialize Automatic Mask Generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=50
        )
        logger.info(f"SAM Automatic Mask Generator initialized on '{device}'.")
        return predictor, mask_generator
    except Exception as e:
        logger.error(f"Failed to initialize SAM components: {e}", exc_info=True)
        raise

def convert_predictor_mask_to_dict(mask_np, score, point_coord, img_shape):
    """Converts a mask from SamPredictor output to the dict format."""
    H, W = img_shape[:2]
    area = np.sum(mask_np)
    if area == 0:
        return None

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not np.any(rows) or not np.any(cols):
         return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    bbox = [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]

    mask_dict = {
        'segmentation': mask_np,
        'area': int(area),
        'bbox': bbox,
        'predicted_iou': float(score),
        'point_coords': [point_coord],
        'stability_score': float(score),
        'crop_box': [0, 0, W, H],
    }
    return mask_dict

def precompute_masks(args):
    """
    Generates and saves SAM masks for the PASCAL VOC dataset using a hybrid approach.
    Assumes images are in a 'JPEGImages' subdirectory of the main PASCAL path.
    """
    device = torch.device(args.device)
    # MODIFIED: Path handling for PASCAL VOC
    pascal_root_path = Path(args.pascal_path)
    image_folder_name = 'JPEGImages' # Standard PASCAL VOC image folder
    image_base_path = pascal_root_path / image_folder_name
    output_dir = Path(args.output_dir)

    if not image_base_path.is_dir():
        # MODIFIED: Error message for PASCAL
        logger.error(f"PASCAL VOC image directory not found: {image_base_path}")
        logger.error(f"Please ensure '{args.pascal_path}' contains a subfolder named '{image_folder_name}' with the images.")
        return

    # MODIFIED: Output directory structure for PASCAL. Masks will be saved under output_dir/JPEGImages/
    # This mimics the input structure and is similar to how FSS-1000 was handled (output_dir/class_name/)
    actual_output_path = output_dir / image_folder_name
    actual_output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for masks: {actual_output_path}")


    try:
        predictor, mask_generator = initialize_sam(args.model_type, args.sam_checkpoint, device)
    except Exception as e:
        logger.error(f"Could not initialize SAM. Aborting. Error: {e}")
        return

    sam_image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
    ])

    # MODIFIED: Simplified image collection for PASCAL
    logger.info(f"Scanning dataset at: {image_base_path}")
    image_files = sorted([f for f in image_base_path.glob('*.jpg')])
    if not image_files:
        logger.warning(f"No jpg images found in: {image_base_path}")
        return

    logger.info(f"Found {len(image_files)} images in {image_folder_name}.")

    total_images = len(image_files)
    processed_images = 0
    processed_with_prompting = 0
    skipped_images = 0
    error_images = 0
    start_time = time.time()

    # MODIFIED: Removed the outer "class_name" loop. We iterate directly over images.
    # The 'class_name' concept is now implicitly 'JPEGImages' for path construction.
    for img_path in tqdm(image_files, desc=f"Processing PASCAL Images ({image_folder_name})"):
        image_id = img_path.stem
        # MODIFIED: Output file path reflects the single image folder
        mask_output_file = actual_output_path / f"{image_id}_sam_hybrid_masks.pkl"

        if not args.overwrite and mask_output_file.exists():
            skipped_images += 1
            continue

        try:
            image_pil = Image.open(img_path).convert("RGB")
            image_pil_resized = sam_image_transform(image_pil)
            image_np = np.array(image_pil_resized).astype(np.uint8)

            if image_np.ndim == 2:
                logger.warning(f"Converting grayscale image to RGB: {img_path}")
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                logger.warning(f"Converting RGBA image to RGB: {img_path}")
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            if image_np.shape != (args.img_size, args.img_size, 3):
                 raise ValueError(f"Unexpected image shape after processing: {image_np.shape} for {img_path}")

            with torch.no_grad():
                masks_initial = mask_generator.generate(image_np)

            if not masks_initial:
                combined_mask = np.zeros((args.img_size, args.img_size), dtype=bool)
                logger.debug(f"No initial masks found for {img_path}")
            else:
                mask_stack = np.stack([m['segmentation'] for m in masks_initial], axis=0)
                combined_mask = np.logical_or.reduce(mask_stack, axis=0)

            uncovered_mask = (~combined_mask).astype(np.uint8)
            prompt_points = []
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(uncovered_mask, connectivity=8)

            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= args.min_gap_area:
                    cx, cy = centroids[label]
                    prompt_points.append([int(round(cx)), int(round(cy))])

            prompted_masks_data = []
            if prompt_points:
                logger.debug(f"Found {len(prompt_points)} gaps >= {args.min_gap_area} pixels for {img_path}. Prompting...")
                processed_with_prompting += 1
                try:
                    predictor.set_image(image_np)
                    for point in prompt_points:
                        input_point = np.array([point])
                        input_label = np.array([1])
                        with torch.no_grad():
                            masks_prompt, scores, _ = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=False,
                            )
                        for mask_p, score in zip(masks_prompt, scores):
                            mask_dict = convert_predictor_mask_to_dict(mask_p, score, point, image_np.shape)
                            if mask_dict:
                                prompted_masks_data.append(mask_dict)
                except Exception as prompt_err:
                     logger.error(f"Error during prompting for {img_path} at point {point}: {prompt_err}", exc_info=False)

            final_masks_data = masks_initial + prompted_masks_data
            logger.debug(f"Image {img_path}: Initial masks={len(masks_initial)}, Prompted masks={len(prompted_masks_data)}, Total={len(final_masks_data)}")

            if not final_masks_data:
                logger.warning(f"No masks generated (initial or prompted) for {img_path}. Skipping save.")
            else:
                with open(mask_output_file, 'wb') as f:
                    pickle.dump(final_masks_data, f)
            processed_images += 1

        except FileNotFoundError:
             logger.error(f"Image file not found (should not happen in glob): {img_path}")
             error_images += 1
        except ValueError as ve:
             logger.error(f"ValueError processing {img_path}: {ve}")
             error_images += 1
        except Exception as e:
            logger.error(f"Failed to process or save masks for {img_path}: {e}", exc_info=True)
            error_images += 1
            if mask_output_file.exists():
                try: mask_output_file.unlink()
                except OSError: pass

    end_time = time.time()
    logger.info("="*30 + " Pre-computation Summary (Hybrid PASCAL) " + "="*30) # MODIFIED: Title
    logger.info(f"Total images found in '{image_folder_name}': {total_images}")
    logger.info(f"Successfully processed images: {processed_images}")
    logger.info(f"Images where prompting was attempted: {processed_with_prompting}")
    logger.info(f"Skipped (already exist): {skipped_images}")
    logger.info(f"Errors encountered: {error_images}")
    logger.info(f"Minimum gap area threshold for prompting: {args.min_gap_area} pixels")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Masks saved in: {actual_output_path}") # MODIFIED: Report actual path
    logger.info("="*80)


if __name__ == "__main__":
    # MODIFIED: Parser description and argument name/default for PASCAL
    parser = argparse.ArgumentParser(description="Pre-compute SAM masks for PASCAL VOC dataset using automatic generation followed by targeted prompting in gaps.")
    parser.add_argument('--pascal_path', type=str, default=DEFAULT_PASCAL_PATH, help='Path to the PASCAL VOC dataset root directory (e.g., ./VOCdevkit/VOC2012).')
    parser.add_argument('--sam_checkpoint', type=str, default=DEFAULT_SAM_CHECKPOINT, help='Path to the SAM model checkpoint (.pth).')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, help='SAM model type (e.g., vit_h, vit_l, vit_b).')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help='Image size to resize to before feeding to SAM.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save the pre-computed hybrid masks.')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (cuda or cpu).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mask files if they exist.')
    parser.add_argument('--min_gap_area', type=int, default=DEFAULT_MIN_GAP_AREA, help='Minimum contiguous area (pixels) of an uncovered region to trigger point prompting.')

    args = parser.parse_args()

    if args.min_gap_area < 1:
        logger.warning("--min_gap_area should be >= 1. Setting to 1.")
        args.min_gap_area = 1

    precompute_masks(args)