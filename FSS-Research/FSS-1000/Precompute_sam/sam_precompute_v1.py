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
DEFAULT_FSS_PATH = '/home/farahani/khan/fewshot_data/fewshot_data'
DEFAULT_SAM_CHECKPOINT = '/home/farahani/khan/sam_vit_h_4b8939.pth'
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_IMG_SIZE = 256
DEFAULT_OUTPUT_DIR = './sam_precomputed_masks_hybrid' # Directory to save masks
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
        # Use SAM default generator settings or match your RL env settings
        mask_generator = SamAutomaticMaskGenerator(
            model=sam, # Pass the model here
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=50 # Consistent with env
        )
        logger.info(f"SAM Automatic Mask Generator initialized on '{device}'.")
        return predictor, mask_generator # Return both
    except Exception as e:
        logger.error(f"Failed to initialize SAM components: {e}", exc_info=True)
        raise

def convert_predictor_mask_to_dict(mask_np, score, point_coord, img_shape):
    """Converts a mask from SamPredictor output to the dict format."""
    # mask_np should be a boolean array HxW
    H, W = img_shape[:2]
    area = np.sum(mask_np)
    if area == 0:
        return None # Skip empty masks

    # Calculate bounding box [x_min, y_min, x_max, y_max]
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not np.any(rows) or not np.any(cols):
         return None # Should not happen if area > 0, but safe check
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # SAM format seems to be [xmin, ymin, width, height]
    bbox = [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]

    mask_dict = {
        'segmentation': mask_np,
        'area': int(area),
        'bbox': bbox,
        'predicted_iou': float(score),
        'point_coords': [point_coord], # Store the prompt point [[x,y]]
        'stability_score': float(score), # Use predicted_iou as proxy
        'crop_box': [0, 0, W, H], # Indicates it's from the full image context
    }
    return mask_dict

def precompute_masks(args):
    """Generates and saves SAM masks for the FSS-1000 dataset using a hybrid approach."""
    device = torch.device(args.device)
    base_path = Path(args.fss_path)
    output_dir = Path(args.output_dir)

    if not base_path.is_dir():
        logger.error(f"FSS-1000 base path not found or not a directory: {base_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for masks: {output_dir}")

    try:
        # Get both predictor and generator
        predictor, mask_generator = initialize_sam(args.model_type, args.sam_checkpoint, device)
    except Exception as e:
        logger.error(f"Could not initialize SAM. Aborting. Error: {e}")
        return

    # Simple image transform matching the one used for SAM input in the RL env
    sam_image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
    ])

    logger.info(f"Scanning dataset at: {base_path}")
    all_classes = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    logger.info(f"Found {len(all_classes)} potential classes.")

    total_images = 0
    processed_images = 0
    processed_with_prompting = 0
    skipped_images = 0
    error_images = 0
    start_time = time.time()

    for class_name in tqdm(all_classes, desc="Processing Classes"):
        class_path = base_path / class_name
        class_output_path = output_dir / class_name
        class_output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted([f for f in class_path.glob('*.jpg')])
        if not image_files:
            logger.warning(f"No jpg images found in class: {class_name}")
            continue

        total_images += len(image_files)

        for img_path in tqdm(image_files, desc=f"Class {class_name}", leave=False):
            image_id = img_path.stem # Get filename without extension (e.g., '1')
            mask_output_file = class_output_path / f"{image_id}_sam_hybrid_masks.pkl" # Save as pickle

            if not args.overwrite and mask_output_file.exists():
                skipped_images += 1
                continue

            try:
                # Load and prepare image for SAM
                image_pil = Image.open(img_path).convert("RGB")
                image_pil_resized = sam_image_transform(image_pil)
                image_np = np.array(image_pil_resized).astype(np.uint8)

                # Ensure correct shape (H, W, 3)
                if image_np.ndim == 2:
                    logger.warning(f"Converting grayscale image to RGB: {img_path}")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                elif image_np.shape[2] == 4:
                    logger.warning(f"Converting RGBA image to RGB: {img_path}")
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

                if image_np.shape != (args.img_size, args.img_size, 3):
                     raise ValueError(f"Unexpected image shape after processing: {image_np.shape} for {img_path}")

                # --- Step 1: Automatic Mask Generation ---
                with torch.no_grad():
                    masks_initial = mask_generator.generate(image_np)

                # --- Step 2: Combine Initial Masks ---
                if not masks_initial:
                    combined_mask = np.zeros((args.img_size, args.img_size), dtype=bool)
                    logger.debug(f"No initial masks found for {img_path}")
                else:
                    # Stack boolean masks for efficient OR operation
                    mask_stack = np.stack([m['segmentation'] for m in masks_initial], axis=0)
                    combined_mask = np.logical_or.reduce(mask_stack, axis=0)

                # --- Step 3: Identify & Filter Uncovered Gaps ---
                uncovered_mask = (~combined_mask).astype(np.uint8) # Invert: 1 where not covered
                prompt_points = []
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(uncovered_mask, connectivity=8)

                # Label 0 is the background (covered area or largest component if fully uncovered)
                large_enough_gap_found = False
                for label in range(1, num_labels): # Iterate through components
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area >= args.min_gap_area:
                        large_enough_gap_found = True
                        cx, cy = centroids[label]
                        # Ensure point is integer for predictor
                        prompt_points.append([int(round(cx)), int(round(cy))])

                # --- Step 4: Targeted Prompting (if gaps found) ---
                prompted_masks_data = []
                if prompt_points:
                    logger.debug(f"Found {len(prompt_points)} gaps >= {args.min_gap_area} pixels for {img_path}. Prompting...")
                    processed_with_prompting += 1 # Count image where prompting occurred
                    try:
                        # Set image *once* for the predictor for efficiency
                        predictor.set_image(image_np)

                        for point in prompt_points:
                            input_point = np.array([point])
                            input_label = np.array([1]) # 1 indicates foreground point

                            with torch.no_grad():
                                masks_prompt, scores, _ = predictor.predict(
                                    point_coords=input_point,
                                    point_labels=input_label,
                                    multimask_output=False, # Get multiple mask proposals per point
                                )

                            # Convert prompted masks to dict format and add
                            for mask_p, score in zip(masks_prompt, scores):
                                mask_dict = convert_predictor_mask_to_dict(mask_p, score, point, image_np.shape)
                                if mask_dict:
                                    prompted_masks_data.append(mask_dict)

                        # Reset predictor if necessary (usually not needed unless changing images)
                        # predictor.reset_image()
                    except Exception as prompt_err:
                         logger.error(f"Error during prompting for {img_path} at point {point}: {prompt_err}", exc_info=False)
                         # Continue without the prompted masks for this point/image

                # --- Step 5: Combine Initial and Prompted Masks ---
                # Combine the original list and the new list
                final_masks_data = masks_initial + prompted_masks_data
                logger.debug(f"Image {img_path}: Initial masks={len(masks_initial)}, Prompted masks={len(prompted_masks_data)}, Total={len(final_masks_data)}")

                # --- Step 6: Save the combined masks ---
                if not final_masks_data:
                    logger.warning(f"No masks generated (initial or prompted) for {img_path}. Skipping save.")
                    # Decide if you want to save an empty file or just skip
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
                logger.error(f"Failed to process or save masks for {img_path}: {e}", exc_info=True) # Set exc_info=True for detailed stack trace
                error_images += 1
                # Optional: Clean up partially written file if error occurs during saving
                if mask_output_file.exists():
                    try: mask_output_file.unlink()
                    except OSError: pass


    end_time = time.time()
    logger.info("="*30 + " Pre-computation Summary (Hybrid) " + "="*30)
    logger.info(f"Total potential images found: {total_images}")
    logger.info(f"Successfully processed images: {processed_images}")
    logger.info(f"Images where prompting was attempted: {processed_with_prompting}")
    logger.info(f"Skipped (already exist): {skipped_images}")
    logger.info(f"Errors encountered: {error_images}")
    logger.info(f"Minimum gap area threshold for prompting: {args.min_gap_area} pixels")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Masks saved in: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute SAM masks for FSS-1000 dataset using automatic generation followed by targeted prompting in gaps.")
    parser.add_argument('--fss_path', type=str, default=DEFAULT_FSS_PATH, help='Path to the FSS-1000 dataset directory.')
    parser.add_argument('--sam_checkpoint', type=str, default=DEFAULT_SAM_CHECKPOINT, help='Path to the SAM model checkpoint (.pth).')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE, help='SAM model type (e.g., vit_h, vit_l, vit_b).')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help='Image size to resize to before feeding to SAM.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save the pre-computed hybrid masks.')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device to use (cuda or cpu).')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mask files if they exist.')
    parser.add_argument('--min_gap_area', type=int, default=DEFAULT_MIN_GAP_AREA, help='Minimum contiguous area (pixels) of an uncovered region to trigger point prompting.')

    args = parser.parse_args()

    # Add a check for min_gap_area
    if args.min_gap_area < 1:
        logger.warning("--min_gap_area should be >= 1. Setting to 1.")
        args.min_gap_area = 1


    precompute_masks(args)
