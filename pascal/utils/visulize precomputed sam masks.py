import os
import sys
import argparse
import pickle
from pathlib import Path
import logging

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm # For progress bar

# --- Configuration (match these with your precomputation script if needed) ---
DEFAULT_PASCAL_PATH = r'C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw'  # Example path, adjust as needed
DEFAULT_MASKS_PATH = r'C:\Users\Khan\PycharmProjects\FSS-Research-\pascal\pascal_sam_precomputed_masks_hybrid' # Path to your precomputed masks
DEFAULT_VIZ_OUTPUT_PATH = './pascal_sam_visualizations' # Default path to save visualizations
DEFAULT_IMAGE_FOLDER = 'JPEGImages' # Standard PASCAL image subfolder
MASK_FILENAME_SUFFIX = '_sam_hybrid_masks.pkl' # Suffix used for mask files

# --- Setup Logging (for general script info, not the percentage summary) ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_image_and_masks(pascal_root_path, masks_root_path, image_id):
    """Loads the original image and its precomputed SAM masks."""
    image_file_path = pascal_root_path / DEFAULT_IMAGE_FOLDER / f"{image_id}.jpg"
    mask_file_path = masks_root_path / DEFAULT_IMAGE_FOLDER / f"{image_id}{MASK_FILENAME_SUFFIX}"

    img_pil_resized = None
    masks_data_loaded = None

    if not image_file_path.exists():
        logger.error(f"Original image file not found: {image_file_path} for image_id {image_id}")
        return None, None # Indicate image error
    if not mask_file_path.exists():
        # This case should ideally not be hit if we are iterating over existing mask files
        logger.error(f"Mask file not found: {mask_file_path} for image_id {image_id}")
        return None, None # Indicate mask error (though unlikely here)

    try:
        img_pil = Image.open(image_file_path).convert("RGB")

        with open(mask_file_path, 'rb') as f:
            masks_data_loaded = pickle.load(f)

        if not masks_data_loaded:
            logger.info(f"No masks found in {mask_file_path} for image {image_id}. Image will be shown without masks.")
            masks_data_loaded = [] # Ensure it's an empty list for consistent handling

        # Resize original image to match the dimensions of the masks for consistent display
        # This assumes all masks in a file have the same dimensions from precomputation
        if masks_data_loaded: # If there are masks, use their shape
            mask_h, mask_w = masks_data_loaded[0]['segmentation'].shape
            img_pil_resized = img_pil.resize((mask_w, mask_h), Image.Resampling.LANCZOS)
        else: # If no masks, use original image size (or a default if precomputation size is known)
            # For consistency, it's better if precomputation always saved something that implies a size
            # If your precomputation script saves empty lists for images with no masks,
            # we need a way to know the target image_size SAM used.
            # Assuming args.img_size from precomputation script if masks are empty.
            # You might need to pass this img_size if it's not inferable.
            # For now, let's assume if masks_data_loaded is empty, the precomputed img_size was standard.
            # A more robust way is to store img_size in the pickle file or infer from args.
            logger.warning(f"No masks for {image_id}, resizing image to a default or inferred size. "
                           f"Ensure this matches precomputation settings if masks were empty.")
            # Fallback: if precomp script had fixed size, use it. Here, let's assume it was 256x256
            # This is a potential point of mismatch if your empty mask files don't imply a size.
            # Ideally, even an empty mask list would be associated with the target processing size.
            try:
                # If we could access the precomputation args.img_size here, it would be best.
                # For now, a common default.
                precomputed_img_size = 256
                img_pil_resized = img_pil.resize((precomputed_img_size, precomputed_img_size), Image.Resampling.LANCZOS)
            except Exception as resize_err:
                logger.error(f"Could not resize image {image_id} when masks were empty: {resize_err}")
                return None, masks_data_loaded # Return masks if loaded, but signal image issue


        return img_pil_resized, masks_data_loaded
    except FileNotFoundError as fnf_err: # Should be caught by exists() checks, but for safety
        logger.error(f"File not found during loading for {image_id}: {fnf_err}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading image or masks for {image_id}: {e}", exc_info=True)
        return None, None


def generate_distinct_colors(num_colors):
    """Generates a list of visually distinct colors."""
    if num_colors <= 0: return []
    if num_colors <= 10:
        cmap = plt.cm.get_cmap('tab10', num_colors)
    elif num_colors <= 20:
        cmap = plt.cm.get_cmap('tab20', num_colors)
    else:
        cmap = plt.cm.get_cmap('gist_rainbow', num_colors)
    colors = [cmap(i)[:3] for i in range(num_colors)]
    return colors

def plot_and_save_masks_on_image(image_pil, masks_data, title, save_path):
    """Displays masks overlaid on the image and saves the plot."""
    plt.close('all') # Ensure clean slate for new plot
    fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted for better control

    if image_pil is None:
        logger.warning(f"Cannot plot masks on image for '{title}', image_pil is None.")
        ax.text(0.5, 0.5, "Original Image Not Loaded", ha='center', va='center')
        ax.set_title(f"{title} (Image Error)")
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved placeholder for overlay image to {save_path}")
        plt.close(fig)
        return

    img_np = np.array(image_pil)
    overlay = img_np.copy()
    ax.imshow(img_np) # Show original image as background

    if not masks_data:
        logger.info(f"No masks to plot on image for '{title}'.")
        ax.set_title(f"{title} (No masks found)")
    else:
        num_masks = len(masks_data)
        colors = generate_distinct_colors(num_masks)
        # Import cv2 here locally if not already imported globally and addWeighted is used
        try:
            import cv2
            use_cv2_blend = True
        except ImportError:
            logger.warning("cv2 not found, using matplotlib alpha blending for overlays (might be slower/different).")
            use_cv2_blend = False

        for i, mask_info in enumerate(masks_data):
            mask = mask_info['segmentation']
            color_np_0_1 = np.array(colors[i % len(colors)]) # Matplotlib colors (0-1)

            if use_cv2_blend:
                color_np_0_255 = (color_np_0_1 * 255).astype(np.uint8)
                colored_mask_layer = np.zeros_like(overlay, dtype=np.uint8)
                colored_mask_layer[mask] = color_np_0_255
                alpha = 0.5
                overlay[mask] = cv2.addWeighted(overlay[mask], 1 - alpha, colored_mask_layer[mask], alpha, 0)
            else: # Matplotlib blending
                # Create an RGBA layer for each mask
                mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
                mask_rgba[mask, :3] = color_np_0_1
                mask_rgba[mask, 3] = 0.5 # Alpha for this mask layer
                ax.imshow(mask_rgba) # Overlay this mask layer

        if use_cv2_blend: # If cv2 was used, update the main image display
            ax.clear() # Clear previous imshow(img_np)
            ax.imshow(overlay)
        ax.set_title(title)

    ax.axis('off')
    if save_path:
        plt.savefig(save_path)
        logger.debug(f"Saved overlay image to {save_path}")
    plt.close(fig)


def plot_and_save_separate_masks(masks_data, image_shape, title, save_path):
    """Displays each mask separately and saves the plot."""
    plt.close('all') # Ensure clean slate

    if not masks_data:
        logger.info(f"No separate masks to plot for '{title}'.")
        # Optionally save a placeholder image or skip
        fig, ax = plt.subplots(figsize=(5,2))
        ax.text(0.5, 0.5, "No Masks to Display", ha='center', va='center')
        ax.set_title(title + " (No Masks)")
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)
        return

    num_masks = len(masks_data)
    colors = generate_distinct_colors(num_masks)
    img_h, img_w = image_shape[:2]

    cols = int(np.ceil(np.sqrt(num_masks)))
    rows = int(np.ceil(num_masks / cols)) if cols > 0 else 1
    if num_masks == 0: cols = rows = 1 # Handle case of zero masks if it slips through

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False) # squeeze=False ensures axes is 2D
    axes_flat = axes.flatten()

    for i, mask_info in enumerate(masks_data):
        mask = mask_info['segmentation']
        color_np = np.array(colors[i % len(colors)])

        mask_display = np.zeros((img_h, img_w, 4), dtype=float)
        mask_display[mask, :3] = color_np
        mask_display[mask, 3] = 1.0

        axes_flat[i].imshow(mask_display)
        axes_flat[i].set_title(f"Mask {i+1}\nArea: {mask_info.get('area', 'N/A')}", fontsize=8)
        axes_flat[i].axis('off')

    for j in range(num_masks, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    if save_path:
        plt.savefig(save_path)
        logger.debug(f"Saved separate masks image to {save_path}")
    plt.close(fig)


def get_mask_percentages_text(masks_data, image_shape, image_id):
    """Calculates and returns a string of the percentage of the image covered by each mask."""
    output_lines = []
    if image_shape is None or len(image_shape) < 2:
        output_lines.append(f"  Cannot calculate percentages for {image_id}: Invalid image_shape.")
        return "\n".join(output_lines)

    img_h, img_w = image_shape[:2]
    total_pixels = img_h * img_w

    output_lines.append(f"--- Mask Coverage for Image: {image_id} (Dimensions: {img_w}x{img_h}, Total Pixels: {total_pixels}) ---")

    if not masks_data:
        output_lines.append("  No masks found for this image.")
        return "\n".join(output_lines)

    combined_coverage_mask = np.zeros((img_h, img_w), dtype=bool)

    for i, mask_info in enumerate(masks_data):
        mask_segmentation = mask_info['segmentation']
        # Ensure mask segmentation has the same shape as the image it's being compared against
        if mask_segmentation.shape != (img_h, img_w):
            output_lines.append(f"    Mask {i+1}: Shape mismatch! Mask is {mask_segmentation.shape}, image is {(img_h, img_w)}. Skipping percentage for this mask.")
            continue

        mask_area = np.sum(mask_segmentation)

        if total_pixels > 0:
            percentage = (mask_area / total_pixels) * 100
            output_lines.append(f"  Mask {i+1}: Area = {mask_area} pixels, Coverage = {percentage:.2f}%")
        else:
            output_lines.append(f"  Mask {i+1}: Area = {mask_area} pixels (Cannot calculate percentage, total_pixels is 0)")
        combined_coverage_mask = np.logical_or(combined_coverage_mask, mask_segmentation)

    total_covered_area = np.sum(combined_coverage_mask)
    if total_pixels > 0:
        overall_percentage = (total_covered_area / total_pixels) * 100
        output_lines.append(f"  Total unique area covered by all masks: {total_covered_area} pixels ({overall_percentage:.2f}%)")
    else:
        output_lines.append(f"  Total unique area covered by all masks: {total_covered_area} pixels (Cannot calculate overall percentage)")
    output_lines.append("--------------------------------------------------")
    return "\n".join(output_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, visualize, and save precomputed SAM masks for PASCAL VOC images.")
    parser.add_argument('--pascal_path', type=str, default=DEFAULT_PASCAL_PATH,
                        help='Path to the PASCAL VOC dataset root directory (e.g., ./VOCdevkit/VOC2012).')
    parser.add_argument('--masks_path', type=str, default=DEFAULT_MASKS_PATH,
                        help='Path to the root directory containing precomputed SAM masks (expects a JPEGImages subfolder).')
    parser.add_argument('--output_viz_dir', type=str, default=DEFAULT_VIZ_OUTPUT_PATH,
                        help='Directory to save the visualization images and summary.')
    # Removed --image_id as we process all found masks

    args = parser.parse_args()

    pascal_root_path = Path(args.pascal_path)
    masks_root_path = Path(args.masks_path)
    viz_output_root_path = Path(args.output_viz_dir)

    # Path where mask .pkl files are stored
    actual_masks_folder_path = masks_root_path / DEFAULT_IMAGE_FOLDER
    # Path where visualization PNGs will be stored
    viz_output_images_subfolder_path = viz_output_root_path / DEFAULT_IMAGE_FOLDER

    viz_output_images_subfolder_path.mkdir(parents=True, exist_ok=True)
    summary_log_file_path = viz_output_root_path / "mask_coverage_summary.txt"

    if not actual_masks_folder_path.is_dir():
        logger.error(f"Masks folder not found: {actual_masks_folder_path}")
        sys.exit(1)

    # Find all precomputed mask files
    mask_files = sorted(list(actual_masks_folder_path.glob(f"*{MASK_FILENAME_SUFFIX}")))

    if not mask_files:
        logger.info(f"No precomputed mask files found in {actual_masks_folder_path} with suffix '{MASK_FILENAME_SUFFIX}'")
        sys.exit(0)

    logger.info(f"Found {len(mask_files)} precomputed mask files. Processing and saving visualizations to {viz_output_root_path}...")

    with open(summary_log_file_path, "w") as summary_f:
        summary_f.write("--- SAM Mask Visualization & Coverage Summary ---\n\n")

        for mask_file_path in tqdm(mask_files, desc="Visualizing Masks"):
            image_id = mask_file_path.name.replace(MASK_FILENAME_SUFFIX, "")
            logger.info(f"Processing Image ID: {image_id}")
            summary_f.write(f"Image ID: {image_id}\n")

            try:
                # Load image and its masks
                image_pil_resized, masks_data = load_image_and_masks(
                    pascal_root_path,
                    masks_root_path, # Pass the root of masks dir
                    image_id
                )

                # Determine save paths for visualizations
                overlay_save_path = viz_output_images_subfolder_path / f"{image_id}_overlay.png"
                separate_save_path = viz_output_images_subfolder_path / f"{image_id}_separate.png"

                current_image_shape = None
                if image_pil_resized:
                    current_image_shape = np.array(image_pil_resized).shape

                # 1. Plot and Save masks overlaid on the image
                plot_and_save_masks_on_image(
                    image_pil_resized,
                    masks_data,
                    title=f"Masks on Image: {image_id}",
                    save_path=overlay_save_path
                )
                if image_pil_resized: # Only log save path if image was loaded
                     summary_f.write(f"  Saved overlay to: {overlay_save_path.relative_to(viz_output_root_path)}\n")
                else:
                    summary_f.write(f"  Overlay for {image_id} not saved (original image missing or error).\n")


                # 2. Plot and Save masks separately
                # Ensure we have a valid shape for separate masks, even if original image load failed
                # but masks were loaded. The mask shape itself is the reference.
                shape_for_separate_masks = current_image_shape
                if not shape_for_separate_masks and masks_data:
                    shape_for_separate_masks = masks_data[0]['segmentation'].shape + (3,) # H, W, C

                if shape_for_separate_masks:
                    plot_and_save_separate_masks(
                        masks_data,
                        shape_for_separate_masks,
                        title=f"Separate Masks for: {image_id}",
                        save_path=separate_save_path
                    )
                    summary_f.write(f"  Saved separate masks to: {separate_save_path.relative_to(viz_output_root_path)}\n")
                else:
                    summary_f.write(f"  Separate masks for {image_id} not saved (image shape undetermined).\n")


                # 3. Get percentages text and write to summary file
                if current_image_shape : # Only if we have a valid image shape for reference
                    percentages_text = get_mask_percentages_text(masks_data, current_image_shape, image_id)
                    summary_f.write(percentages_text + "\n")
                else:
                    summary_f.write(f"  Could not calculate percentages for {image_id} (image not loaded or shape error).\n")


            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {image_id}: {e}", exc_info=True)
                summary_f.write(f"  !! UNEXPECTED ERROR processing {image_id}: {e}\n\n")
            summary_f.write("\n") # Add a newline for separation in the summary file

    logger.info(f"Processing complete. Visualizations and summary saved in: {viz_output_root_path}")
    logger.info(f"Summary log: {summary_log_file_path}")