import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path
import random
from tqdm import tqdm
from types import SimpleNamespace
from skimage.measure import regionprops
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg'


# --- Re-used utility functions (unchanged) ---
def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool);
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum();
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union


def combine_masks(mask_list, ref_shape):
    if not mask_list: return np.zeros(ref_shape, dtype=bool)
    combined = np.zeros(ref_shape, dtype=bool)
    for mask in mask_list:
        if mask.shape != ref_shape: continue
        combined = np.logical_or(combined, mask.astype(bool))
    return combined


PASCAL_CLASS_NAMES_LIST = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# --- END of re-used utility functions ---

def load_graph_data(query_class, query_index, precomputed_graph_dir):
    graph_file_path = Path(precomputed_graph_dir) / query_class / f"{query_index}_graph.pt"
    if not graph_file_path.is_file():
        return None
    try:
        data = torch.load(graph_file_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading precomputed graph {graph_file_path}: {e}")
        return None


def visualize_graph_on_image(graph_data, image_path, output_dir):
    # This function is unchanged from the previous version.
    if graph_data is None:
        print("No graph data to visualize.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_class = graph_data.query_class;
    query_index = graph_data.query_index;
    num_nodes = graph_data.num_nodes
    node_masks_np = graph_data.node_masks_np;
    node_labels = graph_data.y.cpu().numpy()
    ref_shape = graph_data.image_size;
    gt_mask_np = graph_data.gt_mask_np

    try:
        original_image = Image.open(image_path).convert("RGB")
        image_for_plot = original_image.resize(ref_shape, Image.Resampling.BILINEAR)
    except FileNotFoundError:
        print(f"ERROR: Original image not found at {image_path}")
        image_for_plot = Image.new('RGB', ref_shape, (255, 255, 255))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_for_plot, zorder=1)

    selected_oracle_masks = [node_masks_np[i] for i, label in enumerate(node_labels) if label == 1]
    combined_oracle_mask = combine_masks(selected_oracle_masks, ref_shape=ref_shape)
    oracle_iou = calculate_iou(combined_oracle_mask, gt_mask_np.astype(bool))

    ax.set_title(f"Graph on Image: {query_class} / {query_index} (Oracle IoU: {oracle_iou:.3f})")
    ax.axis('off')

    green_overlay = np.zeros((ref_shape[0], ref_shape[1], 4), dtype=np.float32)
    green_overlay[combined_oracle_mask, :] = [0, 1, 0, 0.4]
    ax.imshow(green_overlay, zorder=2)

    G = nx.Graph();
    G.add_nodes_from(range(num_nodes))
    pos = {}
    node_centroids = []
    for i, mask_np_uint8 in enumerate(node_masks_np):
        if mask_np_uint8.sum() > 0:
            try:
                props = regionprops(mask_np_uint8)
                if props:
                    centroid_y, centroid_x = props[0].centroid
                    pos[i] = (centroid_x, centroid_y)
                    node_centroids.append(pos[i])
                else:
                    pos[i] = (random.randint(0, ref_shape[1]), random.randint(0, ref_shape[0]))
            except Exception:
                pos[i] = (random.randint(0, ref_shape[1]), random.randint(0, ref_shape[0]))
        else:
            pos[i] = (random.randint(0, ref_shape[1]), random.randint(0, ref_shape[0]))

    if len(node_centroids) < num_nodes / 2:
        spring_pos = nx.spring_layout(G, seed=42)
        pos = {node: (coord[0] * ref_shape[1], coord[1] * ref_shape[0]) for node, coord in spring_pos.items()}

    num_edges = graph_data.edge_index.shape[1] if graph_data.edge_index is not None else 0
    # print(f"  [Visualizer] Found {num_edges} edges to draw for this graph.")

    if num_edges > 0:
        edges = graph_data.edge_index.t().tolist()
        edge_collection = nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, alpha=1.0, width=1.5, edge_color='cyan')
        if edge_collection: edge_collection.set_zorder(3)

    for i in range(num_nodes):
        node_x, node_y = pos[i]
        border_color = 'green' if node_labels[i] == 1 else 'red'
        circle = plt.Circle((node_x, node_y), radius=8, color=border_color, fill=False, lw=2.5, zorder=4)
        ax.add_artist(circle)
        ax.text(node_x, node_y, str(i), color='white', ha='center', va='center',
                fontweight='bold', fontsize=8,
                bbox=dict(facecolor=border_color, alpha=0.8, boxstyle='circle,pad=0.1', lw=0), zorder=5)

    plt.tight_layout()
    output_filename = output_dir / f"graph_on_image_{query_class}_{query_index}_iou_{oracle_iou:.2f}.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')

    if args.interactive_plots and not args.save_below_iou_threshold:
        print(f"--- Displaying plot for {query_class}/{query_index}. Close window to continue. ---")
        plt.show(block=True)

    print(f"--- Plot saved to {output_filename} ---")
    plt.close(fig)


def main(args):
    graph_dir = Path(args.precomputed_graph_path)
    output_dir = Path(args.plot_output_dir)
    image_dir = Path(args.pascal_image_dir)

    for d in [graph_dir, image_dir]:
        if not d.exists():
            print(f"Error: Directory not found at {d}")
            return
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect all available graphs first for all modes ---
    print("Collecting all available graphs...")
    all_graph_files = []
    class_dirs = [d for d in graph_dir.iterdir() if d.is_dir() and d.name in PASCAL_CLASS_NAMES_LIST]
    if not class_dirs:
        print("Error: No valid class subdirectories found in the graph directory.")
        return
    for class_dir in class_dirs:
        for graph_file in class_dir.glob("*.pt"):
            all_graph_files.append((class_dir.name, graph_file.stem.replace('_graph', '')))

    if not all_graph_files:
        print("Error: No '.pt' graph files found within the class subdirectories.")
        return
    print(f"Found {len(all_graph_files)} total graphs.")

    # --- Mode Logic ---
    samples_to_plot = []
    if args.save_below_iou_threshold is not None:
        # NEW MODE: Iterate all and save if IoU is below the threshold
        print(f"\nMode: Saving all visualizations with Oracle IoU < {args.save_below_iou_threshold}")
        # Disable interactive plotting for this mode to avoid opening hundreds of windows
        args.interactive_plots = False

        for query_class, query_index in tqdm(all_graph_files, desc="Checking IoU for all graphs"):
            graph_data = load_graph_data(query_class, query_index, graph_dir)
            if not graph_data: continue

            selected_masks = [graph_data.node_masks_np[i] for i, label in enumerate(graph_data.y) if label == 1]
            combined_mask = combine_masks(selected_masks, graph_data.image_size)
            iou = calculate_iou(combined_mask, graph_data.gt_mask_np.astype(bool))

            if iou < args.save_below_iou_threshold:
                samples_to_plot.append((query_class, query_index))

        print(f"Found {len(samples_to_plot)} samples with IoU < {args.save_below_iou_threshold}. Generating plots...")

    elif args.plot_random_samples > 0:
        # RANDOM SAMPLES MODE
        num_to_plot = min(args.plot_random_samples, len(all_graph_files))
        samples_to_plot = random.sample(all_graph_files, num_to_plot)
        print(f"\nMode: Plotting {num_to_plot} random samples.")

    elif args.plot_specific_sample_class and args.plot_specific_sample_index:
        # SPECIFIC SAMPLE MODE
        samples_to_plot = [(args.plot_specific_sample_class, args.plot_specific_sample_index)]
        print(f"\nMode: Plotting specific sample: {samples_to_plot[0]}")
    else:
        print("Error: No valid plotting mode selected.")
        return

    # --- Main plotting loop ---
    if not samples_to_plot:
        print("No samples to plot based on the selected criteria.")
        return

    for query_class, query_index in tqdm(samples_to_plot, desc="Generating Plots"):
        graph_data = load_graph_data(query_class, query_index, graph_dir)
        original_image_path = image_dir / f"{query_index}.jpg"

        if graph_data:
            visualize_graph_on_image(graph_data, original_image_path, output_dir)
        else:
            print(f"Warning: Could not load graph for {query_class}/{query_index}. Skipping.")


if __name__ == "__main__":
    # --- IMPORTANT: UPDATE THESE PATHS ---
    DEFAULT_PASCAL_IMAGE_DIR = r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw\JPEGImages"
    DEFAULT_PRECOMPUTED_GRAPH_PATH = './precomputed_pascal_graphs_hybrid_v1_main'
    DEFAULT_PLOT_OUTPUT_DIR = './plot_graphs_precomputation'

    args = SimpleNamespace(
        pascal_image_dir=DEFAULT_PASCAL_IMAGE_DIR,
        precomputed_graph_path=DEFAULT_PRECOMPUTED_GRAPH_PATH,
        plot_output_dir=DEFAULT_PLOT_OUTPUT_DIR,

        # --- CHOOSE PLOTTING BEHAVIOR ---
        interactive_plots=True,

        # --- CHOOSE SAMPLES TO PLOT (Only one mode will be active) ---

        # MODE 1: Save visualizations for all samples with IoU BELOW a threshold.
        # Set to a float value (e.g., 0.4) to activate. Set to None to disable.
        save_below_iou_threshold=0.4,

        # MODE 2: Plot N random samples. Active if `save_below_iou_threshold` is None.
        plot_random_samples=0,

        # MODE 3: Plot one specific sample. Active if both above are disabled/zero.
        plot_specific_sample_class="chair",
        plot_specific_sample_index="2008_007692",
    )

    # Logic to select the mode of operation
    if args.save_below_iou_threshold is not None:
        args.plot_random_samples = 0
        args.plot_specific_sample_class = None
        args.plot_specific_sample_index = None
    elif args.plot_random_samples > 0:
        args.plot_specific_sample_class = None
        args.plot_specific_sample_index = None

    main(args)