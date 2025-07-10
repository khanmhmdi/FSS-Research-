import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, SequentialSampler, RandomSampler

# --- PyTorch Geometric Imports ---
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    # UPDATED: We will use GATConv for the new model
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
except ImportError as e:
    print(f"ERROR: PyTorch Geometric or related libraries not found. {e}")
    print("Please install it following the instructions at:")
    print("https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# --- Standard Library Imports ---
import numpy as np
import cv2
from PIL import Image
import os
import random
import time
from tqdm import tqdm
from types import SimpleNamespace
from skimage.measure import regionprops
import gc
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for servers

# --- PASCAL-5i Dataset (Provided by User) ---
import torch.nn.functional as F  # Required by DatasetPASCAL


# =========================================================================================
# NOTE: The DatasetPASCAL and PASCALAdapterDataset classes remain UNCHANGED.
# The changes begin after the adapter dataset definition.
# =========================================================================================

class DatasetPASCAL(Dataset):  # User-provided DatasetPASCAL
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'folds'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.img_path = os.path.join(datapath, 'JPEGImages')
        self.ann_path = os.path.join(datapath, 'SegmentationClassAug')
        self.transform = transform
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata(datapath)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name,
                                                                                               support_names)
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:],
                                        mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        support_imgs_transformed = [self.transform(s_img) for s_img in support_imgs]
        support_imgs = torch.stack(support_imgs_transformed)
        support_masks = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_mask, _ = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
        support_masks = torch.stack(support_masks)
        batch = {'query_img': query_img, 'query_mask': query_mask, 'query_name': query_name,
                 'org_query_imsize': org_qry_imsize, 'support_imgs': support_imgs, 'support_masks': support_masks,
                 'support_names': support_names, 'class_id': torch.tensor(class_sample)}
        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_trn if self.split == 'trn' else class_ids_val

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1
        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]
        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        return torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))

    def read_img(self, img_name):
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg').convert('RGB')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]
        available_supports = [name for name in self.img_metadata_classwise[class_sample] if name != query_name]
        if len(available_supports) < self.shot:
            support_names = np.random.choice(available_supports, self.shot, replace=True) if available_supports else []
        else:
            support_names = np.random.choice(available_supports, self.shot, replace=False)
        return query_name, list(support_names), class_sample

    def build_img_metadata(self, datapath):
        def read_metadata(split, fold_id):
            split_file_dir = os.path.join(datapath, "splits", "folds")
            fold_n_metadata_path = os.path.join(split_file_dir, '%s' % split, 'fold%d.txt' % fold_id)
            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            return [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata if data]

        img_metadata = []
        if self.split == 'trn':
            for fold_id in range(self.nfolds):
                if fold_id != self.fold:
                    img_metadata += read_metadata(self.split, fold_id)
        else:
            img_metadata = read_metadata(self.split, self.fold)
        logger.info(
            'PASCAL Dataset: Total (%s) images for fold %d are : %d' % (self.split, self.fold, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {cid: [] for cid in range(self.nclass)}
        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class].append(img_name)
        return img_metadata_classwise


class PASCALAdapterDataset(Dataset):
    def __init__(self, pascal_dataset_instance, target_image_size_int, class_names_list):
        self.pascal_dataset = pascal_dataset_instance
        self.target_image_size_tuple = (target_image_size_int, target_image_size_int)
        self.class_names_list = class_names_list
        logger.info(
            f"PASCALAdapterDataset initialized. Wraps a PASCAL dataset with {len(self.pascal_dataset)} base items.")

    def __len__(self):
        return len(self.pascal_dataset)

    def __getitem__(self, idx):
        try:
            raw_item = self.pascal_dataset[idx]
            support_set_out = [{'image': raw_item['support_imgs'][i], 'mask': raw_item['support_masks'][i]} for i in
                               range(raw_item['support_imgs'].shape[0])]
            q_mask_np_uint8 = raw_item['query_mask'].cpu().numpy().astype(np.uint8)
            class_id_int = raw_item['class_id'].item()
            query_class_str = self.class_names_list[class_id_int]
            query_set_out = {'image_resnet': raw_item['query_img'], 'gt_mask_np': q_mask_np_uint8,
                             'class': query_class_str,
                             'query_index': raw_item['query_name']}
            return support_set_out, query_set_out
        except Exception as e:
            logger.error(f"Error in PASCALAdapterDataset for index {idx}: {e}", exc_info=True)
            return None


# --- Configuration and Utility Functions (UNCHANGED) ---
PASCAL_CLASS_NAMES_LIST = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                           "train", "tvmonitor"]
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

DEFAULT_PASCAL_DATAPATH = r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw"
DEFAULT_PRECOMPUTED_MASK_PATH = r'C:\Users\Khan\Desktop\New folder (2)\sam_precomputed_masks_hybrid\pascal_voc2012_sam_masks'
DEFAULT_PRECOMPUTED_NODE_PATH = './precomputed_pascal_nodes'  # CHANGED from graph path to node path
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_K_SHOT = 1
DEFAULT_IMG_SIZE = 256
DEFAULT_FEATURE_SIZE = 2048
DEFAULT_MAX_MASKS_GRAPH = 30


# ... other utils like calculate_iou, combine_masks etc. remain unchanged ...
def _run_greedy_oracle_for_labels(candidate_masks_bool_list, query_gt_mask_bool, ref_shape):
    num_candidates = len(candidate_masks_bool_list)
    selected_indices = set()
    final_iou = 0.0
    if num_candidates == 0:
        return selected_indices, final_iou
    candidate_ious_and_indices_for_add = []
    for i, mask_np_b in enumerate(candidate_masks_bool_list):
        try:
            iou_val = calculate_iou(mask_np_b, query_gt_mask_bool)
            candidate_ious_and_indices_for_add.append((iou_val, i))
        except Exception:
            candidate_ious_and_indices_for_add.append((-1.0, i))
    candidate_ious_and_indices_for_add.sort(key=lambda x: x[0], reverse=True)
    sorted_candidate_indices_for_add = [idx for _, idx in candidate_ious_and_indices_for_add]
    current_combined_mask = np.zeros(ref_shape, dtype=bool)
    try:
        current_best_iou = calculate_iou(current_combined_mask, query_gt_mask_bool)
    except Exception:
        current_best_iou = 0.0
    while True:
        best_iou_this_iteration = current_best_iou
        best_action_type = None
        best_mask_idx_for_action = -1
        for candidate_idx in sorted_candidate_indices_for_add:
            if candidate_idx not in selected_indices:
                potential_combined_mask_add = np.logical_or(current_combined_mask,
                                                            candidate_masks_bool_list[candidate_idx])
                try:
                    potential_iou_add = calculate_iou(potential_combined_mask_add, query_gt_mask_bool)
                except Exception:
                    potential_iou_add = 0.0
                if potential_iou_add > best_iou_this_iteration:
                    best_iou_this_iteration = potential_iou_add
                    best_action_type = 'add'
                    best_mask_idx_for_action = candidate_idx
        indices_to_check_for_removal = list(selected_indices)
        for selected_idx_to_remove in indices_to_check_for_removal:
            temp_selected_indices_after_remove = selected_indices - {selected_idx_to_remove}
            masks_to_combine_after_remove = [candidate_masks_bool_list[idx] for idx in
                                             temp_selected_indices_after_remove]
            potential_combined_mask_remove = combine_masks(masks_to_combine_after_remove, ref_shape)
            try:
                potential_iou_remove = calculate_iou(potential_combined_mask_remove, query_gt_mask_bool)
            except Exception:
                potential_iou_remove = 0.0
            if potential_iou_remove > best_iou_this_iteration:
                best_iou_this_iteration = potential_iou_remove
                best_action_type = 'remove'
                best_mask_idx_for_action = selected_idx_to_remove
        if best_action_type is not None and best_iou_this_iteration > current_best_iou:
            if best_action_type == 'add':
                selected_indices.add(best_mask_idx_for_action)
                current_combined_mask = np.logical_or(current_combined_mask,
                                                      candidate_masks_bool_list[best_mask_idx_for_action])
            elif best_action_type == 'remove':
                selected_indices.remove(best_mask_idx_for_action)
                masks_to_recombine = [candidate_masks_bool_list[idx] for idx in selected_indices]
                current_combined_mask = combine_masks(masks_to_recombine, ref_shape)
            current_best_iou = best_iou_this_iteration
            final_iou = current_best_iou
        else:
            break
    if not selected_indices and num_candidates > 0:
        try:
            final_iou = calculate_iou(np.zeros(ref_shape, dtype=bool), query_gt_mask_bool)
        except Exception:
            final_iou = 0.0
    return selected_indices, final_iou


def calculate_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def combine_masks(mask_list, ref_shape):
    if not mask_list:
        return np.zeros(ref_shape, dtype=bool)
    combined = np.zeros(ref_shape, dtype=bool)
    for mask in mask_list:
        if mask.shape != ref_shape:  # Ensure mask is correct shape before combining
            continue  # Skipping mask if shape is wrong.
        combined = np.logical_or(combined, mask.astype(bool))
    return combined


def get_geometric_features(mask_np):
    mask_np_bool = mask_np.astype(bool)
    if mask_np_bool.sum() == 0:
        return np.zeros(4, dtype=np.float32)
    try:
        mask_np_uint8 = (mask_np_bool).astype(np.uint8)
        props = regionprops(mask_np_uint8)
        if not props:
            return np.zeros(4, dtype=np.float32)
        props = props[0]
        h, w = mask_np_uint8.shape
        if h == 0 or w == 0: return np.zeros(4, dtype=np.float32)
        norm_area = props.area / (h * w) if (h * w) > 0 else 0
        center_y, center_x = props.centroid
        norm_center_y = center_y / h if h > 0 else 0.5
        norm_center_x = center_x / w if w > 0 else 0.5
        minr, minc, maxr, maxc = props.bbox
        bbox_h = float(maxr - minr)
        bbox_w = float(maxc - minc)
        aspect_ratio = bbox_h / bbox_w if bbox_w > 1e-6 else bbox_h / 1e-6
        aspect_ratio = min(max(aspect_ratio, 0.1), 10.0)  # Clip aspect ratio
        features = np.array([norm_area, norm_center_y, norm_center_x, aspect_ratio], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=0.1)  # Handle NaN/Inf
        return features
    except Exception as e:
        return np.zeros(4, dtype=np.float32)


class FeatureExtractor(nn.Module):
    def __init__(self, device=DEFAULT_DEVICE):
        super().__init__()
        logger.info("Initializing ResNet-50 feature extractor...")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        self.resnet_layer4 = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_layer4.eval()
        self.device = device
        self.resnet_layer4.to(device)
        self.feature_dim = 2048
        logger.info(f"ResNet-50 layer4 loaded successfully on device '{device}'. Feature dim: {self.feature_dim}")

    @torch.no_grad()
    def get_feature_map(self, image_tensor_batch):
        if image_tensor_batch.dim() == 3:
            image_tensor_batch = image_tensor_batch.unsqueeze(0)
        return self.resnet_layer4(image_tensor_batch.to(self.device))

    @torch.no_grad()
    def get_masked_features(self, feature_map, mask_tensor):
        if feature_map is None or mask_tensor is None or mask_tensor.sum() < 1e-6:
            return torch.zeros((1, self.feature_dim), device=self.device)

        mask_tensor = mask_tensor.to(self.device)
        if mask_tensor.dim() == 2: mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3: mask_tensor = mask_tensor.unsqueeze(1)

        mask_resized = TF.resize(mask_tensor.float(), feature_map.shape[-2:], interpolation=T.InterpolationMode.NEAREST)
        mask_resized = (mask_resized > 0.5).float()

        masked_fm = feature_map * mask_resized
        sum_features = masked_fm.sum(dim=(2, 3))
        mask_count = mask_resized.sum(dim=(2, 3))
        return sum_features / torch.clamp(mask_count, min=1e-8)


# =========================================================================================
# MODIFICATION 1: PRECOMPUTATION - changed to save node data, not graphs.
# =========================================================================================
@torch.no_grad()
def precompute_node_data_entrypoint(dataset_adapter, indices, args, feature_extractor, device):
    logger.info(f"Starting NODE data precomputation for {len(indices)} samples...")
    logger.info(f"Node data will be saved to: {args.precomputed_node_path}")
    Path(args.precomputed_node_path).mkdir(parents=True, exist_ok=True)

    loader = PyTorchDataLoader(dataset_adapter, batch_size=1, sampler=SequentialSampler(indices),
                               collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)

    saved_count, skipped_count, existing_count = 0, 0, 0
    pbar = tqdm(loader, desc="Precomputing Node Data")
    for batch_items in pbar:
        if batch_items is None: continue

        _, query_set = batch_items[0]
        query_class = query_set.get('class')
        query_idx = query_set.get('query_index')

        if not query_class or not query_idx:
            skipped_count += 1
            continue

        class_node_dir = Path(args.precomputed_node_path) / query_class
        class_node_dir.mkdir(parents=True, exist_ok=True)
        node_file_path = class_node_dir / f"{query_idx}_nodes.pt"

        if not args.overwrite_precomputed_graphs and node_file_path.exists():
            existing_count += 1
            continue

        # --- Extract node data from query image ---
        query_gt_mask_np = query_set.get('gt_mask_np')
        query_img_resnet = query_set.get('image_resnet')
        ref_shape = query_gt_mask_np.shape

        # Load SAM masks
        mask_file_path = Path(args.precomputed_mask_path) / f"{query_idx}_sam_hybrid_masks.pkl"
        if not mask_file_path.is_file():
            skipped_count += 1
            continue
        with open(mask_file_path, 'rb') as f:
            sam_output = pickle.load(f)
        sam_output.sort(key=lambda x: x['area'], reverse=True)
        candidate_masks_data_raw = sam_output[:args.max_masks_graph]

        masks_np_list_bool = [m['segmentation'].astype(bool) for m in candidate_masks_data_raw if
                              m['segmentation'].shape == ref_shape]
        if not masks_np_list_bool:
            skipped_count += 1
            continue

        num_nodes = len(masks_np_list_bool)

        # Extract features for all proposal masks
        query_feature_map = feature_extractor.get_feature_map(query_img_resnet.unsqueeze(0))
        masks_tensor = torch.from_numpy(np.stack(masks_np_list_bool)).unsqueeze(1).float()
        x_base_features = feature_extractor.get_masked_features(query_feature_map.expand(num_nodes, -1, -1, -1),
                                                                masks_tensor)

        x_geo_features = torch.stack([torch.from_numpy(get_geometric_features(m)) for m in masks_np_list_bool])
        x_sam_features = torch.tensor([[d.get('predicted_iou', 0.0), d.get('stability_score', 0.0)] for d in
                                       candidate_masks_data_raw[:num_nodes]], dtype=torch.float32)

        # Get oracle labels
        selected_indices, _ = _run_greedy_oracle_for_labels(masks_np_list_bool, query_gt_mask_np.astype(bool),
                                                            ref_shape)
        target_labels = torch.tensor([1.0 if i in selected_indices else 0.0 for i in range(num_nodes)],
                                     dtype=torch.float)

        node_data = {
            'x_base': x_base_features.cpu(),
            'x_geo': x_geo_features.cpu(),
            'x_sam': x_sam_features.cpu(),
            'y': target_labels.cpu(),
            'node_masks_np': [m.astype(np.uint8) for m in masks_np_list_bool],
            'gt_mask_np': query_gt_mask_np,
            'image_size': ref_shape,
        }

        torch.save(node_data, node_file_path)
        saved_count += 1
        pbar.set_postfix({'saved': saved_count, 'skipped': skipped_count, 'existed': existing_count})

    logger.info(
        f"Node data precomputation finished. Saved: {saved_count}, Skipped: {skipped_count}, Existed: {existing_count}")


def load_precomputed_node_data(query_class, query_idx, precomputed_node_dir):
    node_file_path = Path(precomputed_node_dir) / query_class / f"{query_idx}_nodes.pt"
    if not node_file_path.is_file(): return None
    try:
        return torch.load(node_file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Error loading precomputed node data {node_file_path}: {e}")
        return None


# =========================================================================================
# MODIFICATION 2: NEW UNIFIED GNN MODEL
# =========================================================================================
class UnifiedGNN(nn.Module):
    def __init__(self, node_feature_dim, gnn_hidden_dim, num_gnn_layers, edge_feature_dim, dropout_rate=0.5):
        super().__init__()

        self.input_lin = nn.Linear(node_feature_dim, gnn_hidden_dim)
        self.input_norm = nn.LayerNorm(gnn_hidden_dim)
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)

        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        current_dim = gnn_hidden_dim
        for _ in range(num_gnn_layers):
            conv = GATConv(in_channels=current_dim, out_channels=current_dim, heads=4, dropout=dropout_rate,
                           concat=False, edge_dim=edge_feature_dim)
            self.gnn_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(current_dim))

        self.readout_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )
        logger.info(f"UnifiedGNN initialized. GNN Hidden: {gnn_hidden_dim}, Layers: {num_gnn_layers}")

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.input_lin(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        x = self.input_dropout(x)

        for i, layer in enumerate(self.gnn_layers):
            x_res = x
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.layer_norms[i](x)
            x = x + x_res  # Residual connection

        query_node_features = x[:data.num_query_nodes]
        logits = self.readout_mlp(query_node_features)

        return logits.squeeze(-1)


# =========================================================================================
# MODIFICATION 3: NEW DYNAMIC UNIFIED GRAPH BUILDER
# =========================================================================================
def build_unified_graph(query_node_data, support_set_features, device, args):
    x_q_base = query_node_data['x_base'].to(device)
    x_q_geo = query_node_data['x_geo'].to(device)
    x_q_sam = query_node_data['x_sam'].to(device)
    num_query_nodes = x_q_base.shape[0]

    x_s_base = support_set_features.to(device)
    k_shot = x_s_base.shape[0]
    x_s_geo = torch.zeros((k_shot, x_q_geo.shape[1]), device=device)
    x_s_sam = torch.zeros((k_shot, x_q_sam.shape[1]), device=device)

    # --- THIS IS THE CORRECTED PART ---
    x_query = torch.cat([x_q_base, x_q_geo, x_q_sam], dim=1) # Use x_q_sam
    # --------------------------------

    x_support = torch.cat([x_s_base, x_s_geo, x_s_sam], dim=1)
    x_all = torch.cat([x_query, x_support], dim=0)

    edge_list = []
    edge_attr_list = []

    # Query-Query Edges (KNN based on centroids derived from geo features)
    if num_query_nodes > 1:
        centroids_q = x_q_geo[:, 1:3].cpu().numpy()
        k_actual = min(args.knn_k, num_query_nodes - 1)
        if k_actual > 0:
            try:
                adj_sparse = kneighbors_graph(centroids_q, k_actual, mode='connectivity', include_self=False, n_jobs=1)
                coo = adj_sparse.tocoo()
                q_q_edges = np.vstack((coo.row, coo.col))
                for i in range(q_q_edges.shape[1]):
                    u, v = q_q_edges[0, i], q_q_edges[1, i]
                    edge_list.append([u, v])
                    # Edge Attr: [is_QQ_edge, is_QS_edge, similarity]
                    edge_attr_list.append([1.0, 0.0, 0.0])
            except Exception as e:
                logger.warning(f"Could not build KNN graph for query nodes: {e}")


    # Query-Support Edges
    for i in range(num_query_nodes):
        for j in range(k_shot):
            support_node_idx = num_query_nodes + j
            edge_list.append([i, support_node_idx])
            edge_list.append([support_node_idx, i])

            similarity = F.cosine_similarity(x_q_base[i].unsqueeze(0), x_s_base[j].unsqueeze(0)).item()
            edge_attr_list.append([0.0, 1.0, similarity])
            edge_attr_list.append([0.0, 1.0, similarity])

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 3), device=device)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float, device=device)

    graph_data = Data(x=x_all, edge_index=edge_index, edge_attr=edge_attr)
    graph_data.y = query_node_data['y'].to(device)
    graph_data.num_query_nodes = num_query_nodes
    graph_data.node_masks_np = query_node_data['node_masks_np']
    graph_data.gt_mask_np = query_node_data['gt_mask_np']
    graph_data.image_size = query_node_data['image_size']

    return graph_data

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return batch


# --- Weight Estimation (UNCHANGED) ---
@torch.no_grad()
def estimate_pos_weight(dataset_adapter, indices, precomputed_mask_path, args, device):
    # This function remains exactly the same as it operates on the raw dataset and masks
    logger.info(f"Estimating pos_weight using {args.estimate_weight_subset:.1%} of {len(indices)} training samples...")
    num_pos, num_neg = 0, 0
    subset_size = int(len(indices) * args.estimate_weight_subset)
    if subset_size == 0: return torch.tensor([1.0], device=device)
    subset_indices_for_est = random.sample(indices, subset_size)
    est_loader = PyTorchDataLoader(dataset_adapter, batch_size=1, sampler=SequentialSampler(subset_indices_for_est),
                                   collate_fn=collate_fn, num_workers=args.num_workers)

    pbar_est = tqdm(est_loader, desc="Estimating pos_weight", leave=False)
    for batch_items in pbar_est:
        if batch_items is None: continue
        _, query_set = batch_items[0]
        query_idx = query_set.get('query_index')
        query_gt_mask_np = query_set.get('gt_mask_np')
        if query_idx is None or query_gt_mask_np is None: continue

        mask_file_path = Path(precomputed_mask_path) / f"{query_idx}_sam_hybrid_masks.pkl"
        if not mask_file_path.is_file(): continue
        with open(mask_file_path, 'rb') as f:
            sam_output = pickle.load(f)
        sam_output.sort(key=lambda x: x['area'], reverse=True)
        candidate_masks_data_raw = sam_output[:args.max_masks_graph]
        masks_np_list_bool = [m['segmentation'].astype(bool) for m in candidate_masks_data_raw if
                              m['segmentation'].shape == query_gt_mask_np.shape]

        if not masks_np_list_bool: continue
        selected_oracle_indices, _ = _run_greedy_oracle_for_labels(masks_np_list_bool, query_gt_mask_np.astype(bool),
                                                                   query_gt_mask_np.shape)
        num_pos += len(selected_oracle_indices)
        num_neg += len(masks_np_list_bool) - len(selected_oracle_indices)

    if num_pos == 0:
        logger.warning("No positive nodes found during weight estimation. Using default 1.0.")
        return torch.tensor([1.0], device=device)

    pos_weight_value = num_neg / num_pos
    logger.info(f"Calculated pos_weight = {pos_weight_value:.4f} (neg: {num_neg} / pos: {num_pos})")
    return torch.tensor([pos_weight_value], device=device)


# =========================================================================================
# MODIFICATION 4: UPDATED TRAINING AND EVALUATION LOOPS
# =========================================================================================
def train_gnn(model, train_loader, optimizer, criterion, device, args, feature_extractor, scheduler=None):
    model.train()
    total_loss, processed_graphs, epoch_train_ious = 0, 0, []

    pbar = tqdm(train_loader, desc=f"Epoch {args.current_epoch}/{args.epochs} Training", leave=False)
    for batch_items in pbar:
        if batch_items is None: continue

        optimizer.zero_grad()

        # We will process one episode at a time as graph sizes vary
        for support_set, query_set in batch_items:
            # Load precomputed query node data
            query_node_data = load_precomputed_node_data(query_set['class'], query_set['query_index'],
                                                         args.precomputed_node_path)
            if query_node_data is None or len(query_node_data['node_masks_np']) == 0:
                continue

            # Extract support features
            support_features_list = []
            for item in support_set:
                fm = feature_extractor.get_feature_map(item['image'].unsqueeze(0))
                sf = feature_extractor.get_masked_features(fm, item['mask'])
                support_features_list.append(sf)
            if not support_features_list: continue
            support_features = torch.cat(support_features_list, dim=0)

            # Build unified graph dynamically
            graph = build_unified_graph(query_node_data, support_features, device, args)
            if graph.num_nodes == 0 or graph.edge_index.numel() == 0:
                continue

            # Forward pass
            out_logits = model(graph)
            targets = graph.y
            if out_logits.shape[0] != targets.shape[0]: continue

            loss = criterion(out_logits, targets)
            loss.backward()

            # IoU calculation for monitoring
            with torch.no_grad():
                pred_labels = (torch.sigmoid(out_logits) > args.eval_threshold).cpu().numpy()
                selected_masks = [graph.node_masks_np[i] for i, label in enumerate(pred_labels) if label == 1]
                final_pred_mask = combine_masks(selected_masks, graph.image_size)
                iou = calculate_iou(final_pred_mask, graph.gt_mask_np.astype(bool))
                epoch_train_ious.append(iou)

            total_loss += loss.item()
            processed_graphs += 1

        # Step optimizer and scheduler after processing the batch
        if processed_graphs > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if scheduler: scheduler.step()

        avg_loss = total_loss / processed_graphs if processed_graphs > 0 else 0
        avg_iou = np.mean(epoch_train_ious) if epoch_train_ious else 0
        pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'avg_train_iou': f"{avg_iou:.4f}"})

    avg_loss_epoch = total_loss / processed_graphs if processed_graphs > 0 else 0
    avg_train_iou_epoch = np.mean(epoch_train_ious) if epoch_train_ious else 0
    logger.info(
        f"Epoch {args.current_epoch} Train Finished. Avg Loss: {avg_loss_epoch:.4f}, Avg Train IoU: {avg_train_iou_epoch:.4f}")
    return avg_loss_epoch, avg_train_iou_epoch, 0  # grad norm not tracked per item anymore


@torch.no_grad()
def evaluate_gnn(model, eval_loader, criterion, device, args, feature_extractor, eval_threshold=0.5):
    model.eval()
    total_loss, all_ious, processed_graphs = 0, [], 0

    pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
    for batch_items in pbar:
        if batch_items is None: continue

        for support_set, query_set in batch_items:
            query_node_data = load_precomputed_node_data(query_set['class'], query_set['query_index'],
                                                         args.precomputed_node_path)
            if query_node_data is None or len(query_node_data['node_masks_np']) == 0:
                continue

            support_features_list = []
            for item in support_set:
                fm = feature_extractor.get_feature_map(item['image'].unsqueeze(0))
                sf = feature_extractor.get_masked_features(fm, item['mask'])
                support_features_list.append(sf)
            if not support_features_list: continue
            support_features = torch.cat(support_features_list, dim=0)

            graph = build_unified_graph(query_node_data, support_features, device, args)
            if graph.num_nodes == 0 or graph.edge_index.numel() == 0:
                continue

            out_logits = model(graph)
            targets = graph.y

            if out_logits.shape[0] == targets.shape[0]:
                loss_val = criterion(out_logits, targets)
                total_loss += loss_val.item()

            pred_labels = (torch.sigmoid(out_logits) > eval_threshold).cpu().numpy()
            selected_masks = [graph.node_masks_np[i] for i, label in enumerate(pred_labels) if label == 1]
            final_pred_mask = combine_masks(selected_masks, graph.image_size)
            iou = calculate_iou(final_pred_mask, graph.gt_mask_np.astype(bool))
            all_ious.append(iou)
            processed_graphs += 1

    avg_loss_epoch = total_loss / processed_graphs if processed_graphs > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    logger.info(f"Evaluation Finished. Avg Loss: {avg_loss_epoch:.4f}, Avg IoU: {avg_iou:.4f}")
    return avg_loss_epoch, avg_iou


# =========================================================================================
# MODIFICATION 5: MAIN SCRIPT AND HYPERPARAMETERS
# =========================================================================================
def main(args):
    if args.seed is not None:
        random.seed(args.seed);
        np.random.seed(args.seed);
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # The new edge features for the unified graph
    args.edge_feature_dim = 3  # [is_QQ_edge, is_QS_edge, similarity]

    feature_extractor = None
    if 'precompute' in args.mode or args.mode in ['train', 'eval']:
        logger.info("Initializing FeatureExtractor...")
        feature_extractor = FeatureExtractor(device)

    pascal_image_transform = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.mode == 'precompute_nodes':
        logger.info(
            f"Precomputation mode: Precomputing nodes for 'trn' and 'val' splits of PASCAL fold {args.pascal_fold}")
        for split in ['trn', 'val']:
            precompute_dataset = DatasetPASCAL(datapath=args.pascal_datapath, fold=args.pascal_fold,
                                               transform=pascal_image_transform,
                                               split=split, shot=args.k_shot, use_original_imgsize=False)
            adapter = PASCALAdapterDataset(precompute_dataset, args.img_size, PASCAL_CLASS_NAMES_LIST)
            indices = list(range(len(adapter)))
            precompute_node_data_entrypoint(adapter, indices, args, feature_extractor, device)
        logger.info("Precomputation complete. Exiting.")
        return

    # --- Setup for Train/Eval ---
    base_train_dataset = DatasetPASCAL(datapath=args.pascal_datapath, fold=args.pascal_fold,
                                       transform=pascal_image_transform, split='trn', shot=args.k_shot,
                                       use_original_imgsize=False)
    train_dataset_adapter = PASCALAdapterDataset(base_train_dataset, args.img_size, PASCAL_CLASS_NAMES_LIST)
    base_eval_dataset = DatasetPASCAL(datapath=args.pascal_datapath, fold=args.pascal_fold,
                                      transform=pascal_image_transform, split='val', shot=args.k_shot,
                                      use_original_imgsize=False)
    eval_dataset_adapter = PASCALAdapterDataset(base_eval_dataset, args.img_size, PASCAL_CLASS_NAMES_LIST)

    train_loader = PyTorchDataLoader(train_dataset_adapter, batch_size=args.batch_size,
                                     sampler=RandomSampler(train_dataset_adapter), collate_fn=collate_fn,
                                     num_workers=args.num_workers, drop_last=True)
    eval_loader = PyTorchDataLoader(eval_dataset_adapter, batch_size=args.eval_batch_size,
                                    sampler=SequentialSampler(eval_dataset_adapter), collate_fn=collate_fn,
                                    num_workers=args.num_workers)

    # --- Model Initialization ---
    node_feat_dim = args.feature_size + 4 + 2  # x_base + x_geo + x_sam
    model = UnifiedGNN(
        node_feature_dim=node_feat_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_gnn_layers=args.gnn_layers,
        edge_feature_dim=args.edge_feature_dim,
        dropout_rate=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                     eta_min=args.lr * 0.01) if args.use_scheduler else None

    pos_weight_tensor = torch.tensor([1.0], device=device)
    if args.use_loss_weighting:
        # Re-enable estimation by default
        pos_weight_tensor = estimate_pos_weight(train_dataset_adapter, list(range(len(train_dataset_adapter))),
                                                args.precomputed_mask_path, args, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # --- Training/Evaluation Execution ---
    if args.mode == 'train':
        logger.info(f"======== Starting Training (Epochs: {args.epochs}) ========")
        best_eval_iou = -1.0
        # ... training loop setup ...
        for epoch in range(1, args.epochs + 1):
            args.current_epoch = epoch
            # ...
            train_loss, train_iou, _ = train_gnn(model, train_loader, optimizer, criterion, device, args,
                                                 feature_extractor, scheduler)
            eval_loss, eval_iou = evaluate_gnn(model, eval_loader, criterion, device, args, feature_extractor,
                                               args.eval_threshold)

            logger.info(
                f"Epoch {epoch} Summary | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Eval Loss: {eval_loss:.4f} | Eval IoU: {eval_iou:.4f}")

            if eval_iou > best_eval_iou:
                best_eval_iou = eval_iou
                # ... save best model ...

    elif args.mode == 'eval':
        # ... evaluation logic (remains the same) ...
        pass
    else:
        logger.error(f"Invalid mode '{args.mode}'. Choose 'train', 'eval', or 'precompute_nodes'.")


if __name__ == "__main__":
    args = SimpleNamespace(
        # IMPORTANT: Change mode to 'precompute_nodes' first, then 'train'
        mode='train',  # OPTIONS: 'precompute_nodes', 'train', 'eval'

        pascal_datapath=DEFAULT_PASCAL_DATAPATH,
        pascal_fold=0,
        precomputed_mask_path=DEFAULT_PRECOMPUTED_MASK_PATH,
        precomputed_node_path=DEFAULT_PRECOMPUTED_NODE_PATH,  # NEW
        output_dir="./train_model_outputs_unified",
        eval_model_path="",

        device=DEFAULT_DEVICE,
        seed=42,
        num_workers=0,  # Set to 0 on Windows for easier debugging

        k_shot=DEFAULT_K_SHOT,
        img_size=DEFAULT_IMG_SIZE,
        max_masks_graph=DEFAULT_MAX_MASKS_GRAPH,
        knn_k=10,  # For building query-query edges

        # --- REVISED Model Hyperparameters for UnifiedGNN ---
        feature_size=DEFAULT_FEATURE_SIZE,
        gnn_hidden_dim=256,
        gnn_layers=2,  # REDUCED: A shallower GNN is a better start
        gnn_heads=4,  # Kept, GATConv not as prone to over-smoothing
        dropout=0.5,  # INCREASED: Stronger regularization

        # --- REVISED Training Hyperparameters ---
        epochs=200,  # A lower number of epochs to start
        batch_size=4,  # REDUCED: Dynamic graph building is more memory intensive
        eval_batch_size=4,
        lr=5e-5,  # REDUCED: A more stable learning rate
        weight_decay=1e-4,
        use_scheduler=True,
        clip_grad_norm=1.0,
        use_loss_weighting=True,
        estimate_weight_subset=0.1,

        eval_threshold=0.5,
        overwrite_precomputed_graphs=False,
        current_epoch=0,
    )

    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (KeyboardInterrupt).");
        sys.exit(0)
    except Exception as main_e:
        logger.critical(f"Unhandled exception in main: {main_e}", exc_info=True);
        sys.exit(1)