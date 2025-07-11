# --- Standard and Third-Party Library Imports ---
import logging
import sys
import os
import random
import time
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, Sampler, SequentialSampler, RandomSampler
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms as T

# --- PyTorch Geometric Imports ---
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import TransformerConv
except ImportError:
    print("ERROR: PyTorch Geometric not found.")
    print("Please install it: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# --- Scientific Computing and Image Processing ---
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


# --- PASCAL-5i Dataset (Core Data Loading) ---
class DatasetPASCAL(Dataset):
    """
    Handles loading of images and masks for the PASCAL-5i few-shot segmentation benchmark.
    """

    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.nclass = 20
        self.nfolds = 4

        self.img_path = Path(datapath) / 'JPEGImages'
        self.ann_path = Path(datapath) / 'SegmentationClassAug'
        self.transform = transform

        self.class_ids = self._build_class_ids()
        self.img_metadata = self._build_img_metadata(datapath)
        self.img_metadata_classwise = self._build_img_metadata_classwise()

    def __len__(self):
        # In validation, we often evaluate on a fixed subset (e.g., 1000 samples)
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # The modulo allows for oversampling during validation if needed
        idx %= len(self.img_metadata)
        query_name, support_names, class_sample = self._sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self._load_frame(query_name,
                                                                                                support_names)

        # Apply transformations
        query_img = self.transform(query_img)
        support_imgs = torch.stack([self.transform(s_img) for s_img in support_imgs])

        # Resize masks to match transformed image size and create binary masks
        query_mask, _ = self._process_mask(query_cmask, class_sample, query_img.shape[-2:])
        support_masks = torch.stack(
            [self._process_mask(scmask, class_sample, support_imgs.shape[-2:])[0] for scmask in support_cmasks])

        return {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id': torch.tensor(class_sample)
        }

    def _process_mask(self, mask, class_id, target_size):
        """ Resizes and creates a binary mask for the target class. """
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=target_size, mode='nearest').squeeze()
        # PASCAL class values are 1-20, class_id is 0-19
        binary_mask = (mask == class_id + 1).float()
        ignore_idx = (mask == 255).float()  # Boundary region
        return binary_mask, ignore_idx

    def _build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_trn if self.split == 'trn' else class_ids_val

    def _load_frame(self, query_name, support_names):
        query_img = self._read_img(query_name)
        query_mask = self._read_mask(query_name)
        support_imgs = [self._read_img(name) for name in support_names]
        support_masks = [self._read_mask(name) for name in support_names]
        return query_img, query_mask, support_imgs, support_masks, query_img.size

    def _read_mask(self, img_name):
        mask_path = self.ann_path / f'{img_name}.png'
        return torch.tensor(np.array(Image.open(mask_path)), dtype=torch.long)

    def _read_img(self, img_name):
        img_path = self.img_path / f'{img_name}.jpg'
        return Image.open(img_path).convert('RGB')

    def _sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        # Get all available support images for the class, excluding the query image
        available_supports = [name for name in self.img_metadata_classwise[class_sample] if name != query_name]

        if len(available_supports) >= self.shot:
            # Sample without replacement if enough unique supports are available
            support_names = random.sample(available_supports, self.shot)
        else:
            # Sample with replacement if not enough unique supports are available
            logger.warning(f"Class {class_sample}: Not enough unique support images. Sampling with replacement.")
            support_names = random.choices(self.img_metadata_classwise[class_sample], k=self.shot)

        return query_name, support_names, class_sample

    def _build_img_metadata(self, datapath):
        def read_metadata(split, fold_id):
            split_file_path = Path(datapath) / "splits" / "folds" / f'{split}' / f'fold{fold_id}.txt'
            if not split_file_path.is_file():
                raise FileNotFoundError(f"PASCAL split file not found: {split_file_path}")
            with open(split_file_path, 'r') as f:
                metadata = f.read().splitlines()
            # Format: 'image_id__class_id'
            return [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in metadata if data]

        img_metadata = []
        if self.split == 'trn':
            for fold_id in range(self.nfolds):
                if fold_id != self.fold:
                    img_metadata.extend(read_metadata(self.split, fold_id))
        else:  # val
            img_metadata = read_metadata(self.split, self.fold)

        logger.info(f'PASCAL Fold {self.fold} ({self.split}): {len(img_metadata)} images loaded.')
        return img_metadata

    def _build_img_metadata_classwise(self):
        img_metadata_classwise = {i: [] for i in range(self.nclass)}
        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class].append(img_name)
        return img_metadata_classwise


# --- Adapter Dataset to Structure Data for Model ---
class PASCALAdapterDataset(Dataset):
    """
    Wraps the DatasetPASCAL to provide data in the (support_set, query_set) format
    expected by the training and evaluation loops.
    """

    def __init__(self, pascal_dataset_instance, target_image_size, class_names):
        self.pascal_dataset = pascal_dataset_instance
        self.target_image_size = (target_image_size, target_image_size)
        self.class_names = class_names
        logger.info(f"PASCALAdapterDataset initialized for {len(self.pascal_dataset)} episodes.")

    def __len__(self):
        return len(self.pascal_dataset)

    def __getitem__(self, idx):
        try:
            raw_item = self.pascal_dataset[idx]

            # Structure the support set
            support_set = [
                {'image': raw_item['support_imgs'][i], 'mask': raw_item['support_masks'][i]}
                for i in range(self.pascal_dataset.shot)
            ]

            # Structure the query set
            class_id = raw_item['class_id'].item()
            query_set = {
                'image_tensor': raw_item['query_img'],
                'gt_mask_np': raw_item['query_mask'].cpu().numpy().astype(np.uint8),
                'class_name': self.class_names[class_id],
                'query_name': raw_item['query_name'],
                'class_id': class_id
            }
            return support_set, query_set
        except Exception as e:
            logger.error(f"Error in PASCALAdapterDataset at index {idx}: {e}", exc_info=True)
            return None  # To be filtered by collate_fn


# --- Configuration ---
PASCAL_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# --- Utility Functions ---
def calculate_iou(mask1, mask2):
    """Calculates Intersection over Union (IoU) between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0 if intersection == 0 else 0.0


def combine_masks(mask_list, ref_shape):
    """Combines a list of masks into a single boolean mask."""
    if not mask_list:
        return np.zeros(ref_shape, dtype=bool)
    combined = np.zeros(ref_shape, dtype=bool)
    for mask in mask_list:
        combined = np.logical_or(combined, mask.astype(bool))
    return combined


def get_geometric_features(mask_np):
    """Extracts normalized geometric features from a mask."""
    if mask_np.sum() == 0:
        return np.zeros(4, dtype=np.float32)

    props = regionprops(mask_np.astype(np.uint8))
    if not props:
        return np.zeros(4, dtype=np.float32)

    props = props[0]
    h, w = mask_np.shape
    norm_area = props.area / (h * w)
    center_y, center_x = props.centroid
    norm_center_y, norm_center_x = center_y / h, center_x / w

    minr, minc, maxr, maxc = props.bbox
    bbox_h, bbox_w = float(maxr - minr), float(maxc - minc)
    aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 1.0

    return np.array([norm_area, norm_center_y, norm_center_x, aspect_ratio], dtype=np.float32)


def run_greedy_oracle_for_labels(candidate_masks, gt_mask):
    """Determines the optimal set of candidate masks to reconstruct the ground truth."""
    if not candidate_masks:
        return set()

    selected_indices = set()
    current_combined_mask = np.zeros(gt_mask.shape, dtype=bool)
    current_best_iou = calculate_iou(current_combined_mask, gt_mask)

    while True:
        best_iou_this_iter = current_best_iou
        best_action = None

        # Try adding each unselected mask
        for i, mask in enumerate(candidate_masks):
            if i not in selected_indices:
                potential_mask = np.logical_or(current_combined_mask, mask)
                iou = calculate_iou(potential_mask, gt_mask)
                if iou > best_iou_this_iter:
                    best_iou_this_iter = iou
                    best_action = ('add', i)

        # Try removing each selected mask
        for i in selected_indices:
            temp_selected = selected_indices - {i}
            if not temp_selected:
                potential_mask = np.zeros(gt_mask.shape, dtype=bool)
            else:
                potential_mask = combine_masks([candidate_masks[j] for j in temp_selected], gt_mask.shape)
            iou = calculate_iou(potential_mask, gt_mask)
            if iou > best_iou_this_iter:
                best_iou_this_iter = iou
                best_action = ('remove', i)

        if best_action:
            action_type, idx = best_action
            if action_type == 'add':
                selected_indices.add(idx)
            else:  # remove
                selected_indices.remove(idx)
            current_combined_mask = combine_masks([candidate_masks[j] for j in selected_indices], gt_mask.shape)
            current_best_iou = best_iou_this_iter
        else:
            break

    return selected_indices


# --- Feature Extractor ---
class FeatureExtractor(nn.Module):
    """
    Extracts deep features from images using a pre-trained ResNet-50.
    """

    def __init__(self, device):
        super().__init__()
        logger.info("Initializing ResNet-50 feature extractor...")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        self.resnet_layer4 = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_layer4.eval()
        self.device = device
        self.resnet_layer4.to(device)
        self.feature_dim = 2048
        logger.info(f"ResNet-50 loaded on device '{device}'. Feature dim: {self.feature_dim}")

    @torch.no_grad()
    def get_feature_map(self, image_tensor_batch):
        return self.resnet_layer4(image_tensor_batch.to(self.device))

    @torch.no_grad()
    def get_masked_features(self, feature_map, mask_tensor):
        if mask_tensor.sum() < 1e-6:
            return torch.zeros((feature_map.shape[0], self.feature_dim), device=self.device)

        mask_tensor = mask_tensor.to(self.device)
        fm_h, fm_w = feature_map.shape[-2:]

        # Ensure mask has a channel dimension for interpolation
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        mask_resized = F.interpolate(mask_tensor.float(), size=(fm_h, fm_w), mode='nearest')

        masked_fm = feature_map * mask_resized
        sum_features = masked_fm.sum(dim=(2, 3))
        mask_area = mask_resized.sum(dim=(2, 3))

        return sum_features / torch.clamp(mask_area, min=1e-8)


# --- DYNAMIC Graph Construction ---
@torch.no_grad()
def construct_graph_dynamically(support_set, query_set, feature_extractor, args):
    """
    Constructs a PyG graph object on-the-fly for a given query image and support set.
    """
    query_name = query_set['query_name']
    query_gt_mask_np = query_set['gt_mask_np']
    ref_shape = query_gt_mask_np.shape

    # 1. Load Precomputed Candidate Masks (SAM)
    mask_file_path = Path(args.precomputed_mask_path) / f"{query_name}_sam_hybrid_masks.pkl"
    if not mask_file_path.is_file():
        logger.warning(f"SAM mask file not found for {query_name}. Skipping graph construction.")
        return None

    with open(mask_file_path, 'rb') as f:
        sam_output = pickle.load(f)
    sam_output.sort(key=lambda x: x['area'], reverse=True)

    candidate_masks_bool = [m['segmentation'] for m in sam_output[:args.max_masks_graph]]
    sam_metadata = sam_output[:args.max_masks_graph]

    num_nodes = len(candidate_masks_bool)
    if num_nodes == 0:
        return None

    # 2. Extract Node Features
    query_feature_map = feature_extractor.get_feature_map(query_set['image_tensor'].unsqueeze(0))
    masks_tensor = torch.from_numpy(np.stack(candidate_masks_bool)).unsqueeze(1)

    x_base = feature_extractor.get_masked_features(query_feature_map.expand(num_nodes, -1, -1, -1), masks_tensor)
    x_geo = torch.tensor([get_geometric_features(m) for m in candidate_masks_bool], dtype=torch.float32)
    x_sam = torch.tensor([[m.get('predicted_iou', 0), m.get('stability_score', 0)] for m in sam_metadata],
                         dtype=torch.float32)

    # 3. Determine Node Labels via Greedy Oracle
    selected_indices = run_greedy_oracle_for_labels(candidate_masks_bool, query_gt_mask_np.astype(bool))
    labels = torch.zeros(num_nodes, dtype=torch.float)
    labels[list(selected_indices)] = 1.0

    # 4. Define Graph Edges and Edge Features
    edge_index, edge_attr = torch.empty((2, 0), dtype=torch.long), torch.empty((0, args.edge_feature_dim))
    if num_nodes > 1:
        node_centroids = [regionprops(m.astype(np.uint8))[0].centroid for m in candidate_masks_bool if m.sum() > 0]
        if len(node_centroids) > 1:
            k = min(args.knn_k, len(node_centroids) - 1)
            adj_sparse = kneighbors_graph(node_centroids, k, mode='connectivity', include_self=False)
            coo = adj_sparse.tocoo()
            edge_index = torch.from_numpy(np.vstack((coo.row, coo.col))).long()

            edge_attr_list = []
            for src, trg in edge_index.t().tolist():
                iou = calculate_iou(candidate_masks_bool[src], candidate_masks_bool[trg])
                c1, c2 = np.array(node_centroids[src]), np.array(node_centroids[trg])
                dist = np.linalg.norm(c1 - c2) / np.sqrt(ref_shape[0] ** 2 + ref_shape[1] ** 2)
                area1, area2 = candidate_masks_bool[src].sum(), candidate_masks_bool[trg].sum()
                area_ratio = min(area1, area2) / (max(area1, area2) + 1e-6)
                edge_attr_list.append([iou, dist, area_ratio])
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return Data(
        x_base=x_base.cpu(), x_geo=x_geo.cpu(), x_sam=x_sam.cpu(),
        edge_index=edge_index.cpu(), edge_attr=edge_attr.cpu(), y=labels.cpu(),
        node_masks_np=[m.astype(np.uint8) for m in candidate_masks_bool],
        gt_mask_np=query_gt_mask_np,
        image_size=ref_shape, num_nodes=num_nodes
    )


# --- GNN Model Definition ---
class SupportAwareMaskGNN(nn.Module):
    def __init__(self, base_node_visual_dim, gnn_hidden_dim, num_gnn_layers, num_heads, edge_feature_dim, dropout):
        super().__init__()
        # The GNN's input dimension is the visual features + cosine similarity score
        self.gnn_input_dim = base_node_visual_dim + 1
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(self.gnn_input_dim),
            nn.Linear(self.gnn_input_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gnn_layers = nn.ModuleList([
            TransformerConv(gnn_hidden_dim, gnn_hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout,
                            edge_dim=edge_feature_dim)
            for _ in range(num_gnn_layers)
        ])
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(gnn_hidden_dim) for _ in range(num_gnn_layers)])

        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(gnn_hidden_dim),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )
        logger.info(f"SupportAwareMaskGNN initialized. GNN Input Dim: {self.gnn_input_dim}, Hidden: {gnn_hidden_dim}")

    def forward(self, data, support_prototype):
        x_base, edge_index, edge_attr = data.x_base, data.edge_index, data.edge_attr

        # 1. Compute similarity with support prototype and concatenate
        cosine_sim = F.cosine_similarity(x_base, support_prototype.squeeze(0), dim=1).unsqueeze(1)
        x = torch.cat([x_base, cosine_sim], dim=-1)

        # 2. Pass through GNN
        x = self.input_mlp(x)
        for layer, norm in zip(self.gnn_layers, self.gnn_norms):
            x = x + layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x)

        return self.readout_mlp(x).squeeze(-1)


# --- Custom Collate Function ---
def collate_fn(batch):
    # Filter out None items that may result from errors in __getitem__
    return [item for item in batch if item is not None]


# --- Training and Evaluation Functions ---
def run_epoch(model, loader, feature_extractor, criterion, optimizer, device, args, is_train):
    model.train(is_train)
    total_loss, total_iou, processed_graphs = 0, 0, 0

    desc = f"Epoch {args.current_epoch} Training" if is_train else "Evaluating"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        if not batch: continue

        for support_set, query_set in batch:
            # 1. Dynamically construct the graph for the current query
            graph_data = construct_graph_dynamically(support_set, query_set, feature_extractor, args)
            if graph_data is None or graph_data.num_nodes == 0:
                continue

            # 2. Compute the support prototype
            support_features = []
            for item in support_set:
                feat = feature_extractor.get_masked_features(
                    feature_extractor.get_feature_map(item['image'].unsqueeze(0)),
                    item['mask'].unsqueeze(0)
                )
                support_features.append(feat)

            if not support_features: continue
            support_prototype = torch.stack(support_features).mean(dim=0).to(device)

            # 3. Model forward pass and loss calculation
            graph_data = graph_data.to(device)
            if is_train:
                optimizer.zero_grad()

            logits = model(graph_data, support_prototype)
            loss = criterion(logits, graph_data.y)

            if is_train:
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            # 4. Calculate IoU for this graph
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > args.eval_threshold).cpu().numpy()
                selected_masks = [graph_data.node_masks_np[i] for i, p in enumerate(preds) if p == 1]
                final_mask = combine_masks(selected_masks, graph_data.image_size)
                iou = calculate_iou(final_mask, graph_data.gt_mask_np.astype(bool))

            total_loss += loss.item()
            total_iou += iou
            processed_graphs += 1

            pbar.set_postfix({
                'loss': f"{total_loss / processed_graphs:.4f}",
                'iou': f"{total_iou / processed_graphs:.4f}"
            })

    avg_loss = total_loss / processed_graphs if processed_graphs > 0 else 0
    avg_iou = total_iou / processed_graphs if processed_graphs > 0 else 0
    return avg_loss, avg_iou


# --- Main Script ---
def main(args):
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize Feature Extractor
    feature_extractor = FeatureExtractor(device)

    # Initialize Datasets and DataLoaders
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_train_dataset = DatasetPASCAL(args.pascal_datapath, args.pascal_fold, transform, 'trn', args.k_shot, False)
    train_adapter = PASCALAdapterDataset(base_train_dataset, args.img_size, PASCAL_CLASS_NAMES)
    train_loader = PyTorchDataLoader(train_adapter, batch_size=args.batch_size, sampler=RandomSampler(train_adapter),
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    base_eval_dataset = DatasetPASCAL(args.pascal_datapath, args.pascal_fold, transform, 'val', args.k_shot, False)
    eval_adapter = PASCALAdapterDataset(base_eval_dataset, args.img_size, PASCAL_CLASS_NAMES)
    eval_loader = PyTorchDataLoader(eval_adapter, batch_size=args.eval_batch_size,
                                    sampler=SequentialSampler(eval_adapter),
                                    num_workers=args.num_workers, collate_fn=collate_fn)

    # Initialize Model, Loss, and Optimizer
    model = SupportAwareMaskGNN(
        base_node_visual_dim=feature_extractor.feature_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_gnn_layers=args.gnn_layers,
        num_heads=args.gnn_heads,
        edge_feature_dim=args.edge_feature_dim,
        dropout=args.dropout
    ).to(device)

    pos_weight = torch.tensor([args.pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Handle training or evaluation mode
    if args.mode == 'train':
        logger.info(f"======== Starting Training for {args.epochs} Epochs ========")
        best_eval_iou = -1.0
        output_dir = Path(args.output_dir) / f"pascal_f{args.pascal_fold}_{time.strftime('%Y%m%d-%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            args.current_epoch = epoch
            train_loss, train_iou = run_epoch(model, train_loader, feature_extractor, criterion, optimizer, device,
                                              args, is_train=True)
            eval_loss, eval_iou = run_epoch(model, eval_loader, feature_extractor, criterion, None, device, args,
                                            is_train=False)

            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
                        f"Eval Loss: {eval_loss:.4f}, Eval IoU: {eval_iou:.4f}")

            if eval_iou > best_eval_iou:
                best_eval_iou = eval_iou
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                logger.info(f"*** New best model saved with Eval IoU: {best_eval_iou:.4f} ***")

    elif args.mode == 'eval':
        logger.info(f"======== Starting Evaluation ========")
        if not args.eval_model_path or not Path(args.eval_model_path).is_file():
            logger.error(f"Evaluation model path invalid: {args.eval_model_path}")
            return
        model.load_state_dict(torch.load(args.eval_model_path, map_location=device))

        eval_loss, eval_iou = run_epoch(model, eval_loader, feature_extractor, criterion, None, device, args,
                                        is_train=False)
        logger.info(f"--- Final Evaluation Results ---")
        logger.info(f"Eval Loss: {eval_loss:.4f} | Mean IoU: {eval_iou:.4f}")

    logger.info("Script finished.")


if __name__ == "__main__":
    # --- Configuration ---
    # DEFAULT_FSS_PATH = '/home/farahani/khan/fewshot_data/fewshot_data' # Old
    DEFAULT_PASCAL_DATAPATH = r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw"  # NEW: Root for VOC2012, SegmentationClassAug, splits
    DEFAULT_PRECOMPUTED_MASK_PATH = r'C:\Users\Khan\Desktop\New folder (2)\sam_precomputed_masks_hybrid\pascal_voc2012_sam_masks'  # NEW: Consider separate SAM masks for PASCAL
    # DEFAULT_PRECOMPUTED_MASK_PATH = r'C:\Users\Khan\PycharmProjects\FSS-Research-\pascal\pascal_sam_precomputed_masks_hybrid\JPEGImages'  # NEW: Consider separate SAM masks for PASCAL

    # DEFAULT_PRECOMPUTED_GRAPH_PATH = './precomputed_pascal_graphs_hybrid_v1_main_512'
    DEFAULT_PRECOMPUTED_GRAPH_PATH = './precomputed_pascal_graphs_hybrid_v1_main_fold2'
    DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEFAULT_K_SHOT = 1
    DEFAULT_IMG_SIZE = 256  # This will be used for PASCAL image/mask resizing
    DEFAULT_FEATURE_SIZE = 2048
    DEFAULT_MAX_MASKS_GRAPH = 30
    DEFAULT_GRAPH_CONNECTIVITY = 'knn'
    DEFAULT_KNN_K = 10
    DEFAULT_OVERLAP_THRESH = 0.1
    DEFAULT_EDGE_FEATURE_MODE = 'iou_dist_area'
    DEFAULT_PASCAL_FOLD = 2  # NEW: For PASCAL-5i fold (0, 1, 2, 3)
    DEFAULT_PASCAL_USE_ORIGINAL_IMGSIZE = False  # NEW: Must be False for consistency with fixed IMG_SIZE and SAM

    args = SimpleNamespace(
        # --- Execution Mode ---
        mode='train',  # 'train' or 'eval'

        # --- Paths ---
        pascal_datapath=DEFAULT_PASCAL_DATAPATH,
        precomputed_mask_path=DEFAULT_PRECOMPUTED_MASK_PATH,
        output_dir="./train_outputs",
        eval_model_path="",  # Path to a .pth file for 'eval' mode

        # --- General Settings ---
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        seed=42,
        num_workers=0,  # Set to 0 for debugging, >0 for performance

        # --- Dataset & Episode Settings ---
        pascal_fold=0,
        k_shot=1,
        img_size=256,

        # --- Graph Construction ---
        max_masks_graph=30,
        knn_k=10,
        edge_feature_dim=3,  # Fixed: iou, norm_dist, area_ratio

        # --- Model Hyperparameters ---
        gnn_hidden_dim=256,
        gnn_layers=4,
        gnn_heads=4,
        dropout=0.3,

        # --- Training Hyperparameters ---
        epochs=50,
        batch_size=4,
        eval_batch_size=4,
        lr=1e-4,
        weight_decay=1e-4,
        clip_grad_norm=1.0,
        pos_weight_value=8.0,  # BCEWithLogitsLoss positive class weight

        # --- Evaluation Settings ---
        eval_threshold=0.5,

        # --- Internal (do not set manually) ---
        current_epoch=0,
    )

    try:
        main(args)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)