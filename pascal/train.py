import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
# import torchvision.transforms as transforms # Original
from torchvision import transforms as T  # Changed to T to avoid conflict if DatasetPASCAL uses 'transforms'
from torch.utils.data import Dataset, DataLoader as PyTorchDataLoader, SubsetRandomSampler, SequentialSampler, \
    RandomSampler  # Added RandomSampler

# --- PyTorch Geometric Imports ---
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, TransformerConv
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
matplotlib.use('TkAgg')  # or 'Qt5Agg'

# --- PASCAL-5i Dataset (Provided by User) ---
import torch.nn.functional as F  # Required by DatasetPASCAL


class DatasetPASCAL(Dataset):  # User-provided DatasetPASCAL
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'folds'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        # UPDATED: Use datapath consistently for image and annotation paths
        self.img_path = os.path.join(datapath, 'JPEGImages')
        self.ann_path = os.path.join(datapath, 'SegmentationClassAug')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata(datapath)  # Pass datapath
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name,
                                                                                               support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:  # This ensures mask is resized to transformed image size
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:],
                                        mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs_transformed = []
        for support_img in support_imgs:
            support_imgs_transformed.append(self.transform(support_img))
        support_imgs = torch.stack(support_imgs_transformed)

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            # Resize mask to match transformed support image size
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,  # Float tensor (H,W) with 0.0 or 1.0
                 'query_name': query_name,  # String
                 'query_ignore_idx': query_ignore_idx,
                 'org_query_imsize': org_qry_imsize,  # Tuple (W,H)
                 'support_imgs': support_imgs,  # Tensor (K,C,H,W)
                 'support_masks': support_masks,  # Tensor (K,H,W) float 0.0 or 1.0
                 'support_names': support_names,  # List of strings
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_id': torch.tensor(class_sample)}  # Scalar int tensor

        return batch

    def build_class_ids(self):  # <--- This method exists later in the class
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def extract_ignore_idx(self, mask, class_id):
        # class_id is 0-19. PASCAL masks have original values 1-20 for classes.
        # boundary is 255 in original mask.
        boundary = (mask / 255).floor()  # if original mask had 255, this becomes 1.
        mask[mask != class_id + 1] = 0  # Target class (e.g. class_id=0, actual val 1) becomes 1
        mask[mask == class_id + 1] = 1
        return mask, boundary  # mask is now binary {0,1}

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)  # PIL Image for mask
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]
        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # Mask is loaded as tensor directly in original DatasetPASCAL.
        # This is fine, as F.interpolate needs tensor.
        return torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg').convert('RGB')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]
        support_names = []
        # Ensure K_shot unique support samples that are different from query
        available_supports = [name for name in self.img_metadata_classwise[class_sample] if name != query_name]
        if len(available_supports) < self.shot:
            # Fallback: sample with replacement if not enough unique images, still excluding query
            # This case should be rare if dataset is rich enough for k-shot.
            support_pool = self.img_metadata_classwise[class_sample][:]
            if query_name in support_pool: support_pool.remove(query_name)  # Ensure query is not in pool
            if not support_pool:  # If only query exists for this class (highly unlikely)
                # This is a problematic case for few-shot learning.
                # For now, let's just pick from general pool if this happens, but log it.
                # Or, if this is hit, it implies an issue with dataset structure for k-shot.
                # A robust way might be to re-sample the episode or skip.
                # For now, assume img_metadata_classwise has enough.
                # The original DatasetPASCAL code's while loop might get stuck if not enough unique supports.
                logger.warning(
                    f"Critically low unique support samples for class {class_sample}, query {query_name}. Sampling with replacement from non-query pool.")
                if not support_pool:  # If class only has one image (the query)
                    raise ValueError(
                        f"Cannot sample {self.shot} support images for class {class_sample}, query {query_name}, as it's the only image.")
                support_names = list(np.random.choice(support_pool, self.shot, replace=True))

            else:
                support_names = list(
                    np.random.choice(available_supports, self.shot, replace=len(available_supports) < self.shot))
        else:
            support_names = list(np.random.choice(available_supports, self.shot, replace=False))

        return query_name, support_names, class_sample

    # UPDATED: Pass datapath to build_img_metadata to ensure splits path is correct
    def build_img_metadata(self, datapath):
        def read_metadata(split, fold_id):
            # IMPORTANT: The user needs to ensure this path 'data/splits/folds/...' exists
            # and contains the split files.
            # For robustness, one might pass this 'split_file_dir' as an argument.
            # UPDATED: Use datapath for splits
            split_file_dir = os.path.join(datapath, "splits",
                                          "folds")  # Heuristic path based on VOCdevkit/VOC2012 being datapath
            fold_n_metadata_path = os.path.join(split_file_dir, '%s' % split, 'fold%d.txt' % fold_id)

            if not os.path.exists(fold_n_metadata_path):
                raise FileNotFoundError(f"PASCAL split file not found: {fold_n_metadata_path}. "
                                        f"Please ensure 'splits/folds/' directory is correctly "
                                        f"structured relative to your PASCAL VOC data path, or adjust path in DatasetPASCAL.")

            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata if data]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        if not img_metadata:
            raise ValueError(f"No image metadata loaded for split '{self.split}', fold {self.fold}. Check split files.")

        logger.info(
            'PASCAL Dataset: Total (%s) images for fold %d are : %d' % (self.split, self.fold, len(img_metadata)))
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):  # 0-19
            img_metadata_classwise[class_id] = []
        for img_name, img_class in self.img_metadata:  # img_class is 0-19
            img_metadata_classwise[img_class].append(img_name)
        return img_metadata_classwise


# --- END PASCAL-5i Dataset ---


# --- PASCAL VOC Specifics ---
PASCAL_CLASS_NAMES_LIST = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# --- END PASCAL VOC Specifics ---


# --- Setup Logging ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    stream=sys.stdout,
                    force=True)
logger = logging.getLogger(__name__)


# --- END Logging ---


# --- Adapter Dataset for PASCAL-5i ---
class PASCALAdapterDataset(Dataset):
    def __init__(self, pascal_dataset_instance, target_image_size_int, class_names_list):
        self.pascal_dataset = pascal_dataset_instance
        self.target_image_size_tuple = (target_image_size_int, target_image_size_int)
        self.class_names_list = class_names_list
        logger.info(
            f"PASCALAdapterDataset initialized. Wraps a PASCAL dataset with {len(self.pascal_dataset)} base items.")
        logger.info(
            f"Adapter will ensure output image/masks are ({target_image_size_int},{target_image_size_int}) and query GT is uint8.")

    def __len__(self):
        return len(self.pascal_dataset)

    def __getitem__(self, idx):
        try:
            raw_item = self.pascal_dataset[idx]

            support_set_out = []
            support_imgs_tensor = raw_item['support_imgs']  # (K, C, H, W)
            support_masks_tensor = raw_item['support_masks']  # (K, H, W), float, 0.0 or 1.0

            k_shot = support_imgs_tensor.shape[0]
            for i in range(k_shot):
                s_img = support_imgs_tensor[i]
                s_mask = support_masks_tensor[i]  # Already float (H,W)

                # Ensure correct size (should be handled by DatasetPASCAL's transform and F.interpolate)
                if s_img.shape[1:] != self.target_image_size_tuple:
                    # This should not happen if DatasetPASCAL's transform is set correctly
                    # and use_original_imgsize=False
                    logger.warning(f"Support image size mismatch: {s_img.shape[1:]} vs {self.target_image_size_tuple}")
                    s_img = TF.resize(s_img, list(self.target_image_size_tuple),
                                      interpolation=T.InterpolationMode.BILINEAR)
                if s_mask.shape != self.target_image_size_tuple:
                    logger.warning(f"Support mask size mismatch: {s_mask.shape} vs {self.target_image_size_tuple}")
                    s_mask = TF.resize(s_mask.unsqueeze(0), list(self.target_image_size_tuple),
                                       interpolation=T.InterpolationMode.NEAREST).squeeze(0)

                support_set_out.append({'image': s_img, 'mask': s_mask})

            q_img_tensor = raw_item['query_img']  # (C, H, W)
            q_mask_tensor = raw_item['query_mask']  # (H, W), float, 0.0 or 1.0

            if q_img_tensor.shape[1:] != self.target_image_size_tuple:
                logger.warning(f"Query image size mismatch: {q_img_tensor.shape[1:]} vs {self.target_image_size_tuple}")
                q_img_tensor = TF.resize(q_img_tensor, list(self.target_image_size_tuple),
                                         interpolation=T.InterpolationMode.BILINEAR)
            if q_mask_tensor.shape != self.target_image_size_tuple:
                logger.warning(f"Query mask size mismatch: {q_mask_tensor.shape} vs {self.target_image_size_tuple}")
                q_mask_tensor = TF.resize(q_mask_tensor.unsqueeze(0), list(self.target_image_size_tuple),
                                          interpolation=T.InterpolationMode.NEAREST).squeeze(0)

            q_mask_np_uint8 = q_mask_tensor.cpu().numpy().astype(np.uint8)

            class_id_int = raw_item['class_id'].item()
            query_class_str = self.class_names_list[class_id_int]
            query_name_str = raw_item['query_name']

            query_set_out = {
                'image_resnet': q_img_tensor,
                'gt_mask_np': q_mask_np_uint8,
                'class': query_class_str,
                'query_index': query_name_str,
                'pascal_class_id': class_id_int,  # Optional, for debugging
                'original_imsize': raw_item['org_query_imsize']  # Optional
            }
            return support_set_out, query_set_out
        except Exception as e:
            logger.error(
                f"Error in PASCALAdapterDataset for index {idx} (Query: {raw_item.get('query_name', 'N/A') if 'raw_item' in locals() else 'N/A'}): {e}",
                exc_info=True)
            return None  # For collate_fn


# --- END Adapter ---



# --- Configuration ---
# DEFAULT_FSS_PATH = '/home/farahani/khan/fewshot_data/fewshot_data' # Old
DEFAULT_PASCAL_DATAPATH = r"C:\Users\Khan\PycharmProjects\FSS-Research-\data\pascal\raw"  # NEW: Root for VOC2012, SegmentationClassAug, splits
DEFAULT_PRECOMPUTED_MASK_PATH = r'C:\Users\Khan\Desktop\New folder (2)\sam_precomputed_masks_hybrid\pascal_voc2012_sam_masks'  # NEW: Consider separate SAM masks for PASCAL
# DEFAULT_PRECOMPUTED_MASK_PATH = r'C:\Users\Khan\PycharmProjects\FSS-Research-\pascal\pascal_sam_precomputed_masks_hybrid\JPEGImages'  # NEW: Consider separate SAM masks for PASCAL

# DEFAULT_PRECOMPUTED_GRAPH_PATH = './precomputed_pascal_graphs_hybrid_v1_main_512'
DEFAULT_PRECOMPUTED_GRAPH_PATH = './precomputed_pascal_graphs_hybrid_v1_main'
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_K_SHOT = 1
DEFAULT_IMG_SIZE = 256  # This will be used for PASCAL image/mask resizing
DEFAULT_FEATURE_SIZE = 2048
DEFAULT_MAX_MASKS_GRAPH = 15
DEFAULT_GRAPH_CONNECTIVITY = 'knn'
DEFAULT_KNN_K = 10
DEFAULT_OVERLAP_THRESH = 0.1
DEFAULT_EDGE_FEATURE_MODE = 'iou_dist_area'
DEFAULT_PASCAL_FOLD = 0  # NEW: For PASCAL-5i fold (0, 1, 2, 3)
DEFAULT_PASCAL_USE_ORIGINAL_IMGSIZE = False  # NEW: Must be False for consistency with fixed IMG_SIZE and SAM


# --- Utility Functions (calculate_iou, combine_masks, get_geometric_features, _run_greedy_oracle_for_labels) ---
# These functions remain unchanged from your original script.
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
            # logger.warning(f"Combine_masks: resizing mask from {mask.shape} to {ref_shape}")
            # This should ideally not happen if adapter prepares masks correctly.
            # For safety, one could resize here, but it's better to fix upstream.
            # mask_pil = Image.fromarray(mask.astype(np.uint8)*255)
            # mask_resized_pil = mask_pil.resize(ref_shape[::-1], Image.NEAREST) # W, H for PIL resize
            # mask = (np.array(mask_resized_pil) > 128).astype(bool)
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
        # logger.debug(f"Exception in get_geometric_features: {e}") # Can be noisy
        return np.zeros(4, dtype=np.float32)


# --- FSS-1000 Data Handling (Unchanged, but will NOT be used if PASCAL is primary) ---
class FSSDataset(Dataset):  # This class remains for now, but won't be used by main flow.
    def __init__(self, base_path, image_size=DEFAULT_IMG_SIZE, k_shot=DEFAULT_K_SHOT):
        self.base_path = base_path
        self.image_size = image_size
        self.k_shot = k_shot
        # ... (rest of FSSDataset implementation is unchanged)
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"FSS-1000 dataset not found at: {base_path}")
        try:
            all_classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not all_classes:
                raise ValueError(f"No class directories found in {base_path}")
            self.class_files = {}
            self.all_samples = []
            for cls in sorted(all_classes):
                cls_path = os.path.join(base_path, cls)
                files = sorted([f.split('.')[0] for f in os.listdir(cls_path) if f.endswith('.jpg')])
                valid_files = []
                for file_id in files:
                    img_path = os.path.join(cls_path, f"{file_id}.jpg")
                    mask_path = os.path.join(cls_path, f"{file_id}.png")
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        valid_files.append(file_id)
                if len(valid_files) >= k_shot + 1:  # Need at least k support + 1 query
                    self.class_files[cls] = valid_files
                    for file_id in valid_files:  # All valid files can be queries
                        self.all_samples.append((cls, file_id))
            self.classes = sorted(list(self.class_files.keys()))
            if not self.classes:
                raise ValueError(f"No classes with enough valid samples (k+1={k_shot + 1}) found in {base_path}")
            logger.info(
                f"Initialized FSSDataset: {len(self.all_samples)} total samples across {len(self.classes)} valid classes from {base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize FSSDataset: {e}", exc_info=True)
            raise

        self.resnet_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.support_mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.gt_mask_transform_pil = T.Compose([  # For query GT mask to numpy
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        query_cls, query_file_id = self.all_samples[idx]
        available_files = self.class_files[query_cls][:]
        try:
            available_files.remove(query_file_id)
        except ValueError:
            logger.error(f"Query file ID {query_file_id} not found for class {query_cls}. This is unexpected.")
            if len(available_files) < self.k_shot: return None

        if len(available_files) < self.k_shot:
            support_indices = random.choices(available_files, k=self.k_shot)
        else:
            support_indices = random.sample(available_files, self.k_shot)

        support_set_out = []
        query_set_out = {}
        try:
            for support_id in support_indices:
                img_path = os.path.join(self.base_path, query_cls, f"{support_id}.jpg")
                mask_path = os.path.join(self.base_path, query_cls, f"{support_id}.png")
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                image_tensor = self.resnet_transform(image)
                mask_tensor = self.support_mask_transform(mask)
                mask_tensor = (mask_tensor > 0.5).float()
                support_set_out.append({'image': image_tensor, 'mask': mask_tensor})

            q_img_path = os.path.join(self.base_path, query_cls, f"{query_file_id}.jpg")
            q_mask_path = os.path.join(self.base_path, query_cls, f"{query_file_id}.png")
            q_image = Image.open(q_img_path).convert("RGB")
            q_mask = Image.open(q_mask_path).convert("L")
            q_image_tensor = self.resnet_transform(q_image)
            q_mask_gt_pil = self.gt_mask_transform_pil(q_mask)
            q_mask_gt_np = np.array(q_mask_gt_pil)
            threshold = 128 if q_mask_gt_np.max() > 1 else 0.5
            q_mask_gt_np = (q_mask_gt_np > threshold).astype(np.uint8)

            query_set_out = {
                'image_resnet': q_image_tensor,
                'gt_mask_np': q_mask_gt_np,
                'class': query_cls,
                'query_index': query_file_id
            }
            return support_set_out, query_set_out
        except FileNotFoundError as e:
            logger.warning(f"File not found for item {idx} (Cls {query_cls}, Q {query_file_id}): {e}. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error loading item {idx} (Cls: {query_cls}, Q: {query_file_id}): {e}", exc_info=True)
            return None


# --- Feature Extractor (Unchanged, used only for precomputation) ---
class FeatureExtractor(nn.Module):
    def __init__(self, device=DEFAULT_DEVICE):
        super().__init__()
        logger.info("Initializing ResNet-50 feature extractor...")
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
            self.resnet_layer4 = nn.Sequential(*list(resnet.children())[:-2])  # Output is (B, 2048, H/32, W/32)
            self.resnet_layer4.eval()
            self.device = device
            self.resnet_layer4.to(device)
            self.feature_dim = 2048  # ResNet-50 layer 4 output channels
            logger.info(f"ResNet-50 layer4 loaded successfully on device '{device}'. Feature dim: {self.feature_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize ResNet-50: {e}", exc_info=True)
            raise

    @torch.no_grad()
    def get_feature_map(self, image_tensor_batch):  # Expects (B, C, H, W) tensor
        if image_tensor_batch.dim() == 3:
            image_tensor_batch = image_tensor_batch.unsqueeze(0)
        image_tensor_batch = image_tensor_batch.to(self.device)
        try:
            feature_map = self.resnet_layer4(image_tensor_batch)
            return feature_map  # (B, 2048, H_fm, W_fm)
        except Exception as e:
            logger.error(f"Error during ResNet forward pass (get_feature_map): {e}", exc_info=True)
            return None

    @torch.no_grad()
    def get_masked_features(self, feature_map,
                            mask_tensor):  # feature_map (B, D, Hf, Wf), mask_tensor (B, 1, Hm, Wm) or (B,Hm,Wm)
        if feature_map is None: return None
        if mask_tensor is None or mask_tensor.sum() < 1e-6:  # If mask is empty
            fm_batch_size = feature_map.shape[0] if feature_map.dim() > 3 else 1
            return torch.zeros((fm_batch_size, self.feature_dim), device=self.device)

        mask_tensor = mask_tensor.to(self.device)
        feature_map = feature_map.to(self.device)

        if feature_map.dim() == 3: feature_map = feature_map.unsqueeze(0)  # B=1 case

        # Adjust mask_tensor dimensions: needs to be (B, 1, Hm, Wm) for TF.resize and expand_as
        if mask_tensor.dim() == 2:  # (H, W) -> (1, 1, H, W)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3 and mask_tensor.shape[0] != feature_map.shape[
            0]:  # (H, W) but passed as (B,H,W) from support items
            mask_tensor = mask_tensor.unsqueeze(1)  # (B,1,H,W) assuming B matches FM
        elif mask_tensor.dim() == 3 and mask_tensor.shape[0] == feature_map.shape[
            0]:  # (B,H,W) -> (B,1,H,W) (e.g. batch of support masks)
            mask_tensor = mask_tensor.unsqueeze(1)

        fm_batch_size, num_features, hf, wf = feature_map.shape
        mask_batch_size, _, hm, wm = mask_tensor.shape

        if fm_batch_size == 1 and mask_batch_size > 1:  # One FM, multiple masks (e.g. query FM, N candidate masks)
            try:
                feature_map_expanded = feature_map.expand(mask_batch_size, -1, -1, -1)
            except Exception as e:
                logger.error(f"Error expanding feature map in N-to-1 case: {e}", exc_info=True)
                return torch.zeros((mask_batch_size, num_features), device=self.device)
        else:  # One FM, one mask (e.g. support item) or Batch FM, Batch Masks
            feature_map_expanded = feature_map

        if feature_map_expanded.shape[0] != mask_batch_size:
            logger.error(
                f"Batch size mismatch: Expanded FM ({feature_map_expanded.shape[0]}) vs Mask ({mask_batch_size}). Returning zeros.")
            return torch.zeros((mask_batch_size, num_features), device=self.device)

        try:
            if hm <= 0 or wm <= 0 or hf <= 0 or wf <= 0:
                logger.error(f"Invalid dims for resize: Mask ({hm},{wm}), FM ({hf},{wf}). Zeros.")
                return torch.zeros((mask_batch_size, num_features), device=self.device)
            mask_resized = TF.resize(mask_tensor.float(), (hf, wf), interpolation=T.InterpolationMode.NEAREST)
            mask_resized = (mask_resized > 0.5).float()  # Ensure binary after resize
        except Exception as e:
            logger.error(f"Error resizing mask in get_masked_features: {e}", exc_info=True)
            return torch.zeros((mask_batch_size, num_features), device=self.device)

        try:
            mask_expanded_for_pool = mask_resized.expand_as(feature_map_expanded)  # (B, D, Hf, Wf)
            masked_fm = feature_map_expanded * mask_expanded_for_pool
            sum_features = masked_fm.sum(dim=(2, 3))  # (B, D)
            mask_count = mask_resized.sum(dim=(2, 3))  # (B, 1)
            mask_count = torch.clamp(mask_count, min=1e-8)
            avg_features = sum_features / mask_count
            avg_features = torch.nan_to_num(avg_features, nan=0.0, posinf=0.0, neginf=0.0)
            return avg_features  # (B, D)
        except Exception as e:
            logger.error(f"Error during masked average pooling: {e}", exc_info=True)
            return torch.zeros((mask_batch_size, num_features), device=self.device)


# --- RENAMED: Graph Construction (for precomputation) ---
# (Unchanged internally, but how it gets support_set/query_set will adapt via the adapter)
@torch.no_grad()
def _construct_graph_data_for_precomputation(support_set, query_set,
                                             # support_set is still passed, but not used for support features in graph save
                                             precomputed_mask_path,
                                             feature_extractor, device,
                                             # feature_extractor now only for query FM/node feats
                                             max_masks=DEFAULT_MAX_MASKS_GRAPH,
                                             connectivity=DEFAULT_GRAPH_CONNECTIVITY,
                                             knn_k=DEFAULT_KNN_K,
                                             overlap_thresh=DEFAULT_OVERLAP_THRESH,
                                             edge_feature_mode='iou_dist_area',
                                             expected_edge_dim=3):
    query_class = query_set.get('class', 'N/A')  # Will be string from PASCALAdapterDataset
    query_idx = query_set.get('query_index', 'N/A')  # Will be string ID from PASCALAdapterDataset
    log_prefix = f"[GraphConstruct Cls {query_class}, Query {query_idx}]:"

    # 1. Extract Individual Support Foreground Features - REMOVED from precomputation
    # This block is removed. Support features will be computed dynamically during training/evaluation.

    # 2. Load Precomputed Candidate Masks for Query Image
    query_gt_mask_np = query_set.get('gt_mask_np')  # This is (H,W) uint8 from adapter
    query_img_resnet = query_set.get('image_resnet')  # This is (C,H,W) tensor from adapter
    if query_gt_mask_np is None or query_img_resnet is None:
        logger.error(f"{log_prefix} Missing required query data (gt_mask_np or image_resnet).")
        return None
    query_gt_mask_bool = query_gt_mask_np.astype(bool)
    ref_shape = query_gt_mask_bool.shape
    if ref_shape != (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE):
        logger.error(
            f"{log_prefix} Incorrect query GT mask shape {ref_shape}. Expected ({DEFAULT_IMG_SIZE}, {DEFAULT_IMG_SIZE})")
        # This might indicate PASCALAdapterDataset or transforms are not resizing correctly.
        return None

    candidate_masks_data_raw = []
    masks_np_list_bool = []
    original_sam_data_for_nodes = []
    try:
        # Path uses query_class (string name) and query_idx (string ID) from adapter
        # UPDATED: Use precomputed_mask_path directly from args
        mask_file_path = Path(precomputed_mask_path) / f"{query_idx}_sam_hybrid_masks.pkl"
        if not mask_file_path.is_file():
            logger.warning(f"{log_prefix} Precomputed SAM mask file NOT FOUND: {mask_file_path}")
            return None
        with open(mask_file_path, 'rb') as f:
            sam_output = pickle.load(f)

        sam_output.sort(key=lambda x: x['area'], reverse=True)
        candidate_masks_data_raw = sam_output[:max_masks]

        valid_masks_for_graph = []
        valid_sam_data_for_graph = []
        for m_data in candidate_masks_data_raw:
            mask_np = m_data['segmentation']
            if mask_np.shape == ref_shape:
                valid_masks_for_graph.append(mask_np.astype(bool))
                valid_sam_data_for_graph.append(m_data)
        masks_np_list_bool = valid_masks_for_graph
        original_sam_data_for_nodes = valid_sam_data_for_graph
    except Exception as e:
        logger.error(f"{log_prefix} Error loading/processing precomputed SAM masks: {e}", exc_info=False)
        return None

    num_nodes = len(masks_np_list_bool)
    if num_nodes == 0:
        logger.info(f"{log_prefix} No valid candidate masks found after filtering. Cannot build graph.")
        return None

    # 3. Pre-calculate Query Feature Map
    try:
        query_feature_map = feature_extractor.get_feature_map(query_img_resnet.unsqueeze(0))
        if query_feature_map is None: raise ValueError("Query feature map is None")
    except Exception as e:
        logger.error(f"{log_prefix} Failed to get query feature map: {e}", exc_info=False)
        return None

    # 4. Extract Node Feature Components
    try:
        masks_stacked_np = np.stack(masks_np_list_bool, axis=0)  # (N, H, W)
        masks_tensor = torch.from_numpy(masks_stacked_np).unsqueeze(1).float()  # (N, 1, H, W)
        # get_masked_features expects masks as (B,1,H,W) or (B,H,W) which it then unsqueezes to (B,1,H,W)
        # Here, masks_tensor is (N,1,H,W) and query_feature_map is (1,D,Hf,Wf)
        # get_masked_features handles the B=1 FM and N>1 masks case.
        x_base_features = feature_extractor.get_masked_features(query_feature_map, masks_tensor)
        if x_base_features is None or x_base_features.shape[0] != num_nodes:
            logger.error(
                f"{log_prefix} Failed to extract base visual features for nodes. Got shape {x_base_features.shape if x_base_features is not None else 'None'}")
            return None
    except Exception as e:
        logger.error(f"{log_prefix} Error in batch visual feature extraction: {e}", exc_info=False)
        return None

    x_geo_features_list = []
    x_sam_features_list = []
    node_centroids = []
    for i in range(num_nodes):
        mask_np_bool_node = masks_np_list_bool[i]
        sam_data_node = original_sam_data_for_nodes[i]
        geo_feats_np = get_geometric_features(mask_np_bool_node)
        x_geo_features_list.append(torch.from_numpy(geo_feats_np))
        sam_iou_conf = sam_data_node.get('predicted_iou', 0.0)
        sam_stability = sam_data_node.get('stability_score', 0.0)
        x_sam_features_list.append(torch.tensor([sam_iou_conf, sam_stability], dtype=torch.float32))
        try:
            props = regionprops(mask_np_bool_node.astype(np.uint8))
            node_centroids.append(props[0].centroid if props else (ref_shape[0] / 2.0, ref_shape[1] / 2.0))
        except:
            node_centroids.append((ref_shape[0] / 2.0, ref_shape[1] / 2.0))

    x_geo_features = torch.stack(x_geo_features_list).to(dtype=torch.float32)
    x_sam_features = torch.stack(x_sam_features_list).to(dtype=torch.float32)

    # 5. Determine Node Labels (Greedy Oracle)
    target_labels = [0.0] * num_nodes
    selected_by_oracle_indices, _ = _run_greedy_oracle_for_labels(
        masks_np_list_bool, query_gt_mask_bool, ref_shape
    )
    for i in selected_by_oracle_indices: target_labels[i] = 1.0
    target_labels_tensor = torch.tensor(target_labels, dtype=torch.float)

    # 6. Define Graph Edges and Edge Features
    # ... (rest of graph construction is unchanged) ...
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr_list = []
    adj_matrix_for_edges_np = None

    if num_nodes > 1:
        if connectivity == 'fully_connected':
            adj_dense_np = np.ones((num_nodes, num_nodes), dtype=np.uint8) - np.eye(num_nodes, dtype=np.uint8)
            rows, cols = np.where(adj_dense_np > 0)
            adj_matrix_for_edges_np = np.vstack((rows, cols))
        elif connectivity == 'knn' and node_centroids:
            try:
                node_centroids_np = np.array(node_centroids)
                k_actual = min(knn_k, num_nodes - 1)
                if k_actual > 0:
                    adj_sparse = kneighbors_graph(node_centroids_np, k_actual, mode='connectivity', include_self=False,
                                                  n_jobs=1)
                    adj_sparse_symmetric = adj_sparse + adj_sparse.T
                    adj_sparse_symmetric[adj_sparse_symmetric > 1] = 1
                    coo = adj_sparse_symmetric.tocoo()
                    adj_matrix_for_edges_np = np.vstack((coo.row, coo.col))
            except Exception as e_knn:
                logger.warning(f"{log_prefix} Error building KNN graph: {e_knn}. No edges will be added.")
        elif connectivity == 'overlap':
            rows_overlap, cols_overlap = [], []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    overlap_iou_val = calculate_iou(masks_np_list_bool[i], masks_np_list_bool[j])
                    if overlap_iou_val > overlap_thresh:
                        rows_overlap.extend([i, j]);
                        cols_overlap.extend([j, i])
            if rows_overlap:
                adj_matrix_for_edges_np = np.vstack((rows_overlap, cols_overlap))

    if adj_matrix_for_edges_np is not None and adj_matrix_for_edges_np.shape[1] > 0:
        edge_index = torch.from_numpy(adj_matrix_for_edges_np).long()
        for src_idx, trg_idx in edge_index.t().tolist():
            mask1_node = masks_np_list_bool[src_idx];
            mask2_node = masks_np_list_bool[trg_idx]
            current_edge_feats = []
            if edge_feature_mode == 'iou_dist_area':
                inter_mask_iou = calculate_iou(mask1_node, mask2_node)
                c1 = np.array(node_centroids[src_idx]);
                c2 = np.array(node_centroids[trg_idx])
                dist = np.linalg.norm(c1 - c2)
                max_dist = np.sqrt(ref_shape[0] ** 2 + ref_shape[1] ** 2)
                norm_dist = dist / (max_dist + 1e-6)
                area1 = mask1_node.sum();
                area2 = mask2_node.sum()
                area_ratio = min(area1, area2) / (max(area1, area2) + 1e-6) if max(area1, area2) > 0 else 0.0
                current_edge_feats = [inter_mask_iou, norm_dist, area_ratio]
            elif edge_feature_mode == 'iou_only':
                current_edge_feats = [calculate_iou(mask1_node, mask2_node)]
            else:
                current_edge_feats = [calculate_iou(mask1_node, mask2_node)]
                while len(current_edge_feats) < expected_edge_dim: current_edge_feats.append(0.0)
                current_edge_feats = current_edge_feats[:expected_edge_dim]
            edge_attr_list.append(torch.tensor(current_edge_feats, dtype=torch.float32))

    edge_attr = torch.stack(edge_attr_list) if edge_attr_list else torch.empty((0, expected_edge_dim),
                                                                               dtype=torch.float32)

    if edge_index.numel() > 0 and edge_attr.shape[0] != edge_index.shape[1]:
        logger.error(
            f"{log_prefix} Mismatch: edge_attr rows ({edge_attr.shape[0]}) vs edge_index cols ({edge_index.shape[1]}). Clearing edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, expected_edge_dim), dtype=torch.float32)

    # 7. Create PyG Data Object (all tensors should be CPU for saving)
    try:
        data = Data(
            x_base=x_base_features.cpu(),
            x_geo=x_geo_features.cpu(),
            x_sam=x_sam_features.cpu(),
            # support_k_shot_features=support_k_shot_features.cpu(), # <-- REMOVED
            edge_index=edge_index.cpu(),
            edge_attr=edge_attr.cpu(),
            y=target_labels_tensor.cpu(),
            node_masks_np=[m.astype(np.uint8) for m in masks_np_list_bool],
            gt_mask_np=query_gt_mask_np,
            image_size=(query_gt_mask_np.shape[0], query_gt_mask_np.shape[1]),
            query_class=query_class,
            query_index=query_idx,
            num_nodes=num_nodes
        )
        if not (data.x_base.shape[0] == data.x_geo.shape[0] == data.x_sam.shape[0] == data.y.shape[
            0] == data.num_nodes):
            raise ValueError(f"Inconsistent node counts in Data obj components for {query_class}/{query_idx}.")
        if data.edge_index.numel() > 0 and data.edge_attr.shape[0] != data.edge_index.shape[1]:
            raise ValueError(
                f"Edge_attr row count ({data.edge_attr.shape[0]}) != edge_index column count ({data.edge_index.shape[1]}) for {query_class}/{query_idx}")
        if data.edge_attr.numel() > 0 and data.edge_attr.shape[1] != expected_edge_dim:
            raise ValueError(
                f"Edge_attr feature dim ({data.edge_attr.shape[1]}) != expected ({expected_edge_dim}) for {query_class}/{query_idx}")
        # if data.support_k_shot_features.shape[1] != feature_extractor.feature_dim and data.support_k_shot_features.numel() > 0 : # <-- REMOVED
        #     raise ValueError(f"Support features dim mismatch for {query_class}/{query_idx}")
        return data
    except Exception as e:
        logger.error(f"{log_prefix} Error creating final PyG Data object: {e}", exc_info=True)
        return None


# --- NEW: Function to load precomputed graph data --- (Unchanged)
def load_precomputed_graph_data(query_class, query_idx, precomputed_graph_dir):
    # UPDATED: Path uses query_idx directly, as class name is implicitly part of dir structure from saving
    graph_file_path = Path(precomputed_graph_dir) / query_class / f"{query_idx}_graph.pt"
    if not graph_file_path.is_file():
        # Fallback for older structure if precomputed_graph_dir already contains class subdir
        graph_file_path_fallback = Path(precomputed_graph_dir) / f"{query_idx}_graph.pt"
        if not graph_file_path_fallback.is_file():
            return None
        else:
            graph_file_path = graph_file_path_fallback

    try:
        data = torch.load(graph_file_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        logger.error(f"Error loading precomputed graph {graph_file_path}: {e}", exc_info=True)
        return None


# --- NEW: Function for orchestrating graph precomputation --- (Dataset init will change)
def precompute_graphs_entrypoint(dataset_adapter, indices, args, feature_extractor, device):  # Takes adapter now
    logger.info(f"Starting graph precomputation for {len(indices)} samples...")
    logger.info(f"Graphs will be saved to: {args.precomputed_graph_path}")
    Path(args.precomputed_graph_path).mkdir(parents=True, exist_ok=True)

    # Use a sequential loader with the adapter dataset.
    # `dataset_adapter` is an instance of PASCALAdapterDataset. `indices` are for this adapter.
    # Collate_fn now correctly passes support_set and query_set
    loader = PyTorchDataLoader(dataset_adapter, batch_size=1, sampler=SequentialSampler(indices),
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               pin_memory=False)

    processed_count = 0
    saved_count = 0
    skipped_construction_count = 0
    skipped_existing_count = 0

    pbar = tqdm(loader, desc="Precomputing Graphs")
    for batch_items in pbar:
        if batch_items is None or not batch_items:
            skipped_construction_count += 1
            continue

        # Batch_items is a list of (support_set, query_set) tuples. Since batch_size=1, it's [ (s,q) ]
        support_set, query_set = batch_items[0]

        query_class = query_set.get('class')
        query_idx = query_set.get('query_index')

        if not query_class or not query_idx:
            logger.warning("Missing query_class or query_idx in dataset item during precomputation. Skipping.")
            skipped_construction_count += 1
            continue

        class_graph_dir = Path(args.precomputed_graph_path) / query_class  # Save to class-specific subdir
        class_graph_dir.mkdir(parents=True, exist_ok=True)
        graph_file_path = class_graph_dir / f"{query_idx}_graph.pt"

        processed_count += 1
        if not args.overwrite_precomputed_graphs and graph_file_path.exists():
            skipped_existing_count += 1
        else:
            # Pass support_set here, but _construct_graph_data_for_precomputation will ignore its features for saving
            graph_data = _construct_graph_data_for_precomputation(
                support_set, query_set,  # support_set is still passed here
                precomputed_mask_path=args.precomputed_mask_path,
                feature_extractor=feature_extractor,
                device=device,
                max_masks=args.max_masks_graph,
                connectivity=args.graph_connectivity,
                knn_k=args.knn_k,
                overlap_thresh=args.overlap_thresh,
                edge_feature_mode=args.edge_feature_mode,
                expected_edge_dim=args.edge_feature_dim
            )

            if graph_data is not None and graph_data.num_nodes > 0:
                try:
                    torch.save(graph_data, graph_file_path)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving graph for {query_class}/{query_idx}: {e}", exc_info=True)
                    skipped_construction_count += 1
            else:
                skipped_construction_count += 1

        pbar.set_postfix({
            'saved': saved_count,
            'skip_constr': skipped_construction_count,
            'skip_exist': skipped_existing_count,
            'total': processed_count
        })

    logger.info(f"Graph precomputation finished.")
    logger.info(f"Total samples processed: {processed_count}")
    logger.info(f"Graphs successfully saved: {saved_count}")
    logger.info(f"Graphs skipped (failed construction/save): {skipped_construction_count}")
    logger.info(f"Graphs skipped (already existed, no overwrite): {skipped_existing_count}")


# --- GNN Model Definitions (Unchanged) ---
class SupportCrossAttention(nn.Module):
    def __init__(self, query_dim, support_feature_dim, num_heads, output_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(support_feature_dim, output_dim)
        self.value_proj = nn.Linear(support_feature_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query_node_features,
                support_features_k_shot):  # <--- support_features_k_shot is a direct argument
        num_nodes = query_node_features.shape[0]
        k_shot = support_features_k_shot.shape[0]

        if k_shot == 0:
            return torch.zeros(num_nodes, self.num_heads * self.head_dim, device=query_node_features.device)

        q = self.query_proj(query_node_features).reshape(num_nodes, self.num_heads, self.head_dim)
        k = self.key_proj(support_features_k_shot).reshape(k_shot, self.num_heads, self.head_dim)
        v = self.value_proj(support_features_k_shot).reshape(k_shot, self.num_heads, self.head_dim)

        scores = torch.einsum("nhd,mhd->nhm", q, k) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.einsum("nhm,mhd->nhd", attn_weights, v)
        context = context.reshape(num_nodes, self.num_heads * self.head_dim)
        return self.out_proj(context)


class SupportAwareMaskGNN(nn.Module):
    def __init__(self, base_node_visual_dim, geometric_dim, sam_meta_dim,
                 support_feature_dim, spt_attn_num_heads, spt_attn_output_dim,
                 gnn_hidden_dim, num_gnn_layers, num_heads_gnn, edge_feature_dim,
                 dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.spt_cross_attention = SupportCrossAttention(
            query_dim=base_node_visual_dim,
            support_feature_dim=support_feature_dim,
            num_heads=spt_attn_num_heads,
            output_dim=spt_attn_output_dim,
            dropout=dropout_rate
        )
        self.input_gnn_dim = base_node_visual_dim + geometric_dim + sam_meta_dim + spt_attn_output_dim
        self.input_norm = nn.LayerNorm(self.input_gnn_dim)
        self.input_lin = nn.Linear(self.input_gnn_dim, gnn_hidden_dim)
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        self.gnn_acts = nn.ModuleList()
        current_dim = gnn_hidden_dim
        for _ in range(num_gnn_layers):
            conv = TransformerConv(
                in_channels=current_dim,
                out_channels=gnn_hidden_dim // num_heads_gnn,
                heads=num_heads_gnn,
                concat=True,
                dropout=dropout_rate,
                edge_dim=edge_feature_dim
            )
            self.gnn_layers.append(conv)
            self.gnn_norms.append(nn.LayerNorm(gnn_hidden_dim))
            self.gnn_acts.append(nn.ReLU())
            current_dim = gnn_hidden_dim
        self.readout_mlp = nn.Sequential(
            nn.LayerNorm(gnn_hidden_dim),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gnn_hidden_dim // 2, 1)
        )
        logger.info(
            f"SupportAwareMaskGNN initialized. GNN Input Dim: {self.input_gnn_dim}, GNN Hidden: {gnn_hidden_dim}, GNN Layers: {num_gnn_layers}, Edge Dim: {edge_feature_dim}")

    # UPDATED: support_k_shot_features is now passed directly as an argument
    def forward(self, data, support_k_shot_features):
        x_base, x_geo, x_sam, edge_index, edge_attr = \
            data.x_base, data.x_geo, data.x_sam, \
                data.edge_index, data.edge_attr  # support_k_shot_features is NOT from data object

        spt_conditioned_feats = self.spt_cross_attention(x_base, support_k_shot_features)
        x = torch.cat([x_base, x_geo, x_sam, spt_conditioned_feats], dim=-1)
        x = self.input_norm(x)
        x = self.input_lin(x)
        x = self.input_act(x)
        x = self.input_dropout(x)
        x_res = x
        for i, layer in enumerate(self.gnn_layers):
            current_edge_attr = edge_attr if edge_index.numel() > 0 and edge_attr is not None and edge_attr.numel() > 0 else None
            x_new = layer(x_res, edge_index, edge_attr=current_edge_attr)
            x_new = self.gnn_norms[i](x_new)
            x = x_res + x_new
            x = self.gnn_acts[i](x)
            x_res = x
        x = self.readout_mlp(x)
        return x.squeeze(-1)

    # --- Custom Collate Function (Unchanged) ---


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return batch


# --- Helper Function for Weight Estimation (Dataset init will change) ---
@torch.no_grad()
def estimate_pos_weight(dataset_adapter, indices, precomputed_mask_path, args, device):  # Takes adapter
    logger.info(
        f"Estimating pos_weight using {args.estimate_weight_subset:.1%} of {len(indices)} training samples (Greedy Oracle Labels)...")
    num_pos, num_neg, nodes_processed = 0, 0, 0

    subset_size = int(len(indices) * args.estimate_weight_subset)
    if subset_size == 0:
        logger.warning("Subset size for weight estimation is zero. Using default 1.0.")
        return torch.tensor([1.0], device=device)

    subset_indices_for_est = random.sample(indices, subset_size)  # Sample from provided train indices

    # Use dataset_adapter (PASCALAdapterDataset instance)
    est_loader = PyTorchDataLoader(dataset_adapter, batch_size=1, sampler=SequentialSampler(subset_indices_for_est),
                                   collate_fn=collate_fn, num_workers=max(0, args.num_workers // 2))

    pbar_est = tqdm(est_loader, desc="Estimating pos_weight (Greedy Oracle)", leave=False, mininterval=0.5)
    items_skipped_dataset = 0

    for batch_items in pbar_est:
        if batch_items is None: items_skipped_dataset += 1; continue

        # batch_items is [ (support_set, query_set) ] since batch_size=1
        _support_set, query_set = batch_items[0]

        query_class = query_set.get('class')
        query_idx = query_set.get('query_index')
        query_gt_mask_np = query_set.get('gt_mask_np')

        if query_class is None or query_idx is None or query_gt_mask_np is None:
            items_skipped_dataset += 1;
            continue

        query_gt_mask_bool = query_gt_mask_np.astype(bool)
        ref_shape = query_gt_mask_bool.shape

        if ref_shape != (args.img_size, args.img_size):
            items_skipped_dataset += 1;
            continue

        candidate_masks_bool_list = []
        try:
            # Path uses query_class (string name) and query_idx (string ID)
            mask_file_path = Path(precomputed_mask_path) / f"{query_idx}_sam_hybrid_masks.pkl"
            if not mask_file_path.is_file(): items_skipped_dataset += 1; continue

            with open(mask_file_path, 'rb') as f:
                sam_output = pickle.load(f)

            sam_output.sort(key=lambda x: x['area'], reverse=True)
            candidate_masks_data_raw = sam_output[:args.max_masks_graph]

            for m_data in candidate_masks_data_raw:
                mask_np = m_data['segmentation']
                if mask_np.shape == ref_shape:
                    candidate_masks_bool_list.append(mask_np.astype(bool))

            if not candidate_masks_bool_list: items_skipped_dataset += 1; continue

            selected_oracle_indices, _ = _run_greedy_oracle_for_labels(
                candidate_masks_bool_list, query_gt_mask_bool, ref_shape
            )

            for i in range(len(candidate_masks_bool_list)):
                if i in selected_oracle_indices:
                    num_pos += 1
                else:
                    num_neg += 1
                nodes_processed += 1
            pbar_est.set_postfix({'pos': num_pos, 'neg': num_neg, 'nodes': nodes_processed})
        except Exception as e_inner:
            items_skipped_dataset += 1;
            continue

    processed_samples_count = len(subset_indices_for_est) - items_skipped_dataset
    logger.info(
        f"Weight Estimation Done: Processed {nodes_processed} nodes from {processed_samples_count}/{len(subset_indices_for_est)} valid samples.")
    logger.info(f"Counts -> Positive: {num_pos}, Negative: {num_neg}")

    if num_pos == 0:
        logger.warning("No positive nodes found by oracle during weight estimation. Using default pos_weight=1.0.")
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_neg / num_pos
        logger.info(f"Calculated pos_weight (Greedy Oracle) = {pos_weight_value:.4f}")

    return torch.tensor([pos_weight_value], device=device)


# --- Training Function (Uses precomputed graphs) --- (Unchanged)
# UPDATED: Added feature_extractor as argument
def train_gnn(model, train_loader, optimizer, criterion, device,
              args, feature_extractor,  # <--- ADDED feature_extractor
              scheduler=None):
    model.train()
    total_loss, processed_graphs, processed_nodes, skipped_items_load = 0, 0, 0, 0
    grad_norm_history, epoch_train_ious = [], []

    pbar = tqdm(train_loader, desc=f"Epoch {args.current_epoch}/{args.epochs} Training", leave=False, mininterval=1.0)
    batch_num = 0

    for batch_items in pbar:
        batch_num += 1
        if batch_items is None:
            skipped_items_load += args.batch_size
            continue

        # batch_items is a list of (support_set, query_set) tuples.
        # We need to process each (graph, support_features) pair individually
        # as the support features are now dynamic and not part of the precomputed graph.

        individual_graphs_to_process = []
        individual_support_features = []
        individual_original_data_for_iou = []

        for item_idx, (support_set, query_set) in enumerate(batch_items):
            query_class = query_set.get('class')
            query_idx = query_set.get('query_index')

            if not query_class or not query_idx:
                skipped_items_load += 1
                continue

            graph_data = load_precomputed_graph_data(
                query_class,
                query_idx,
                precomputed_graph_dir=args.precomputed_graph_path
            )
            # In your main script, before the training loop
            import matplotlib.pyplot as plt
            def debug_visualize_sample(support_set, query_set, graph_data):
                fig, axs = plt.subplots(2, args.k_shot + 1, figsize=(15, 6))

                # Query
                q_img_np = (query_set['image_resnet'].cpu().numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485,
                                                                                                                  0.456,
                                                                                                                  0.406])  # Denormalize
                q_img_np = np.clip(q_img_np, 0, 1)
                q_mask_np = query_set['gt_mask_np']

                axs[0, 0].imshow(q_img_np)
                axs[0, 0].set_title(f"Q: {query_set['class']}")
                axs[0, 0].axis('off')

                axs[1, 0].imshow(q_mask_np, cmap='gray')
                axs[1, 0].set_title(f"Q GT Mask")
                axs[1, 0].axis('off')

                # Supports
                for i, s_item in enumerate(support_set):
                    if i >= args.k_shot: break  # Only show K
                    s_img_np = (s_item['image'].cpu().numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485,
                                                                                                            0.456,
                                                                                                            0.406])
                    s_img_np = np.clip(s_img_np, 0, 1)
                    s_mask_np = s_item['mask'].squeeze().cpu().numpy()

                    axs[0, i + 1].imshow(s_img_np)
                    axs[0, i + 1].set_title(f"S{i + 1}")
                    axs[0, i + 1].axis('off')

                    axs[1, i + 1].imshow(s_mask_np, cmap='gray')
                    axs[1, i + 1].set_title(f"S{i + 1} Mask")
                    axs[1, i + 1].axis('off')

                plt.tight_layout()
                plt.suptitle(f"Sample: {query_set['query_index']}, Class: {query_set['class']}")
                plt.show()

                # Optionally visualize some candidate masks and oracle labels from graph_data
                if graph_data and graph_data.num_nodes > 0:
                    fig_nodes, axs_nodes = plt.subplots(1, min(20, graph_data.num_nodes), figsize=(12, 3))
                    if min(5, graph_data.num_nodes) == 1: axs_nodes = [axs_nodes]

                    for i in range(min(20, graph_data.num_nodes)):
                        mask = graph_data.node_masks_np[i]
                        label = graph_data.y[i].item()
                        axs_nodes[i].imshow(mask, cmap='gray')
                        axs_nodes[i].set_title(f"Node {i}\nLabel: {int(label)}")
                        axs_nodes[i].axis('off')
                    plt.suptitle(f"Query {query_set['query_index']} Candidate Masks & Oracle Labels")
                    plt.show()

            # Call this in train_gnn and evaluate_gnn (e.g., for the first batch, or randomly)
            # Example in train_gnn loop:
            # if batch_num == 1 and args.current_epoch == 1: # Only for first batch of first epoch
            debug_visualize_sample(support_set, query_set, graph_data)
            if graph_data is not None and graph_data.num_nodes > 0:
                if not hasattr(graph_data, 'x_base') or not hasattr(graph_data, 'y') or \
                        graph_data.x_base.shape[0] != graph_data.num_nodes or \
                        graph_data.y.shape[0] != graph_data.num_nodes:
                    logger.warning(
                        f"Epoch {args.current_epoch}, Batch {batch_num}, Item {query_class}/{query_idx}: Invalid graph data loaded. Skipping.")
                    skipped_items_load += 1
                    continue

                # --- NEW: Dynamically extract support features for this episode ---
                k_shot_support_features_list = []
                for support_item in support_set:  # support_item['mask'] is (H,W) float tensor
                    s_img = support_item['image'].unsqueeze(0)
                    s_mask = support_item['mask']
                    if s_mask.sum() < 1e-6:
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))
                        continue
                    s_fm = feature_extractor.get_feature_map(s_img)
                    if s_fm is not None:
                        s_fg_feat = feature_extractor.get_masked_features(s_fm, s_mask.unsqueeze(0))
                        if s_fg_feat is not None and s_fg_feat.numel() > 0:
                            k_shot_support_features_list.append(s_fg_feat.squeeze(0))
                        else:
                            k_shot_support_features_list.append(
                                torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))
                    else:
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))

                if not k_shot_support_features_list and len(support_set) > 0:
                    # Fallback if all support feature extraction failed for current episode
                    logger.warning(
                        f"No support features extracted for current episode {query_class}/{query_idx}. Padding with zeros.")
                    for _ in range(len(support_set)):
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))

                current_graph_support_features = torch.stack(k_shot_support_features_list).to(
                    device) if k_shot_support_features_list else torch.empty((0, feature_extractor.feature_dim),
                                                                             device=device)
                # --- END NEW ---

                individual_graphs_to_process.append(graph_data)
                individual_support_features.append(current_graph_support_features)
                individual_original_data_for_iou.append({
                    'node_masks_np': graph_data.node_masks_np,
                    'gt_mask_np': graph_data.gt_mask_np.astype(bool),
                    'image_size': graph_data.image_size
                })
            else:
                skipped_items_load += 1

        if not individual_graphs_to_process: continue

        # Process each individual graph from the original batch
        # This means the effective batch_size for the GNN forward pass is 1.
        for i in range(len(individual_graphs_to_process)):
            current_graph_data = individual_graphs_to_process[i]
            current_support_features = individual_support_features[i]

            try:
                # Create a PyG Batch of size 1 for the current graph
                single_graph_batch = Batch.from_data_list([current_graph_data]).to(device)

                optimizer.zero_grad()
                if single_graph_batch.num_nodes == 0: continue

                # UPDATED: Pass support_k_shot_features as a separate argument
                out_logits = model(single_graph_batch, current_support_features)
                targets = single_graph_batch.y

                if out_logits.shape[0] != targets.shape[0]:
                    logger.error(
                        f"Epoch {args.current_epoch}, Batch {batch_num}, Sub-Graph {i}: Logits shape {out_logits.shape} != target shape {targets.shape}. Skipping.")
                    continue

                loss = criterion(out_logits, targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"Epoch {args.current_epoch}, Batch {batch_num}, Sub-Graph {i}: NaN/Inf loss ({loss.item()}). Skipping backward.")
                    optimizer.zero_grad();
                    continue

                loss.backward()

                if args.clip_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    grad_norm_history.append(grad_norm.item())

                optimizer.step()

                # Scheduler step should typically happen once per optimizer step, not per individual graph
                # If using CosineAnnealingLR that depends on total steps, it's better to step once per
                # original batch_items loop or based on total steps.
                # For simplicity here, we keep it per optimizer step.
                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()

                with torch.no_grad():
                    pred_probs = torch.sigmoid(out_logits.detach())
                    pred_labels = (pred_probs > args.eval_threshold).cpu().numpy().astype(np.int8)

                    # Original data for IoU is now accessed directly from individual_original_data_for_iou
                    original_data = individual_original_data_for_iou[i]
                    node_masks_np_list = original_data['node_masks_np']
                    gt_mask_bool = original_data['gt_mask_np']
                    ref_shape_tuple = original_data['image_size']

                    # For a single graph, pred_labels corresponds to its nodes
                    graph_pred_labels = pred_labels

                    if len(node_masks_np_list) != len(graph_pred_labels):  # Check consistency
                        logger.warning(
                            f"Epoch {args.current_epoch}, Batch {batch_num}, Sub-Graph {i}: Mask count mismatch for IoU. Skipping this graph IoU.")
                        continue

                    selected_masks_for_final = [
                        node_masks_np_list[j] for j, label in enumerate(graph_pred_labels) if label == 1
                    ]
                    final_pred_mask_bool = combine_masks(selected_masks_for_final, ref_shape=ref_shape_tuple)

                    try:
                        iou = calculate_iou(final_pred_mask_bool, gt_mask_bool)
                        epoch_train_ious.append(iou)  # Accumulate for the epoch
                    except Exception:
                        epoch_train_ious.append(0.0)

                total_loss += loss.item()  # Accumulate loss over individual graphs
                processed_graphs += 1  # Count each graph processed
                processed_nodes += single_graph_batch.num_nodes
            except:
                break

            # Update progress bar postfix after processing all individual graphs from original batch_items
            avg_loss_so_far = total_loss / (processed_graphs if processed_graphs > 0 else 1)
            avg_train_iou_so_far = np.mean(epoch_train_ious) if epoch_train_ious else 0.0
            pbar_postfix = {
                'loss': f"{avg_loss_so_far:.4f}",
                'avg_train_iou': f"{avg_train_iou_so_far:.4f}",
            }
            if args.clip_grad_norm > 0 and grad_norm_history: pbar_postfix['grad_norm'] = f"{grad_norm_history[-1]:.2f}"
            pbar.set_postfix(pbar_postfix)

            del single_graph_batch, out_logits, targets, loss
            if 'pred_probs' in locals(): del pred_probs
            if 'pred_labels' in locals(): del pred_labels
            del current_graph_data, current_support_features  # Clean up individual graph items

        del individual_graphs_to_process, individual_support_features, individual_original_data_for_iou  # Clean up batch lists

    avg_loss_epoch = total_loss / (processed_graphs if processed_graphs > 0 else 1)
    avg_train_iou_epoch = np.mean(epoch_train_ious) if epoch_train_ious else 0.0
    avg_grad_norm = np.mean(grad_norm_history) if grad_norm_history else 0

    logger.info(
        f"Epoch {args.current_epoch} Train Finished. Avg Loss: {avg_loss_epoch:.4f}, Avg Train IoU: {avg_train_iou_epoch:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, Skipped Loads: {skipped_items_load}")
    return avg_loss_epoch, avg_train_iou_epoch, avg_grad_norm


# --- Evaluation Function (Uses precomputed graphs) --- (Unchanged)
# UPDATED: Added feature_extractor as argument
@torch.no_grad()
def evaluate_gnn(model, eval_loader, criterion, device,
                 args, feature_extractor,  # <--- ADDED feature_extractor
                 eval_threshold=0.5):
    model.eval()
    total_loss, all_ious, processed_graphs, processed_nodes, skipped_items_load = 0, [], 0, 0, 0
    num_pos_preds_nodes, total_nodes_evaluated = 0, 0

    pbar = tqdm(eval_loader, desc="Evaluating", leave=False, mininterval=1.0)
    batch_num = 0
    for batch_items in pbar:
        batch_num += 1
        if batch_items is None:
            skipped_items_load += args.eval_batch_size
            continue

        individual_graphs_to_process = []
        individual_support_features = []
        individual_original_data_for_iou = []

        for item_idx, (support_set, query_set) in enumerate(batch_items):
            query_class = query_set.get('class')
            query_idx = query_set.get('query_index')

            if not query_class or not query_idx:
                skipped_items_load += 1
                continue

            graph_data = load_precomputed_graph_data(
                query_class,
                query_idx,
                precomputed_graph_dir=args.precomputed_graph_path
            )

            if graph_data is not None and graph_data.num_nodes > 0:
                if not hasattr(graph_data, 'x_base') or not hasattr(graph_data, 'y') or \
                        graph_data.x_base.shape[0] != graph_data.num_nodes or \
                        graph_data.y.shape[0] != graph_data.num_nodes:
                    skipped_items_load += 1
                    continue

                # --- NEW: Dynamically extract support features for this episode ---
                k_shot_support_features_list = []
                for support_item in support_set:
                    s_img = support_item['image'].unsqueeze(0)
                    s_mask = support_item['mask']
                    if s_mask.sum() < 1e-6:
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))
                        continue
                    s_fm = feature_extractor.get_feature_map(s_img)
                    if s_fm is not None:
                        s_fg_feat = feature_extractor.get_masked_features(s_fm, s_mask.unsqueeze(0))
                        if s_fg_feat is not None and s_fg_feat.numel() > 0:
                            k_shot_support_features_list.append(s_fg_feat.squeeze(0))
                        else:
                            k_shot_support_features_list.append(
                                torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))
                    else:
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))

                if not k_shot_support_features_list and len(support_set) > 0:
                    logger.warning(
                        f"No support features extracted for current episode {query_class}/{query_idx}. Padding with zeros.")
                    for _ in range(len(support_set)):
                        k_shot_support_features_list.append(
                            torch.zeros(feature_extractor.feature_dim, device=feature_extractor.device))

                current_graph_support_features = torch.stack(k_shot_support_features_list).to(
                    device) if k_shot_support_features_list else torch.empty((0, feature_extractor.feature_dim),
                                                                             device=device)
                # --- END NEW ---

                individual_graphs_to_process.append(graph_data)
                individual_support_features.append(current_graph_support_features)
                individual_original_data_for_iou.append({
                    'node_masks_np': graph_data.node_masks_np,
                    'gt_mask_np': graph_data.gt_mask_np.astype(bool),
                    'image_size': graph_data.image_size,
                    'query_class': graph_data.query_class,
                    'query_index': graph_data.query_index
                })
            else:
                skipped_items_load += 1

        if not individual_graphs_to_process: continue

        # Process each individual graph from the original batch
        for i in range(len(individual_graphs_to_process)):
            current_graph_data = individual_graphs_to_process[i]
            current_support_features = individual_support_features[i]

            try:
                single_graph_batch = Batch.from_data_list([current_graph_data]).to(device)

                if single_graph_batch.num_nodes == 0: continue

                # UPDATED: Pass support_k_shot_features as a separate argument
                out_logits = model(single_graph_batch, current_support_features)
                targets = single_graph_batch.y
                loss_val = None

                if out_logits.shape[0] == targets.shape[0]:
                    loss_val = criterion(out_logits, targets)
                    total_loss += loss_val.item()  # Accumulate loss over individual graphs

                processed_graphs += 1  # Count each graph processed
                processed_nodes += single_graph_batch.num_nodes

                pred_probs = torch.sigmoid(out_logits)
                pred_labels_nodes = (pred_probs > eval_threshold).cpu().numpy().astype(np.int8)

                num_nodes_in_this_graph = single_graph_batch.num_nodes
                total_nodes_evaluated += num_nodes_in_this_graph
                graph_node_pred_labels = pred_labels_nodes
                num_pos_preds_nodes += graph_node_pred_labels.sum()

                original_data = individual_original_data_for_iou[i]
                node_masks_np_list = original_data['node_masks_np']
                gt_mask_bool = original_data['gt_mask_np']
                ref_shape_tuple = original_data['image_size']

                if len(node_masks_np_list) != num_nodes_in_this_graph:
                    logger.warning(
                        f"[Eval] Batch {batch_num}, Sub-Graph {i}: Mask count mismatch for IoU. Skipping IoU.")
                    continue

                selected_masks_for_final = [
                    node_masks_np_list[j] for j, label in enumerate(graph_node_pred_labels) if label == 1
                ]
                final_pred_mask_bool = combine_masks(selected_masks_for_final, ref_shape=ref_shape_tuple)

                try:
                    iou = calculate_iou(final_pred_mask_bool, gt_mask_bool)
                    all_ious.append(iou)  # Accumulate for the epoch
                except Exception as e_iou:
                    logger.error(
                        f"[Eval IoU Calc] Batch {batch_num}, Sub-Graph {i}: Error calculating final IoU: {e_iou}")
                    all_ious.append(0.0)
            except:
                break
                    # Update progress bar postfix after processing all individual graphs from original batch_items

            avg_loss_so_far = total_loss / (processed_graphs if processed_graphs > 0 else 1)
            avg_iou_so_far = np.mean(all_ious) if all_ious else 0.0
            pos_pred_rate_so_far = num_pos_preds_nodes / (total_nodes_evaluated if total_nodes_evaluated > 0 else 1)
            pbar_postfix = {'val_loss': f"{avg_loss_so_far:.4f}"}
            if all_ious: pbar_postfix['avg_iou'] = f"{avg_iou_so_far:.4f}"
            if total_nodes_evaluated > 0: pbar_postfix['pos_preds%'] = f"{pos_pred_rate_so_far:.1%}"
            pbar.set_postfix(pbar_postfix)

            del single_graph_batch, out_logits, targets, pred_probs, pred_labels_nodes
            if 'loss_val' in locals() and loss_val is not None: del loss_val
            del current_graph_data, current_support_features  # Clean up individual graph items

        del individual_graphs_to_process, individual_support_features, individual_original_data_for_iou  # Clean up batch lists

    avg_loss_epoch = total_loss / (processed_graphs if processed_graphs > 0 else 1)
    avg_iou = np.mean(all_ious) if all_ious else 0.0
    std_iou = np.std(all_ious) if all_ious else 0.0
    pos_pred_rate_nodes = num_pos_preds_nodes / (total_nodes_evaluated if total_nodes_evaluated > 0 else 1)

    logger.info(
        f"Evaluation Finished. Avg Loss: {avg_loss_epoch:.4f}, Avg IoU: {avg_iou:.4f} (+/- {std_iou:.4f}), Node Pos Pred Rate: {pos_pred_rate_nodes:.2%}, Skipped Loads: {skipped_items_load}")
    return avg_loss_epoch, avg_iou


# --- Main Script ---
def main(args):
    if args.seed is not None:
        random.seed(args.seed);
        np.random.seed(args.seed);
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if args.edge_feature_mode == 'iou_dist_area':
        args.edge_feature_dim = 3
    elif args.edge_feature_mode == 'iou_only':
        args.edge_feature_dim = 1
    else:
        logger.warning(
            f"Unknown edge_feature_mode '{args.edge_feature_mode}'. Defaulting to 'iou_only' with edge_feature_dim = 1.")
        args.edge_feature_mode = 'iou_only'
        args.edge_feature_dim = 1
    logger.info(f"Effective edge_feature_mode: '{args.edge_feature_mode}', edge_feature_dim: {args.edge_feature_dim}")

    feature_extractor = None
    # UPDATED: Feature extractor is now needed for train/eval modes as well
    if args.mode == 'precompute_graphs' or args.mode == 'train' or args.mode == 'eval':
        logger.info("Initializing FeatureExtractor...")
        try:
            # Path checks for PASCAL
            if not os.path.isdir(args.pascal_datapath): raise FileNotFoundError(
                f"PASCAL datapath missing: {args.pascal_datapath}")
            if not os.path.isdir(args.precomputed_mask_path): raise FileNotFoundError(
                f"Precomputed SAM mask path invalid: {args.precomputed_mask_path}")
            feature_extractor = FeatureExtractor(device)
        except Exception as e:
            logger.critical(f"Failed to initialize FeatureExtractor: {e}", exc_info=True);
            return
    else:
        logger.info(f"Mode: {args.mode}. FeatureExtractor will not be initialized.")
        # Path checks still needed even if FE is not initialized (for graph loading)
        if not os.path.isdir(args.pascal_datapath): raise FileNotFoundError(
            f"PASCAL datapath missing: {args.pascal_datapath}")
        if not os.path.isdir(args.precomputed_graph_path): raise FileNotFoundError(
            f"Precomputed graph path invalid: {args.precomputed_graph_path}")
        if args.use_loss_weighting and args.pos_weight_value is None:
            if not os.path.isdir(args.precomputed_mask_path): raise FileNotFoundError(
                f"Precomputed SAM mask path (for weight est.) invalid: {args.precomputed_mask_path}")

    # --- Initialize PASCAL Dataset & Adapter ---
    logger.info("Initializing PASCAL dataset and adapter...")

    # Define image transform for PASCAL dataset (similar to FSSDataset's resnet_transform)
    pascal_image_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset_adapter, eval_dataset_adapter = None, None
    train_indices, val_indices = [], []  # Will be full range for each dataset

    try:
        if args.mode == 'precompute_graphs':
            # For precomputation, create a combined dataset for all folds or a specific one.
            # Here, we'll just precompute for the training split of the specified fold.
            # User might want to run precomputation for 'val' split separately if needed.
            logger.info(f"Precomputation mode: Will precompute for 'trn' split of PASCAL fold {args.pascal_fold}")
            precompute_pascal_dataset_trn = DatasetPASCAL(  # Dataset for TRN split
                datapath=args.pascal_datapath,
                fold=args.pascal_fold,
                transform=pascal_image_transform,
                split='trn',
                shot=args.k_shot,
                use_original_imgsize=args.pascal_use_original_imgsize
            )
            dataset_for_precompute_trn = PASCALAdapterDataset(
                precompute_pascal_dataset_trn,
                args.img_size,
                PASCAL_CLASS_NAMES_LIST
            )
            all_dataset_indices_trn = list(range(len(dataset_for_precompute_trn)))
            logger.info(
                f"Precomputing graphs for {len(all_dataset_indices_trn)} samples in 'trn' split of fold {args.pascal_fold}.")

            precomputation_start_time = time.time()
            precompute_graphs_entrypoint(
                dataset_adapter=dataset_for_precompute_trn,
                indices=all_dataset_indices_trn,
                args=args,
                feature_extractor=feature_extractor,  # Pass FE for query feature map
                device=device
            )
            precomputation_time_trn = time.time() - precomputation_start_time
            logger.info(
                f"Graph precomputation for 'trn' split finished in {time.strftime('%H:%M:%S', time.gmtime(precomputation_time_trn))}.")

            # Optionally precompute for 'val' split as well
            logger.info(f"Precomputation mode: Will precompute for 'val' split of PASCAL fold {args.pascal_fold}")
            precompute_pascal_dataset_val = DatasetPASCAL(  # Dataset for VAL split
                datapath=args.pascal_datapath,
                fold=args.pascal_fold,
                transform=pascal_image_transform,
                split='val',
                shot=args.k_shot,
                use_original_imgsize=args.pascal_use_original_imgsize
            )
            dataset_for_precompute_val = PASCALAdapterDataset(
                precompute_pascal_dataset_val,
                args.img_size,
                PASCAL_CLASS_NAMES_LIST
            )
            all_dataset_indices_val = list(range(len(dataset_for_precompute_val)))
            logger.info(
                f"Precomputing graphs for {len(all_dataset_indices_val)} samples in 'val' split of fold {args.pascal_fold}.")

            precomputation_start_time_val = time.time()
            precompute_graphs_entrypoint(
                dataset_adapter=dataset_for_precompute_val,
                indices=all_dataset_indices_val,
                args=args,
                feature_extractor=feature_extractor,  # Pass FE for query feature map
                device=device
            )
            precomputation_time_val = time.time() - precomputation_start_time_val
            logger.info(
                f"Graph precomputation for 'val' split finished in {time.strftime('%H:%M:%S', time.gmtime(precomputation_time_val))}.")

            logger.info("Precomputation complete. Exiting.")
            if feature_extractor: del feature_extractor; gc.collect(); torch.cuda.empty_cache() if "cuda" in device.type else None
            return

        # For 'train' or 'eval' modes:
        logger.info(f"Setting up PASCAL datasets for fold {args.pascal_fold}")
        base_train_dataset = DatasetPASCAL(
            datapath=args.pascal_datapath,
            fold=args.pascal_fold,
            transform=pascal_image_transform,
            split='trn',
            shot=args.k_shot,
            use_original_imgsize=args.pascal_use_original_imgsize
        )
        train_dataset_adapter = PASCALAdapterDataset(base_train_dataset, args.img_size, PASCAL_CLASS_NAMES_LIST)
        train_indices = list(range(len(train_dataset_adapter)))  # Full range for this dataset

        base_eval_dataset = DatasetPASCAL(
            datapath=args.pascal_datapath,
            fold=args.pascal_fold,
            transform=pascal_image_transform,
            split='val',
            shot=args.k_shot,
            use_original_imgsize=args.pascal_use_original_imgsize
        )
        eval_dataset_adapter = PASCALAdapterDataset(base_eval_dataset, args.img_size, PASCAL_CLASS_NAMES_LIST)
        val_indices = list(range(len(eval_dataset_adapter)))  # Full range for this dataset

        logger.info(
            f"PASCAL datasets initialized. Train items: {len(train_dataset_adapter)}, Eval items: {len(eval_dataset_adapter)}")

    except Exception as e:
        logger.critical(f"Failed to initialize PASCAL dataset/adapter: {e}", exc_info=True);
        return

    # --- DataLoaders for train/eval ---
    train_loader, eval_loader = None, None
    try:
        # Use RandomSampler for training, SequentialSampler for validation
        train_sampler = RandomSampler(train_dataset_adapter)  # PASCALAdapterDataset is the data_source
        val_sampler = SequentialSampler(eval_dataset_adapter)

        pin_memory_flag = True if "cuda" in device.type else False
        persistent_workers_flag = True if args.num_workers > 0 else False

        train_loader = PyTorchDataLoader(train_dataset_adapter, batch_size=args.batch_size, sampler=train_sampler,
                                         collate_fn=collate_fn, num_workers=args.num_workers,
                                         pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag,
                                         drop_last=True)
        eval_loader = PyTorchDataLoader(eval_dataset_adapter, batch_size=args.eval_batch_size, sampler=val_sampler,
                                        collate_fn=collate_fn, num_workers=args.num_workers,
                                        pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag)
        logger.info(f"DataLoaders created. Train batches: ~{len(train_loader)}, Val batches: {len(eval_loader)}")
    except Exception as e:
        logger.critical(f"Failed to initialize DataLoaders for PASCAL: {e}", exc_info=True);
        return

    pos_weight_tensor = torch.tensor([1.0], device=device)
    if args.use_loss_weighting:
        if args.pos_weight_value is not None:
            pos_weight_tensor = torch.tensor([args.pos_weight_value if args.pos_weight_value > 0 else 1.0],
                                             device=device)
            logger.info(f"Using fixed pos_weight_value: {pos_weight_tensor.item()}")
        else:
            try:
                # Pass train_dataset_adapter and its indices to estimate_pos_weight
                pos_weight_tensor = estimate_pos_weight(train_dataset_adapter, train_indices,
                                                        args.precomputed_mask_path, args, device)
                if pos_weight_tensor.item() > 1000:
                    pos_weight_tensor = torch.tensor([1000.0], device=device)
                elif pos_weight_tensor.item() <= 0:
                    pos_weight_tensor = torch.tensor([1.0], device=device)
                logger.info(f"Using estimated pos_weight: {pos_weight_tensor.item()}")
            except Exception as est_e:
                logger.error(f"Failed to estimate pos_weight: {est_e}. Using default 1.0.", exc_info=True)
                pos_weight_tensor = torch.tensor([1.0], device=device)
    else:
        logger.info("Loss weighting disabled. Using pos_weight=1.0.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    logger.info("Initializing GNN model...")
    model, optimizer, scheduler = None, None, None
    try:
        model = SupportAwareMaskGNN(
            base_node_visual_dim=args.feature_size,
            geometric_dim=4,
            sam_meta_dim=2,
            support_feature_dim=args.feature_size,
            spt_attn_num_heads=args.spt_attn_num_heads,
            spt_attn_output_dim=args.spt_attn_output_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            num_gnn_layers=args.gnn_layers,
            num_heads_gnn=args.gnn_heads,
            edge_feature_dim=args.edge_feature_dim,
            dropout_rate=args.dropout
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"SupportAwareMaskGNN initialized. Params: {total_params:,}. On device: {device}")
    except Exception as e:
        logger.critical(f"Failed to initialize GNN model: {e}", exc_info=True);
        return

    try:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    except Exception as e:
        logger.critical(f"Failed to initialize optimizer: {e}", exc_info=True);
        return

    if args.use_scheduler:
        try:
            total_steps = args.epochs * len(train_loader)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)
            logger.info(f"Using CosineAnnealingLR scheduler, T_max={total_steps}.")
        except Exception as e:
            logger.error(f"Failed to init LR scheduler: {e}. No scheduler will be used.", exc_info=True)
            scheduler = None

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name_suffix = f"pascal_f{args.pascal_fold}_{args.k_shot}shot_{args.img_size}px_{args.graph_connectivity}_{args.edge_feature_mode}"
    if args.use_loss_weighting: run_name_suffix += "_weighted"
    run_name_suffix += f"_{timestamp}"
    run_name = f"scr-gt_pascal_PRECOMP_{run_name_suffix}"

    model_save_dir = Path(args.output_dir) / run_name
    try:
        model_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run Name: {run_name}")
        logger.info(f"Model save directory: {model_save_dir}")
        with open(model_save_dir / 'config_args.txt', 'w') as f:
            for k, v in vars(args).items(): f.write(f"{k}: {v}\n")
    except Exception as e:
        logger.error(f"Failed to create output dir/save config: {e}")


    if args.mode == 'train':
        logger.info(f"======== Starting Training (Epochs: {args.epochs}) ========")
        best_eval_iou = -1.0
        train_losses, train_ious_epochs, eval_losses, eval_ious_epochs, grad_norms_epochs = [], [], [], [], []
        start_train_time = time.time()

        for epoch in range(1, args.epochs + 1):
            args.current_epoch = epoch
            epoch_start_time = time.time()
            logger.info(f"--- Epoch {epoch}/{args.epochs} ---")

            train_loss, train_iou_epoch, avg_grad_norm_epoch = train_gnn(
                model, train_loader, optimizer, criterion, device, args, feature_extractor,  # <--- Pass FE
                scheduler if args.use_scheduler else None
            )
            train_losses.append(train_loss)
            train_ious_epochs.append(train_iou_epoch)
            grad_norms_epochs.append(avg_grad_norm_epoch)

            eval_loss, eval_iou = evaluate_gnn(
                model, eval_loader, criterion, device, args, feature_extractor,  # <--- Pass FE
                eval_threshold=args.eval_threshold
            )
            eval_losses.append(eval_loss)
            eval_ious_epochs.append(eval_iou)

            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} Summary | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | "
                        f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou_epoch:.4f} | "
                        f"Eval Loss: {eval_loss:.4f} | Eval IoU: {eval_iou:.4f}")

            if eval_iou > best_eval_iou:
                best_eval_iou = eval_iou
                save_path = model_save_dir / "best_model.pth"
                try:
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"*** New best model saved with Eval IoU: {best_eval_iou:.4f} to {save_path} ***")
                except Exception as e:
                    logger.error(f"Error saving best model: {e}")

            if args.save_interval > 0 and epoch % args.save_interval == 0:
                ckpt_path = model_save_dir / f"checkpoint_epoch_{epoch}.pth"
                try:
                    save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'best_eval_iou': best_eval_iou,
                                 'train_losses': train_losses, 'train_ious': train_ious_epochs,
                                 'eval_losses': eval_losses, 'eval_ious': eval_ious_epochs,
                                 'grad_norms': grad_norms_epochs,
                                 'args': vars(args)}
                    if scheduler: save_dict['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(save_dict, ckpt_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint epoch {epoch}: {e}")

        total_training_time = time.time() - start_train_time
        logger.info(
            f"======== Training Finished ({time.strftime('%H:%M:%S', time.gmtime(total_training_time))}) ========")
        logger.info(f"Best Validation IoU: {best_eval_iou:.4f}")
        try:
            torch.save(model.state_dict(), model_save_dir / "final_model.pth")
        except Exception as e:
            logger.error(f"Error saving final model: {e}")

        try:
            fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            epochs_range = range(1, args.epochs + 1)
            axs[0].plot(epochs_range, train_losses, label='Training Loss', marker='.');
            axs[0].plot(epochs_range, eval_losses, label='Validation Loss', marker='.')
            axs[0].set_ylabel('Loss');
            axs[0].legend();
            axs[0].grid(True, alpha=0.6);
            axs[0].set_title('Loss History')
            axs[1].plot(epochs_range, train_ious_epochs, label='Training IoU', marker='.');
            axs[1].plot(epochs_range, eval_ious_epochs, label='Validation IoU', marker='.')
            axs[1].set_ylabel('Mean IoU');
            axs[1].axhline(y=best_eval_iou, color='r', linestyle=':', label=f'Best Val IoU: {best_eval_iou:.4f}');
            axs[1].legend();
            axs[1].grid(True, alpha=0.6);
            axs[1].set_title('Mean IoU History');
            axs[1].set_ylim(
                bottom=max(0, np.min(train_ious_epochs + eval_ious_epochs) - 0.05 if train_ious_epochs else 0))
            axs[2].plot(epochs_range, grad_norms_epochs, label='Avg Grad Norm (Train)', marker='.')
            axs[2].set_ylabel('Gradient Norm');
            axs[2].set_xlabel('Epoch');
            axs[2].legend();
            axs[2].grid(True, alpha=0.6);
            axs[2].set_title('Avg Grad Norm History');
            axs[2].set_yscale('log')
            fig.suptitle(f'Training History ({run_name})', fontsize=14);
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(model_save_dir / "training_history.png", dpi=150);
            plt.close(fig)
            logger.info(f"Training history plot saved to {model_save_dir / 'training_history.png'}")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}", exc_info=True)

    elif args.mode == 'eval':
        logger.info(f"======== Starting Evaluation Mode ========")
        if not args.eval_model_path or not os.path.exists(args.eval_model_path):
            logger.error(f"Eval mode requires valid 'eval_model_path'. Path '{args.eval_model_path}' invalid.");
            return

        logger.info(f"Loading model from {args.eval_model_path} for evaluation...")
        try:
            state_dict = torch.load(args.eval_model_path, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load model from {args.eval_model_path}: {e}", exc_info=True);
            return

        logger.info(
            f"Starting evaluation on PASCAL fold {args.pascal_fold} validation set ({len(eval_dataset_adapter)} samples), threshold {args.eval_threshold}...")
        eval_start_time = time.time()
        eval_loss, eval_iou = evaluate_gnn(
            model, eval_loader, criterion, device, args, feature_extractor,  # <--- Pass FE
            eval_threshold=args.eval_threshold
        )
        eval_time = time.time() - eval_start_time
        logger.info(f"--- Final Evaluation Results ---");
        logger.info(f"Model: {args.eval_model_path}");
        logger.info(f"Samples: {len(eval_dataset_adapter)}")
        logger.info(
            f"Time: {eval_time:.2f}s | Threshold: {args.eval_threshold:.2f} | Eval Loss: {eval_loss:.4f} | Mean IoU: {eval_iou:.4f}")

    else:
        logger.error(f"Invalid mode '{args.mode}'. Choose 'train', 'eval', or 'precompute_graphs'.")

    logger.info("Cleaning up resources for train/eval modes...");
    if 'model' in locals() and model is not None: del model
    if 'feature_extractor' in locals() and feature_extractor is not None: del feature_extractor  # Clean up FE
    if 'train_dataset_adapter' in locals() and train_dataset_adapter is not None: del train_dataset_adapter
    if 'eval_dataset_adapter' in locals() and eval_dataset_adapter is not None: del eval_dataset_adapter
    if 'train_loader' in locals() and train_loader is not None: del train_loader
    if 'eval_loader' in locals() and eval_loader is not None: del eval_loader
    if 'optimizer' in locals() and optimizer is not None: del optimizer
    if 'criterion' in locals() and criterion is not None: del criterion
    if 'scheduler' in locals() and scheduler is not None: del scheduler
    gc.collect();
    torch.cuda.empty_cache() if "cuda" in device.type else None;
    logger.info("Script finished.")


if __name__ == "__main__":
    args = SimpleNamespace(
        mode='train',

        # PASCAL Paths & Settings (NEW)
        pascal_datapath=DEFAULT_PASCAL_DATAPATH,  # Path to VOCdevkit folder or similar structure
        pascal_fold=DEFAULT_PASCAL_FOLD,  # Fold for PASCAL-5i (0, 1, 2, 3)
        pascal_use_original_imgsize=DEFAULT_PASCAL_USE_ORIGINAL_IMGSIZE,  # Should be False
        # General Paths (update for PASCAL if needed)
        # fss_path=DEFAULT_FSS_PATH, # OLD, deprecated
        precomputed_mask_path=DEFAULT_PRECOMPUTED_MASK_PATH,
        precomputed_graph_path=DEFAULT_PRECOMPUTED_GRAPH_PATH,
        output_dir="./train_model_outputs",  # Output for PASCAL runs
        eval_model_path="",  # Path to model for 'eval' mode

        # General settings
        device=DEFAULT_DEVICE,
        seed=42,
        num_workers=4,

        # Dataset and Episode settings
        k_shot=DEFAULT_K_SHOT,
        img_size=DEFAULT_IMG_SIZE,  # Used for PASCAL image/mask resizing via transform
        # val_split=0.4, # OLD, deprecated for PASCAL

        # Graph Construction (used during precomputation_mode)
        max_masks_graph=DEFAULT_MAX_MASKS_GRAPH,
        graph_connectivity=DEFAULT_GRAPH_CONNECTIVITY,
        knn_k=DEFAULT_KNN_K,
        overlap_thresh=DEFAULT_OVERLAP_THRESH,
        edge_feature_mode=DEFAULT_EDGE_FEATURE_MODE,
        edge_feature_dim=3,

        # Model Hyperparameters
        feature_size=DEFAULT_FEATURE_SIZE,
        gnn_hidden_dim=256,
        gnn_layers=4,
        gnn_heads=4,
        dropout=0.3,
        spt_attn_num_heads=4,
        spt_attn_output_dim=128,

        # Training Hyperparameters
        epochs=1000,  # Adjusted for PASCAL
        batch_size=8 * 2,  # Adjusted for PASCAL
        eval_batch_size=8 * 2,
        lr=1e-4,  # Adjusted LR
        weight_decay=1e-4,
        use_scheduler=True,
        clip_grad_norm=1.0,

        # Loss Weighting
        use_loss_weighting=True,
        pos_weight_value=8,
        estimate_weight_subset=0.9,  # Use 25% of PASCAL train split of the fold for estimation

        # Saving and Logging
        save_interval=100,

        # Evaluation settings
        eval_threshold=0.5,

        # Precomputation settings
        overwrite_precomputed_graphs=False,  # Default to False to avoid accidental overwrite

        # Internal (do not set manually)
        current_epoch=0,
    )

    # --- Ensure PASCAL specific settings are respected ---
    if args.pascal_use_original_imgsize:
        logger.warning("`pascal_use_original_imgsize` is True. This is generally NOT recommended "
                       "as the pipeline expects fixed-size images/masks matching `args.img_size` "
                       "for SAM masks and feature consistency. Forcing to False.")
        args.pascal_use_original_imgsize = False

    if not (0 <= args.pascal_fold <= 3):
        raise ValueError(f"pascal_fold must be between 0 and 3. Got {args.pascal_fold}")

    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (KeyboardInterrupt).");
        sys.exit(0)
    except Exception as main_e:
        logger.critical(f"Unhandled exception in main: {main_e}", exc_info=True);
        sys.exit(1)

