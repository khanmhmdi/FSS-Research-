# --- IMPORTANT FIX ---
# This line MUST be at the very top of your script, before any other imports
# It is the most likely solution to the crash. The logs below are to confirm it.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import numpy as np
import torch  # Import torch for CUDA check

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QRadioButton, QGroupBox, QStatusBar, QMessageBox,
    QButtonGroup, QFormLayout, QSpinBox, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal

# SAM imports
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Imports for Graph Generation Feature
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph

# --- Configuration ---
SAM_CHECKPOINT_PATH = r"C:\Users\Khan\PycharmProjects\FSS-Research-\model\sam_vit_h_4b8939.pth"  # Make sure this path is correct!
MODEL_TYPE = "vit_h"

DEFAULT_SAM_AUTO_GENERATOR_PARAMS = {
    "points_per_side": 32, "pred_iou_thresh": 0.88, "stability_score_thresh": 0.92,
    "crop_n_layers": 1, "crop_n_points_downscale_factor": 2, "min_mask_region_area": 100,
}


# --- MODIFIED: Helper function with extensive debug logging ---
def create_graph_from_masks(masks_data: list, connectivity='knn', knn_k=5):
    """
    Creates a graph structure from a list of segmentation masks with debug logs.
    """
    print("--- DEBUG LOG ---: Entered 'create_graph_from_masks' function.", flush=True)

    if not masks_data:
        print("--- DEBUG LOG ---: No masks_data provided. Exiting function.", flush=True)
        return None, None

    masks_np_list_bool = [m['segmentation'] for m in masks_data]
    num_nodes = len(masks_np_list_bool)
    print(f"--- DEBUG LOG ---: Processing {num_nodes} masks to create graph nodes.", flush=True)

    if num_nodes == 0:
        return [], None

    # Calculate centroids for each mask to use as node positions
    node_centroids = []
    print("--- DEBUG LOG ---: About to calculate regionprops for all masks...", flush=True)
    try:
        for i in range(num_nodes):
            props = regionprops(masks_np_list_bool[i].astype(np.uint8))
            if props:
                node_centroids.append(props[0].centroid)
            else:
                node_centroids.append((0, 0))
    except Exception as e:
        print(f"--- DEBUG LOG ---: CRITICAL ERROR during regionprops calculation: {e}", flush=True)
        return None, None
    print(f"--- DEBUG LOG ---: Successfully calculated {len(node_centroids)} centroids.", flush=True)

    # Determine graph edges based on the chosen connectivity
    edge_index = None
    if num_nodes > 1 and connectivity == 'knn':
        print("--- DEBUG LOG ---: Preparing to build KNN graph.", flush=True)
        try:
            node_centroids_np = np.array(node_centroids)
            k_actual = min(knn_k, num_nodes - 1)
            print(f"--- DEBUG LOG ---: KNN k_actual = {k_actual}", flush=True)

            if k_actual > 0:
                # --- THIS IS THE MOST LIKELY CRASH POINT ---
                print(
                    "\n--- DEBUG LOG ---: About to call kneighbors_graph from scikit-learn. If the app crashes NOW, the MKL conflict is confirmed.\n",
                    flush=True)
                adj_sparse = kneighbors_graph(node_centroids_np, k_actual, mode='connectivity', include_self=False)
                print("--- DEBUG LOG ---: Call to kneighbors_graph SUCCEEDED.", flush=True)
                # --- CRASH POINT END ---

                adj_sparse_symmetric = adj_sparse + adj_sparse.T
                adj_sparse_symmetric[adj_sparse_symmetric > 1] = 1
                coo = adj_sparse_symmetric.tocoo()
                edge_index = np.vstack((coo.row, coo.col))
                print(f"--- DEBUG LOG ---: KNN graph construction complete. Found {edge_index.shape[1]} edges.",
                      flush=True)
        except Exception as e_knn:
            print(f"--- DEBUG LOG ---: CRITICAL ERROR building KNN graph: {e_knn}", flush=True)
            edge_index = None

    print("--- DEBUG LOG ---: Exiting 'create_graph_from_masks' function.", flush=True)
    return node_centroids, edge_index


class ImageDisplayLabel(QLabel):
    promptSegmentRequested = pyqtSignal(object, object, np.ndarray, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.original_image_rgb = None
        self.current_prompt_mask = None
        self.current_points = []
        self.current_bbox = None
        self.current_automatic_masks = None
        self.graph_nodes = None
        self.graph_edges = None
        self.input_mode = 'point'
        self.point_label = 1
        self.start_point = None
        self.end_point = None
        self.is_drawing_bbox = False
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, image_rgb: np.ndarray):
        self.original_image_rgb = image_rgb
        self.clear_segmentation()
        self._update_base_pixmap()

    def set_prompt_mask(self, mask: np.ndarray):
        self.current_prompt_mask = mask
        self.current_automatic_masks = None
        self.update()

    def set_automatic_masks(self, masks: list):
        self.current_automatic_masks = masks
        self.current_prompt_mask = None
        self.current_points = []
        self.current_bbox = None
        self.update()

    def set_input_mode(self, mode: str):
        self.input_mode = mode
        self.clear_segmentation()
        self.update()

    def set_point_label(self, label: int):
        self.point_label = label

    def clear_segmentation(self):
        self.current_points = []
        self.current_bbox = None
        self.current_prompt_mask = None
        self.current_automatic_masks = None
        self.is_drawing_bbox = False
        self.start_point = None
        self.end_point = None
        self.graph_nodes = None
        self.graph_edges = None
        self.update()
        if self.input_mode in ['point', 'bbox']:
            self.trigger_segmentation_request()

    def _update_base_pixmap(self):
        if self.original_image_rgb is None:
            self.setPixmap(QPixmap())
            return
        h, w, ch = self.original_image_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(self.original_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def _calculate_display_geometry(self):
        if self.original_image_rgb is None: return 0, 0, 0, 0, 0, 0
        img_h, img_w, _ = self.original_image_rgb.shape
        label_w, label_h = self.width(), self.height()
        if label_w == 0 or label_h == 0: return img_w, img_h, 0, 0, 0, 0
        aspect_ratio_label = label_w / label_h
        aspect_ratio_img = img_w / img_h
        if aspect_ratio_img > aspect_ratio_label:
            display_w = label_w
            display_h = int(label_w / aspect_ratio_img)
        else:
            display_h = label_h
            display_w = int(label_h * aspect_ratio_img)
        x_offset = (label_w - display_w) // 2
        y_offset = (label_h - display_h) // 2
        return img_w, img_h, display_w, display_h, x_offset, y_offset

    def get_scaled_coordinates(self, mouse_x, mouse_y):
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()
        if img_w == 0 or display_w == 0 or display_h == 0: return None, None
        x_on_img_display = mouse_x - x_offset
        y_on_img_display = mouse_y - y_offset
        if not (0 <= x_on_img_display < display_w and 0 <= y_on_img_display < display_h): return None, None
        original_x = int(x_on_img_display * (img_w / display_w))
        original_y = int(y_on_img_display * (img_h / display_h))
        original_x = max(0, min(original_x, img_w - 1))
        original_y = max(0, min(original_y, img_h - 1))
        return original_x, original_y

    def get_display_coordinates(self, original_x, original_y):
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()
        if img_w == 0 or display_w == 0 or display_h == 0: return None, None
        display_x = int(original_x * (display_w / img_w) + x_offset)
        display_y = int(original_y * (display_h / img_h) + y_offset)
        return display_x, display_y

    def mousePressEvent(self, event):
        if self.original_image_rgb is None or self.input_mode == 'automatic': return
        original_x, original_y = self.get_scaled_coordinates(event.x(), event.y())
        if original_x is None: return
        if self.input_mode == 'point':
            self.current_points.append((original_x, original_y, self.point_label))
            self.update()
        elif self.input_mode == 'bbox':
            self.is_drawing_bbox = True
            self.start_point = QPoint(event.x(), event.y())
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.original_image_rgb is None or self.input_mode == 'automatic': return
        if self.input_mode == 'bbox' and self.is_drawing_bbox:
            self.end_point = QPoint(event.x(), event.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.original_image_rgb is None or self.input_mode == 'automatic': return
        if self.input_mode == 'bbox' and self.is_drawing_bbox:
            self.is_drawing_bbox = False
            ox1, oy1 = self.get_scaled_coordinates(self.start_point.x(), self.start_point.y())
            ox2, oy2 = self.get_scaled_coordinates(event.x(), event.y())
            if ox1 is None or ox2 is None:
                self.current_bbox = None;
                self.update();
                return
            x1, y1, x2, y2 = min(ox1, ox2), min(oy1, oy2), max(ox1, ox2), max(oy1, oy2)
            if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
                self.current_bbox = None
            else:
                self.current_bbox = np.array([x1, y1, x2, y2])
            self.update()
        self.start_point = None
        self.end_point = None

    def trigger_segmentation_request(self):
        input_point_coords, input_point_labels = None, None
        if self.current_points:
            input_point_coords = np.array([p[:2] for p in self.current_points])
            input_point_labels = np.array([p[2] for p in self.current_points])
        bbox_to_emit = self.current_bbox if self.current_bbox is not None else np.array([])
        self.promptSegmentRequested.emit(input_point_coords, input_point_labels, bbox_to_emit, self.input_mode)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.original_image_rgb is None or self.pixmap() is None: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()
        if self.current_automatic_masks is not None:
            for i, mask_data in enumerate(self.current_automatic_masks):
                mask = mask_data["segmentation"]
                color = QColor.fromHsvF((i * 0.61803) % 1.0, 0.8, 0.9, 0.5)
                mask_rgb = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                mask_rgb[mask] = [color.red(), color.green(), color.blue(), color.alpha()]
                mask_q_image = QImage(mask_rgb.data, img_w, img_h, img_w * 4, QImage.Format_ARGB32)
                scaled_mask_q_image = mask_q_image.scaled(display_w, display_h, Qt.IgnoreAspectRatio,
                                                          Qt.SmoothTransformation)
                painter.drawImage(x_offset, y_offset, scaled_mask_q_image)
        elif self.current_prompt_mask is not None:
            mask = self.current_prompt_mask
            mask_rgb = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgb[mask] = [0, 150, 255, 128]
            mask_q_image = QImage(mask_rgb.data, img_w, img_h, img_w * 4, QImage.Format_ARGB32)
            scaled_mask_q_image = mask_q_image.scaled(display_w, display_h, Qt.IgnoreAspectRatio,
                                                      Qt.SmoothTransformation)
            painter.drawImage(x_offset, y_offset, scaled_mask_q_image)
        if self.graph_nodes is not None and self.graph_edges is not None:
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1.5))
            for i in range(self.graph_edges.shape[1]):
                src_node_idx, trg_node_idx = self.graph_edges[:, i]
                src_y, src_x = self.graph_nodes[src_node_idx]
                trg_y, trg_x = self.graph_nodes[trg_node_idx]
                display_x1, display_y1 = self.get_display_coordinates(src_x, src_y)
                display_x2, display_y2 = self.get_display_coordinates(trg_x, trg_y)
                if display_x1 is not None: painter.drawLine(display_x1, display_y1, display_x2, display_y2)
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(QColor(255, 0, 255)))
            for node_y, node_x in self.graph_nodes:
                display_x, display_y = self.get_display_coordinates(node_x, node_y)
                if display_x is not None: painter.drawEllipse(QPoint(display_x, display_y), 4, 4)
        if self.input_mode == 'point':
            for ox, oy, label in self.current_points:
                dx, dy = self.get_display_coordinates(ox, oy)
                if dx is not None:
                    color = Qt.green if label == 1 else Qt.red
                    painter.setBrush(color)
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(dx - 5, dy - 5, 10, 10)
        if self.input_mode == 'bbox':
            if self.current_bbox is not None:
                x1, y1, x2, y2 = self.current_bbox
                dx1, dy1 = self.get_display_coordinates(x1, y1)
                dx2, dy2 = self.get_display_coordinates(x2, y2)
                if dx1 is not None:
                    painter.setPen(QPen(Qt.blue, 2))
                    painter.drawRect(dx1, dy1, dx2 - dx1, dy2 - dy1)
            if self.is_drawing_bbox and self.start_point and self.end_point:
                painter.setPen(QPen(Qt.magenta, 2, Qt.DotLine))
                x, y = min(self.start_point.x(), self.end_point.x()), min(self.start_point.y(), self.end_point.y())
                w, h = abs(self.end_point.x() - self.start_point.x()), abs(self.end_point.y() - self.start_point.y())
                painter.drawRect(x, y, w, h)
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_base_pixmap()


class SAMSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Interactive & Automatic Segmentation")
        self.setGeometry(100, 100, 1200, 800)
        self.original_image_rgb = None
        self.sam = None
        self.sam_predictor = None
        self.sam_automatic_predictor = None
        self.current_auto_params = DEFAULT_SAM_AUTO_GENERATOR_PARAMS.copy()
        self.init_ui()
        self.load_sam_model()
        self.clear_btn.clicked.connect(self.reset_stats_display)

    def init_ui(self):
        central_widget = QWidget();
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        image_layout = QVBoxLayout()
        self.image_label = ImageDisplayLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.image_label.promptSegmentRequested.connect(self.perform_prompt_segmentation)
        image_layout.addWidget(self.image_label)
        main_layout.addLayout(image_layout, 3)
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setAlignment(Qt.AlignTop)
        main_layout.addLayout(control_panel_layout, 1)
        load_image_btn = QPushButton("Load Image");
        load_image_btn.clicked.connect(self.load_image);
        load_image_btn.setFont(QFont("Arial", 12))
        control_panel_layout.addWidget(load_image_btn);
        control_panel_layout.addSpacing(20)
        mode_group = QGroupBox("Segmentation Mode");
        mode_layout = QVBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.point_mode_radio = QRadioButton("Interactive: Points (Click)");
        self.point_mode_radio.setChecked(True)
        self.point_mode_radio.toggled.connect(lambda checked: self.set_mode('point', checked));
        self.mode_button_group.addButton(self.point_mode_radio)
        self.bbox_mode_radio = QRadioButton("Interactive: Bounding Box (Drag)")
        self.bbox_mode_radio.toggled.connect(lambda checked: self.set_mode('bbox', checked));
        self.mode_button_group.addButton(self.bbox_mode_radio)
        self.automatic_mode_radio = QRadioButton("Automatic: All Objects")
        self.automatic_mode_radio.toggled.connect(lambda checked: self.set_mode('automatic', checked));
        self.mode_button_group.addButton(self.automatic_mode_radio)
        mode_layout.addWidget(self.point_mode_radio);
        mode_layout.addWidget(self.bbox_mode_radio);
        mode_layout.addWidget(self.automatic_mode_radio)
        mode_group.setLayout(mode_layout);
        control_panel_layout.addWidget(mode_group);
        control_panel_layout.addSpacing(10)
        self.segment_prompts_btn = QPushButton("Segment (Apply Prompts)");
        self.segment_prompts_btn.clicked.connect(self.image_label.trigger_segmentation_request)
        self.segment_prompts_btn.setFont(QFont("Arial", 12, QFont.Bold));
        control_panel_layout.addWidget(self.segment_prompts_btn);
        control_panel_layout.addSpacing(10)
        self.label_group = QGroupBox("Point Label (for Points Mode)");
        label_layout = QVBoxLayout()
        self.fg_label_radio = QRadioButton("Foreground (Green)");
        self.fg_label_radio.setChecked(True)
        self.fg_label_radio.toggled.connect(lambda: self.image_label.set_point_label(1))
        self.bg_label_radio = QRadioButton("Background (Red)");
        self.bg_label_radio.toggled.connect(lambda: self.image_label.set_point_label(0))
        label_layout.addWidget(self.fg_label_radio);
        label_layout.addWidget(self.bg_label_radio);
        self.label_group.setLayout(label_layout)
        control_panel_layout.addWidget(self.label_group);
        control_panel_layout.addSpacing(10)
        self.stats_group = QGroupBox("Automatic Segmentation Statistics");
        stats_layout = QFormLayout()
        self.stats_mask_count_label = QLabel("N/A");
        self.stats_coverage_label = QLabel("N/A");
        self.stats_overlap_label = QLabel("N/A")
        stats_layout.addRow("Total Generated Masks:", self.stats_mask_count_label);
        stats_layout.addRow("Image Coverage (%):", self.stats_coverage_label)
        stats_layout.addRow("Highly Overlapping Masks:", self.stats_overlap_label)
        self.stats_group.setLayout(stats_layout);
        self.stats_group.setEnabled(False);
        control_panel_layout.addWidget(self.stats_group);
        control_panel_layout.addSpacing(10)
        self.auto_params_group = QGroupBox("Automatic Segmentation Parameters");
        auto_params_layout = QFormLayout()
        self.spin_points_per_side = QSpinBox();
        self.spin_points_per_side.setRange(8, 128);
        self.spin_points_per_side.setSingleStep(8)
        self.spin_points_per_side.setValue(self.current_auto_params["points_per_side"]);
        auto_params_layout.addRow("Points Per Side:", self.spin_points_per_side)
        self.spin_pred_iou_thresh = QDoubleSpinBox();
        self.spin_pred_iou_thresh.setRange(0.0, 1.0);
        self.spin_pred_iou_thresh.setSingleStep(0.01);
        self.spin_pred_iou_thresh.setDecimals(2)
        self.spin_pred_iou_thresh.setValue(self.current_auto_params["pred_iou_thresh"]);
        auto_params_layout.addRow("Pred IOU Thresh:", self.spin_pred_iou_thresh)
        self.spin_stability_score_thresh = QDoubleSpinBox();
        self.spin_stability_score_thresh.setRange(0.0, 1.0);
        self.spin_stability_score_thresh.setSingleStep(0.01);
        self.spin_stability_score_thresh.setDecimals(2)
        self.spin_stability_score_thresh.setValue(self.current_auto_params["stability_score_thresh"]);
        auto_params_layout.addRow("Stability Score Thresh:", self.spin_stability_score_thresh)
        self.spin_crop_n_layers = QSpinBox();
        self.spin_crop_n_layers.setRange(0, 5);
        self.spin_crop_n_layers.setSingleStep(1)
        self.spin_crop_n_layers.setValue(self.current_auto_params["crop_n_layers"]);
        auto_params_layout.addRow("Crop N Layers:", self.spin_crop_n_layers)
        self.spin_crop_n_points_downscale_factor = QSpinBox();
        self.spin_crop_n_points_downscale_factor.setRange(1, 4);
        self.spin_crop_n_points_downscale_factor.setSingleStep(1)
        self.spin_crop_n_points_downscale_factor.setValue(self.current_auto_params["crop_n_points_downscale_factor"]);
        auto_params_layout.addRow("Crop Points Downscale Factor:", self.spin_crop_n_points_downscale_factor)
        self.spin_min_mask_region_area = QSpinBox();
        self.spin_min_mask_region_area.setRange(0, 10000);
        self.spin_min_mask_region_area.setSingleStep(100)
        self.spin_min_mask_region_area.setValue(self.current_auto_params["min_mask_region_area"]);
        auto_params_layout.addRow("Min Mask Region Area:", self.spin_min_mask_region_area)
        self.btn_update_auto_params = QPushButton("Update & Re-run Automatic");
        self.btn_update_auto_params.clicked.connect(self.update_automatic_params_and_rerun)
        auto_params_layout.addRow(self.btn_update_auto_params);
        self.auto_params_group.setLayout(auto_params_layout);
        control_panel_layout.addWidget(self.auto_params_group)
        self.btn_generate_graph = QPushButton("Generate & Show Graph");
        self.btn_generate_graph.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_generate_graph.clicked.connect(self.generate_and_show_graph);
        control_panel_layout.addWidget(self.btn_generate_graph);
        control_panel_layout.addSpacing(10)
        self.clear_btn = QPushButton("Clear Segmentation");
        self.clear_btn.clicked.connect(self.image_label.clear_segmentation);
        self.clear_btn.setFont(QFont("Arial", 12))
        control_panel_layout.addWidget(self.clear_btn);
        control_panel_layout.addStretch(1)
        self.statusBar = QStatusBar();
        self.setStatusBar(self.statusBar);
        self.statusBar.showMessage("Ready. Load an image to start.")
        self.update_control_states('point')

    def update_control_states(self, current_mode: str):
        is_prompt_mode = (current_mode == 'point' or current_mode == 'bbox')
        is_auto_mode = (current_mode == 'automatic')
        has_auto_masks = self.image_label.current_automatic_masks is not None
        self.label_group.setEnabled(is_prompt_mode)
        self.segment_prompts_btn.setEnabled(is_prompt_mode and self.original_image_rgb is not None)
        self.auto_params_group.setEnabled(is_auto_mode)
        self.btn_update_auto_params.setEnabled(is_auto_mode and self.original_image_rgb is not None)
        self.stats_group.setEnabled(is_auto_mode and has_auto_masks)
        self.btn_generate_graph.setEnabled(is_auto_mode and has_auto_masks)

    def set_mode(self, mode: str, checked: bool):
        if checked:
            self.image_label.set_input_mode(mode)
            self.update_control_states(mode)
            if self.original_image_rgb is not None:
                if mode == 'automatic':
                    self.perform_automatic_segmentation()
                else:
                    self.reset_stats_display()
            else:
                self.statusBar.showMessage(f"Switched to {mode} mode. Load an image to proceed.")

    def load_sam_model(self):
        self.statusBar.showMessage(f"Loading SAM model ({MODEL_TYPE})...")
        QApplication.processEvents()
        try:
            self.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
            if torch.cuda.is_available():
                self.sam.to(device="cuda"); self.statusBar.showMessage(f"SAM model ({MODEL_TYPE}) loaded to CUDA.")
            else:
                self.statusBar.showMessage(f"SAM model ({MODEL_TYPE}) loaded to CPU.")
            self.sam_predictor = SamPredictor(self.sam)
            self.sam_automatic_predictor = SamAutomaticMaskGenerator(self.sam, **self.current_auto_params)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SAM model: {e}"); self.statusBar.showMessage(
                "Failed to load SAM model.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                self.original_image_rgb = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                if self.sam_predictor: self.sam_predictor.set_image(self.original_image_rgb)
                self.image_label.set_image(self.original_image_rgb)
                self.reset_stats_display()
                self.statusBar.showMessage(f"Image loaded: {file_path}")
                self.update_control_states(self.image_label.input_mode)
                if self.automatic_mode_radio.isChecked(): self.perform_automatic_segmentation()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}"); self.statusBar.showMessage(
                    "Failed to load image.")

    def perform_prompt_segmentation(self, point_coords: np.ndarray, point_labels: np.ndarray, bbox: np.ndarray,
                                    mode: str):
        if self.original_image_rgb is None or not self.sam_predictor: return
        has_points = point_coords is not None and point_coords.size > 0
        has_bbox = bbox is not None and bbox.size > 0
        if not (has_points or has_bbox): self.image_label.set_prompt_mask(None); return
        try:
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords if has_points else None, point_labels=point_labels if has_points else None,
                box=bbox[None, :] if has_bbox else None, multimask_output=False)
            self.image_label.set_prompt_mask(masks[0])
            self.statusBar.showMessage(f"Segmentation complete. Score: {scores[0]:.2f}")
        except Exception as e:
            QMessageBox.warning(self, "Segmentation Error", f"Error: {e}"); self.image_label.set_prompt_mask(None)

    def update_automatic_params_and_rerun(self):
        if self.sam is None: return
        self.current_auto_params.update({k: getattr(self, f"spin_{k}").value() for k in self.current_auto_params})
        self.statusBar.showMessage("Updating automatic parameters...")
        QApplication.processEvents()
        try:
            self.sam_automatic_predictor = SamAutomaticMaskGenerator(self.sam, **self.current_auto_params)
            if self.automatic_mode_radio.isChecked() and self.original_image_rgb is not None: self.perform_automatic_segmentation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update parameters: {e}")

    def perform_automatic_segmentation(self):
        if self.original_image_rgb is None or not self.sam_automatic_predictor: self.update_control_states(
            self.image_label.input_mode); return
        self.statusBar.showMessage("Performing automatic segmentation...");
        self.reset_stats_display();
        QApplication.processEvents()
        try:
            masks_data = self.sam_automatic_predictor.generate(self.original_image_rgb)
            if not masks_data:
                self.statusBar.showMessage("No objects found."); self.image_label.set_automatic_masks(None)
            else:
                self.image_label.set_automatic_masks(masks_data)
                self.calculate_and_display_stats(masks_data)
                self.statusBar.showMessage(f"Automatic segmentation complete. Found {len(masks_data)} objects.")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Automatic segmentation error: {e}"); self.image_label.set_automatic_masks(None)
        self.update_control_states(self.image_label.input_mode)

    def reset_stats_display(self):
        self.stats_mask_count_label.setText("N/A");
        self.stats_coverage_label.setText("N/A");
        self.stats_overlap_label.setText("N/A")
        self.stats_group.setEnabled(False)

    def calculate_and_display_stats(self, masks_data: list):
        if not masks_data or self.original_image_rgb is None: self.reset_stats_display(); return
        num_masks = len(masks_data);
        h, w, _ = self.original_image_rgb.shape
        self.stats_mask_count_label.setText(f"{num_masks}")
        union_mask = np.zeros((h, w), dtype=bool)
        for mask_info in masks_data: union_mask |= mask_info['segmentation']
        self.stats_coverage_label.setText(f"{(np.sum(union_mask) / (h * w)) * 100:.2f}%")
        all_masks, mask_areas = [m['segmentation'] for m in masks_data], [m['area'] for m in masks_data]
        overlapping_indices = set()
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                intersection = np.sum(all_masks[i] & all_masks[j])
                if intersection > 0 and ((intersection / mask_areas[i] > 0.1) or (intersection / mask_areas[j] > 0.1)):
                    overlapping_indices.add(i);
                    overlapping_indices.add(j)
        self.stats_overlap_label.setText(f"{len(overlapping_indices)}");
        self.stats_group.setEnabled(True)

    # --- MODIFIED: Calling function with debug logs ---
    def generate_and_show_graph(self):
        print("\n--- DEBUG LOG ---: 'Generate & Show Graph' button clicked.", flush=True)
        if self.image_label.current_automatic_masks is None:
            QMessageBox.information(self, "No Masks", "Please run automatic segmentation first.")
            return

        self.statusBar.showMessage("Generating graph from mask centroids...")
        QApplication.processEvents()

        nodes, edges = None, None
        try:
            print("--- DEBUG LOG ---: About to call the 'create_graph_from_masks' helper function.", flush=True)
            nodes, edges = create_graph_from_masks(
                self.image_label.current_automatic_masks, connectivity='knn', knn_k=5
            )
            print("--- DEBUG LOG ---: Returned from 'create_graph_from_masks' helper function.", flush=True)
        except Exception as e:
            # This will probably NOT catch the 0xC0000409 crash, but is good practice
            print(f"--- DEBUG LOG ---: A Python-level exception occurred during graph creation: {e}", flush=True)
            QMessageBox.critical(self, "Graph Error", f"A Python error occurred during graph creation:\n{e}")
            self.image_label.graph_nodes = None
            self.image_label.graph_edges = None
            self.image_label.update()
            return

        if nodes is None or edges is None:
            self.statusBar.showMessage("Graph generation failed or not enough masks.")
            self.image_label.graph_nodes, self.image_label.graph_edges = None, None
        else:
            self.image_label.graph_nodes, self.image_label.graph_edges = nodes, edges
            self.statusBar.showMessage(f"Graph generated with {len(nodes)} nodes and {edges.shape[1]} edges.")

        print("--- DEBUG LOG ---: Requesting repaint of the image label to show the graph.", flush=True)
        self.image_label.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMSegmentationApp()
    window.show()
    sys.exit(app.exec_())