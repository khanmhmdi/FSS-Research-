import sys
import cv2
import numpy as np
import torch  # Import torch for CUDA check

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QRadioButton, QGroupBox, QStatusBar, QMessageBox,
    QButtonGroup, QFormLayout, QSpinBox, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal

# SAM imports
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# --- Configuration ---
SAM_CHECKPOINT_PATH = r"C:\Users\Khan\PycharmProjects\FSS-Research-\model\sam_vit_h_4b8939.pth"  # Make sure this path is correct!
MODEL_TYPE = "vit_h"  # Matches the checkpoint type (e.g., "vit_h", "vit_l", "vit_b")

# Initial configuration for SamAutomaticMaskGenerator
# These will be the default values shown in the UI
DEFAULT_SAM_AUTO_GENERATOR_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,  # Minimum number of pixels in a segmented mask
}


# --- Custom Image Display Widget ---
class ImageDisplayLabel(QLabel):
    # Signal emitted when a prompt-based segmentation request should be made
    promptSegmentRequested = pyqtSignal(object, object, np.ndarray, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking for live bbox drawing
        self.setCursor(Qt.CrossCursor)  # Change cursor to crosshair

        self.original_image_rgb = None  # Stores the original image (NumPy array, RGB)
        # self.display_pixmap is no longer strictly needed as we draw directly on QLabel in paintEvent
        # but kept as a reference for the base image for setPixmap.

        # For prompt-based segmentation (points/bbox)
        self.current_prompt_mask = None  # Stores the latest single segmentation mask
        self.current_points = []  # List of (x, y, label) tuples for points (label: 1=fg, 0=bg)
        self.current_bbox = None  # (x1, y1, x2, y2) tuple for bbox

        # For automatic segmentation
        self.current_automatic_masks = None  # List of masks from automatic generator

        self.input_mode = 'point'  # 'point', 'bbox', or 'automatic'
        self.point_label = 1  # 1 (foreground) or 0 (background)

        # For bounding box drawing
        self.start_point = None
        self.end_point = None
        self.is_drawing_bbox = False

        self.setMinimumSize(300, 300)  # Ensure it has some initial size
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow it to expand

    def set_image(self, image_rgb: np.ndarray):
        """Sets the image to be displayed and resets all segmentation state."""
        self.original_image_rgb = image_rgb
        self.clear_segmentation()  # Clear all masks and inputs
        self._update_base_pixmap()  # Update the base image shown by QLabel

    def set_prompt_mask(self, mask: np.ndarray):
        """Sets the segmentation mask for prompt-based modes."""
        self.current_prompt_mask = mask
        self.current_automatic_masks = None  # Clear automatic masks if present
        self.update()  # Request repaint

    def set_automatic_masks(self, masks: list):
        """Sets multiple segmentation masks for automatic mode."""
        self.current_automatic_masks = masks  # masks is a list of {"segmentation": bool_mask, ...} dicts
        self.current_prompt_mask = None  # Clear prompt mask if present
        self.current_points = []
        self.current_bbox = None
        self.update()  # Request repaint

    def set_input_mode(self, mode: str):
        """Sets whether the user is providing points, bbox, or in automatic mode."""
        self.input_mode = mode
        self.clear_segmentation()  # Clear all inputs and outputs when changing mode
        self.update()  # Request repaint

    def set_point_label(self, label: int):
        """Sets the label for new points (1 for foreground, 0 for background)."""
        self.point_label = label

    def clear_segmentation(self):
        """Clears all points, bbox, and all current masks."""
        self.current_points = []
        self.current_bbox = None
        self.current_prompt_mask = None
        self.current_automatic_masks = None
        self.is_drawing_bbox = False
        self.start_point = None
        self.end_point = None
        self.update()  # Request repaint
        # Explicitly trigger a segment request to clear SAM's internal state
        if self.input_mode in ['point', 'bbox']:
            self.trigger_segmentation_request()

    def _update_base_pixmap(self):
        """Updates the base QPixmap displayed by the QLabel."""
        if self.original_image_rgb is None:
            self.setPixmap(QPixmap())  # Clear display
            return

        h, w, ch = self.original_image_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(self.original_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale pixmap to fit the label, keeping aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)  # Set the scaled pixmap to the QLabel
        # No need to call update() here, setPixmap() internally triggers a paint event.

    def _calculate_display_geometry(self):
        """
        Helper to calculate the actual displayed image size and offsets within the QLabel,
        considering Qt.KeepAspectRatio scaling.
        Returns: img_w, img_h, display_w, display_h, x_offset, y_offset
        """
        if self.original_image_rgb is None:
            return 0, 0, 0, 0, 0, 0

        img_h, img_w, _ = self.original_image_rgb.shape
        label_w, label_h = self.width(), self.height()

        # Handle case where label size might be zero initially
        if label_w == 0 or label_h == 0:
            return img_w, img_h, 0, 0, 0, 0

        # Calculate actual displayed dimensions based on Qt.KeepAspectRatio
        aspect_ratio_label = label_w / label_h
        aspect_ratio_img = img_w / img_h

        if aspect_ratio_img > aspect_ratio_label:
            # Image is wider than label's aspect ratio, fills width, height is scaled down
            display_w = label_w
            display_h = int(label_w / aspect_ratio_img)
        else:
            # Label is wider than image's aspect ratio, image fills height, width is scaled down
            display_h = label_h
            display_w = int(label_h * aspect_ratio_img)

        # Calculate centering offsets
        x_offset = (label_w - display_w) // 2
        y_offset = (label_h - display_h) // 2

        return img_w, img_h, display_w, display_h, x_offset, y_offset

    def get_scaled_coordinates(self, mouse_x, mouse_y):
        """
        Converts mouse event coordinates (on the QLabel) to
        original image coordinates, respecting aspect ratio scaling.
        """
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()
        if img_w == 0 or display_w == 0 or display_h == 0:  # No image loaded or label size is zero
            return None, None

        # Map mouse coordinates relative to the displayed image area
        x_on_img_display = mouse_x - x_offset
        y_on_img_display = mouse_y - y_offset

        # Check if mouse is within the actual displayed image area
        if not (0 <= x_on_img_display < display_w and 0 <= y_on_img_display < display_h):
            return None, None  # Mouse outside the actual image display

        # Scale to original image coordinates
        original_x = int(x_on_img_display * (img_w / display_w))
        original_y = int(y_on_img_display * (img_h / display_h))

        # Clamp coordinates to image boundaries (safe practice)
        original_x = max(0, min(original_x, img_w - 1))
        original_y = max(0, min(original_y, img_h - 1))

        return original_x, original_y

    def get_display_coordinates(self, original_x, original_y):
        """
        Converts original image coordinates to display coordinates on the QLabel,
        respecting aspect ratio scaling. Used for drawing points/bbox.
        """
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()
        if img_w == 0 or display_w == 0 or display_h == 0:  # No image loaded or label size is zero
            return None, None

        # Scale from original to displayed image coordinates, then add offset
        display_x = int(original_x * (display_w / img_w) + x_offset)
        display_y = int(original_y * (display_h / img_h) + y_offset)

        return display_x, display_y

    def mousePressEvent(self, event):
        # Ignore mouse events if no image is loaded or in automatic mode
        if self.original_image_rgb is None or self.input_mode == 'automatic':
            return

        original_x, original_y = self.get_scaled_coordinates(event.x(), event.y())
        if original_x is None:  # Clicked outside the image area
            return

        if self.input_mode == 'point':
            self.current_points.append((original_x, original_y, self.point_label))
            self.update()  # Update visualization immediately
            # Segmentation will be triggered by button click
        elif self.input_mode == 'bbox':
            self.is_drawing_bbox = True
            self.start_point = QPoint(event.x(), event.y())
            self.end_point = self.start_point
            self.update()  # Force repaint to show starting point of bbox

    def mouseMoveEvent(self, event):
        # Ignore mouse events if no image is loaded or in automatic mode
        if self.original_image_rgb is None or self.input_mode == 'automatic':
            return

        if self.input_mode == 'bbox' and self.is_drawing_bbox:
            self.end_point = QPoint(event.x(), event.y())
            self.update()  # Force repaint to show live bbox

    def mouseReleaseEvent(self, event):
        # Ignore mouse events if no image is loaded or in automatic mode
        if self.original_image_rgb is None or self.input_mode == 'automatic':
            return

        if self.input_mode == 'bbox' and self.is_drawing_bbox:
            self.is_drawing_bbox = False

            ox1, oy1 = self.get_scaled_coordinates(self.start_point.x(), self.start_point.y())
            ox2, oy2 = self.get_scaled_coordinates(event.x(), event.y())

            if ox1 is None or ox2 is None:
                self.current_bbox = None
                self.update()  # Request repaint to clear live bbox
                return

            x1 = min(ox1, ox2)
            y1 = min(oy1, oy2)
            x2 = max(ox1, ox2)
            y2 = max(oy1, oy2)

            if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:  # Ignore very small accidental bboxes
                self.current_bbox = None
            else:
                self.current_bbox = np.array([x1, y1, x2, y2])

            self.update()  # Redraw with confirmed bbox
            # Segmentation will be triggered by button click

        self.start_point = None
        self.end_point = None

    def trigger_segmentation_request(self):
        """Public method to emit the segmentation signal, typically called by a button."""
        input_point_coords = None
        input_point_labels = None
        if self.current_points:
            input_point_coords = np.array([p[:2] for p in self.current_points])
            input_point_labels = np.array([p[2] for p in self.current_points])

        # Ensure bbox_to_emit is always a NumPy array, even if current_bbox is None
        bbox_to_emit = self.current_bbox if self.current_bbox is not None else np.array([])

        self.promptSegmentRequested.emit(input_point_coords, input_point_labels, bbox_to_emit, self.input_mode)

    def paintEvent(self, event):
        """
        Overrides QLabel's paintEvent to draw the base image and then overlays.
        """
        super().paintEvent(event)  # Let QLabel draw its pixmap first

        if self.original_image_rgb is None or self.pixmap() is None:
            return

        painter = QPainter(self)  # Create painter on the widget itself

        # Get the actual displayed image dimensions and offsets
        img_w, img_h, display_w, display_h, x_offset, y_offset = self._calculate_display_geometry()

        # Draw the segmentation mask(s)
        if self.current_automatic_masks is not None:
            for i, mask_data in enumerate(self.current_automatic_masks):
                mask = mask_data["segmentation"]
                color_hue = (i * 0.6180339887) % 1.0
                color = QColor.fromHsvF(color_hue, 0.8, 0.9, 0.5)

                # Create mask image at original image resolution
                mask_rgb = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                mask_rgb[mask] = [color.red(), color.green(), color.blue(), color.alpha()]
                mask_q_image = QImage(mask_rgb.data, img_w, img_h, img_w * 4, QImage.Format_ARGB32)

                # Scale mask QImage to the actual displayed image size
                scaled_mask_q_image = mask_q_image.scaled(
                    display_w, display_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
                # Draw the scaled mask at the correct offset on the QLabel
                painter.drawImage(x_offset, y_offset, scaled_mask_q_image)

        elif self.current_prompt_mask is not None:
            mask = self.current_prompt_mask
            mask_rgb = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgb[mask] = [0, 150, 255, 128]
            mask_q_image = QImage(mask_rgb.data, img_w, img_h, img_w * 4, QImage.Format_ARGB32)

            scaled_mask_q_image = mask_q_image.scaled(
                display_w, display_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            painter.drawImage(x_offset, y_offset, scaled_mask_q_image)

        # Draw points (only relevant for point mode)
        if self.input_mode == 'point':
            for ox, oy, label in self.current_points:
                dx, dy = self.get_display_coordinates(ox, oy)
                if dx is not None:  # Ensure coordinates are valid for display
                    color = Qt.green if label == 1 else Qt.red
                    painter.setBrush(color)
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(dx - 5, dy - 5, 10, 10)  # 10x10 circle

        # Draw bounding box (only relevant for bbox mode)
        if self.input_mode == 'bbox':
            if self.current_bbox is not None:
                x1, y1, x2, y2 = self.current_bbox
                dx1, dy1 = self.get_display_coordinates(x1, y1)
                dx2, dy2 = self.get_display_coordinates(x2, y2)
                if dx1 is not None:
                    painter.setPen(QPen(Qt.blue, 2))
                    painter.drawRect(dx1, dy1, dx2 - dx1, dy2 - dy1)

            # Draw live bounding box while drawing (widget coordinates)
            if self.is_drawing_bbox and self.start_point and self.end_point:
                painter.setPen(QPen(Qt.magenta, 2, Qt.DotLine))
                x = min(self.start_point.x(), self.end_point.x())
                y = min(self.start_point.y(), self.end_point.y())
                w = abs(self.end_point.x() - self.start_point.x())
                h = abs(self.end_point.y() - self.start_point.y())
                painter.drawRect(x, y, w, h)

        painter.end()

    def resizeEvent(self, event):
        """Called when the widget is resized."""
        super().resizeEvent(event)
        self._update_base_pixmap()  # Re-scale base pixmap when size changes
        # paintEvent will be called automatically after this.


# --- Main Application Window ---
class SAMSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Interactive & Automatic Segmentation")
        self.setGeometry(100, 100, 1200, 800)  # Slightly wider window

        self.original_image_rgb = None  # Stores the current loaded image (NumPy RGB)
        self.sam = None  # Store the base SAM model to re-initialize automatic generator
        self.sam_predictor = None
        self.sam_automatic_predictor = None
        self.current_auto_params = DEFAULT_SAM_AUTO_GENERATOR_PARAMS.copy()  # Editable copy

        self.init_ui()
        self.load_sam_model()

    def init_ui(self):
        # --- Central Widget & Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Image Display Area ---
        image_layout = QVBoxLayout()
        self.image_label = ImageDisplayLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.image_label.promptSegmentRequested.connect(self.perform_prompt_segmentation)
        image_layout.addWidget(self.image_label)
        main_layout.addLayout(image_layout, 3)  # Give more space to image

        # --- Control Panel ---
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setAlignment(Qt.AlignTop)
        main_layout.addLayout(control_panel_layout, 1)  # Less space for controls

        # Load Image Button
        load_image_btn = QPushButton("Load Image")
        load_image_btn.clicked.connect(self.load_image)
        load_image_btn.setFont(QFont("Arial", 12))
        control_panel_layout.addWidget(load_image_btn)
        control_panel_layout.addSpacing(20)

        # Input Mode Group
        mode_group = QGroupBox("Segmentation Mode")
        mode_layout = QVBoxLayout()

        self.mode_button_group = QButtonGroup(self)  # Manages exclusive selection

        self.point_mode_radio = QRadioButton("Interactive: Points (Click)")
        self.point_mode_radio.setChecked(True)  # Default
        self.point_mode_radio.toggled.connect(lambda checked: self.set_mode('point', checked))
        self.mode_button_group.addButton(self.point_mode_radio)

        self.bbox_mode_radio = QRadioButton("Interactive: Bounding Box (Drag)")
        self.bbox_mode_radio.toggled.connect(lambda checked: self.set_mode('bbox', checked))
        self.mode_button_group.addButton(self.bbox_mode_radio)

        self.automatic_mode_radio = QRadioButton("Automatic: All Objects")
        self.automatic_mode_radio.toggled.connect(lambda checked: self.set_mode('automatic', checked))
        self.mode_button_group.addButton(self.automatic_mode_radio)

        mode_layout.addWidget(self.point_mode_radio)
        mode_layout.addWidget(self.bbox_mode_radio)
        mode_layout.addWidget(self.automatic_mode_radio)
        mode_group.setLayout(mode_layout)
        control_panel_layout.addWidget(mode_group)
        control_panel_layout.addSpacing(10)

        # Segment Prompts Button (New)
        self.segment_prompts_btn = QPushButton("Segment (Apply Prompts)")
        self.segment_prompts_btn.clicked.connect(self.image_label.trigger_segmentation_request)
        self.segment_prompts_btn.setFont(QFont("Arial", 12, QFont.Bold))  # Make it stand out
        control_panel_layout.addWidget(self.segment_prompts_btn)
        control_panel_layout.addSpacing(10)

        # Point Label Group (for points mode)
        self.label_group = QGroupBox("Point Label (for Points Mode)")
        label_layout = QVBoxLayout()
        self.fg_label_radio = QRadioButton("Foreground (Green)")
        self.fg_label_radio.setChecked(True)  # Default
        self.fg_label_radio.toggled.connect(lambda: self.image_label.set_point_label(1))
        self.bg_label_radio = QRadioButton("Background (Red)")
        self.bg_label_radio.toggled.connect(lambda: self.image_label.set_point_label(0))
        label_layout.addWidget(self.fg_label_radio)
        label_layout.addWidget(self.bg_label_radio)
        self.label_group.setLayout(label_layout)
        control_panel_layout.addWidget(self.label_group)
        control_panel_layout.addSpacing(20)

        # Automatic Segmentation Parameters Group
        self.auto_params_group = QGroupBox("Automatic Segmentation Parameters")
        auto_params_layout = QFormLayout()

        # points_per_side
        self.spin_points_per_side = QSpinBox()
        self.spin_points_per_side.setRange(8, 128)
        self.spin_points_per_side.setSingleStep(8)
        self.spin_points_per_side.setValue(self.current_auto_params["points_per_side"])
        auto_params_layout.addRow("Points Per Side:", self.spin_points_per_side)

        # pred_iou_thresh
        self.spin_pred_iou_thresh = QDoubleSpinBox()
        self.spin_pred_iou_thresh.setRange(0.0, 1.0)
        self.spin_pred_iou_thresh.setSingleStep(0.01)
        self.spin_pred_iou_thresh.setDecimals(2)
        self.spin_pred_iou_thresh.setValue(self.current_auto_params["pred_iou_thresh"])
        auto_params_layout.addRow("Pred IOU Thresh:", self.spin_pred_iou_thresh)

        # stability_score_thresh
        self.spin_stability_score_thresh = QDoubleSpinBox()
        self.spin_stability_score_thresh.setRange(0.0, 1.0)
        self.spin_stability_score_thresh.setSingleStep(0.01)
        self.spin_stability_score_thresh.setDecimals(2)
        self.spin_stability_score_thresh.setValue(self.current_auto_params["stability_score_thresh"])
        auto_params_layout.addRow("Stability Score Thresh:", self.spin_stability_score_thresh)

        # crop_n_layers
        self.spin_crop_n_layers = QSpinBox()
        self.spin_crop_n_layers.setRange(0, 5)
        self.spin_crop_n_layers.setSingleStep(1)
        self.spin_crop_n_layers.setValue(self.current_auto_params["crop_n_layers"])
        auto_params_layout.addRow("Crop N Layers:", self.spin_crop_n_layers)

        # crop_n_points_downscale_factor
        self.spin_crop_n_points_downscale_factor = QSpinBox()
        self.spin_crop_n_points_downscale_factor.setRange(1, 4)
        self.spin_crop_n_points_downscale_factor.setSingleStep(1)
        self.spin_crop_n_points_downscale_factor.setValue(self.current_auto_params["crop_n_points_downscale_factor"])
        auto_params_layout.addRow("Crop Points Downscale Factor:", self.spin_crop_n_points_downscale_factor)

        # min_mask_region_area
        self.spin_min_mask_region_area = QSpinBox()
        self.spin_min_mask_region_area.setRange(0, 10000)
        self.spin_min_mask_region_area.setSingleStep(100)
        self.spin_min_mask_region_area.setValue(self.current_auto_params["min_mask_region_area"])
        auto_params_layout.addRow("Min Mask Region Area:", self.spin_min_mask_region_area)

        self.btn_update_auto_params = QPushButton("Update & Re-run Automatic")
        self.btn_update_auto_params.clicked.connect(self.update_automatic_params_and_rerun)
        auto_params_layout.addRow(self.btn_update_auto_params)

        self.auto_params_group.setLayout(auto_params_layout)
        control_panel_layout.addWidget(self.auto_params_group)
        control_panel_layout.addSpacing(20)

        # Clear Button
        clear_btn = QPushButton("Clear Segmentation")
        clear_btn.clicked.connect(self.image_label.clear_segmentation)
        clear_btn.setFont(QFont("Arial", 12))
        control_panel_layout.addWidget(clear_btn)

        control_panel_layout.addStretch(1)  # Pushes everything to the top

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load an image to start.")

        # Initial state of controls
        self.update_control_states('point')

    def update_control_states(self, current_mode: str):
        """Enables/disables controls based on the current segmentation mode."""
        is_prompt_mode = (current_mode == 'point' or current_mode == 'bbox')
        is_auto_mode = (current_mode == 'automatic')

        self.label_group.setEnabled(is_prompt_mode)
        # Enable segment prompts button only if an image is loaded and in prompt mode
        self.segment_prompts_btn.setEnabled(is_prompt_mode and self.original_image_rgb is not None)

        self.auto_params_group.setEnabled(is_auto_mode)
        # The 'Update & Re-run Automatic' button inside the group also needs to be enabled if in auto mode and image loaded
        self.btn_update_auto_params.setEnabled(is_auto_mode and self.original_image_rgb is not None)

    def set_mode(self, mode: str, checked: bool):
        """Handles changes in the segmentation mode radio buttons."""
        if checked:  # Only act when a radio button is selected (not unselected)
            self.image_label.set_input_mode(mode)
            self.update_control_states(mode)

            if self.original_image_rgb is not None:
                if mode == 'automatic':
                    self.perform_automatic_segmentation()
                else:
                    # Clear automatic masks if switching back to interactive
                    self.image_label.clear_segmentation()  # This will also trigger prompt segment request
            else:
                self.statusBar.showMessage(f"Switched to {mode} mode. Load an image to proceed.")

    def load_sam_model(self):
        """Loads the SAM model and initializes the predictors."""
        self.statusBar.showMessage(f"Loading SAM model ({MODEL_TYPE})... This might take a moment.")
        QApplication.processEvents()  # Update GUI to show message

        try:
            self.sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)

            # --- CUDA (GPU) Activation ---
            if torch.cuda.is_available():
                self.sam.to(device="cuda")
                self.statusBar.showMessage(f"SAM model ({MODEL_TYPE}) loaded to CUDA.")
            else:
                self.statusBar.showMessage(f"SAM model ({MODEL_TYPE}) loaded to CPU. CUDA not available.")
            # ---------------------------

            self.sam_predictor = SamPredictor(self.sam)
            # Initialize automatic predictor with current parameters
            self.sam_automatic_predictor = SamAutomaticMaskGenerator(self.sam, **self.current_auto_params)

            self.statusBar.showMessage(f"SAM model ({MODEL_TYPE}) loaded successfully.")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 f"SAM checkpoint not found at: {SAM_CHECKPOINT_PATH}\n"
                                 f"Please download it and place it in the correct directory.")
            self.statusBar.showMessage("Failed to load SAM model. Check path.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SAM model: {e}")
            self.statusBar.showMessage("Failed to load SAM model.")

    def load_image(self):
        """Opens a file dialog to select and load an image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            try:
                image_bgr = cv2.imread(file_path)
                if image_bgr is None:
                    raise ValueError(f"Could not load image: {file_path}")

                self.original_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Set the image for SAM predictor (for interactive modes)
                if self.sam_predictor:
                    self.sam_predictor.set_image(self.original_image_rgb)

                # Set the image for display in our custom QLabel
                self.image_label.set_image(self.original_image_rgb)
                self.statusBar.showMessage(f"Image loaded: {file_path}")

                # Update control states after image loaded (to enable segment buttons)
                self.update_control_states(self.image_label.input_mode)

                # If currently in automatic mode, run segmentation immediately
                if self.automatic_mode_radio.isChecked():
                    self.perform_automatic_segmentation()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
                self.statusBar.showMessage("Failed to load image.")

    def perform_prompt_segmentation(self, point_coords: np.ndarray, point_labels: np.ndarray, bbox: np.ndarray,
                                    mode: str):
        """
        Performs segmentation using SAM based on the provided points or bbox.
        This slot is connected to the promptSegmentRequested signal from ImageDisplayLabel.
        """
        if self.original_image_rgb is None:
            self.statusBar.showMessage("No image loaded.")
            self.image_label.set_prompt_mask(None)  # Clear any old mask
            return

        if not self.sam_predictor:
            self.statusBar.showMessage("SAM model not loaded.")
            self.image_label.set_prompt_mask(None)
            return

        # Determine if there's enough input for SAM
        has_points = point_coords is not None and point_coords.size > 0
        has_bbox = bbox is not None and bbox.size > 0

        if not (has_points or has_bbox):
            # No valid input, clear mask
            self.image_label.set_prompt_mask(None)
            self.statusBar.showMessage("Ready for input (click 'Segment' button).")
            return

        input_point = point_coords if has_points else None
        input_label = point_labels if has_points else None
        input_box = bbox[None, :] if has_bbox else None

        if mode == 'point' and has_points:
            self.statusBar.showMessage(f"Segmenting with {len(point_coords)} point(s)...")
        elif mode == 'bbox' and has_bbox:
            self.statusBar.showMessage("Segmenting with bounding box...")
        else:
            self.image_label.set_prompt_mask(None)
            self.statusBar.showMessage("Invalid prompt input combination.")
            return

        try:
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            )

            predicted_mask = masks[0]
            self.image_label.set_prompt_mask(predicted_mask)
            self.statusBar.showMessage(f"Segmentation complete. Score: {scores[0]:.2f}")

        except Exception as e:
            QMessageBox.warning(self, "Segmentation Error", f"An error occurred during segmentation: {e}")
            self.statusBar.showMessage("Segmentation failed.")
            self.image_label.set_prompt_mask(None)

    def update_automatic_params_and_rerun(self):
        """Reads current parameter values from spin boxes, updates the generator, and reruns segmentation."""
        if self.sam is None:
            QMessageBox.warning(self, "Model Not Loaded", "SAM model must be loaded before updating parameters.")
            return

        self.current_auto_params["points_per_side"] = self.spin_points_per_side.value()
        self.current_auto_params["pred_iou_thresh"] = self.spin_pred_iou_thresh.value()
        self.current_auto_params["stability_score_thresh"] = self.spin_stability_score_thresh.value()
        self.current_auto_params["crop_n_layers"] = self.spin_crop_n_layers.value()
        self.current_auto_params["crop_n_points_downscale_factor"] = self.spin_crop_n_points_downscale_factor.value()
        self.current_auto_params["min_mask_region_area"] = self.spin_min_mask_region_area.value()

        self.statusBar.showMessage("Updating automatic segmentation parameters...")
        QApplication.processEvents()
        try:
            self.sam_automatic_predictor = SamAutomaticMaskGenerator(self.sam, **self.current_auto_params)
            self.statusBar.showMessage("Automatic segmentation parameters updated.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update automatic parameters: {e}")
            self.statusBar.showMessage("Failed to update parameters.")
            return

        if self.automatic_mode_radio.isChecked() and self.original_image_rgb is not None:
            self.perform_automatic_segmentation()
        else:
            self.statusBar.showMessage("Parameters updated. Switch to 'Automatic: All Objects' mode to apply.")

    def perform_automatic_segmentation(self):
        """Performs automatic segmentation of all objects in the image."""
        if self.original_image_rgb is None:
            self.statusBar.showMessage("No image loaded for automatic segmentation.")
            self.image_label.set_automatic_masks(None)
            return

        if not self.sam_automatic_predictor:
            self.statusBar.showMessage("SAM automatic predictor not loaded.")
            self.image_label.set_automatic_masks(None)
            return

        self.statusBar.showMessage("Performing automatic segmentation... This might take a while for large images.")
        QApplication.processEvents()

        try:
            masks_data = self.sam_automatic_predictor.generate(self.original_image_rgb)

            if not masks_data:
                self.statusBar.showMessage("No objects found by automatic segmentation.")
                self.image_label.set_automatic_masks(None)
                return

            self.image_label.set_automatic_masks(masks_data)
            self.statusBar.showMessage(f"Automatic segmentation complete. Found {len(masks_data)} objects.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during automatic segmentation: {e}")
            self.statusBar.showMessage("Automatic segmentation failed.")
            self.image_label.set_automatic_masks(None)


# --- Main Application Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMSegmentationApp()
    window.show()
    sys.exit(app.exec_())