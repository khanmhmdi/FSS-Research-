import sys
import torch
import networkx as nx
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QFileDialog, QAction, QSplitter, QTabWidget
)
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# --- Matplotlib Canvas Widgets ---

class MplCanvas(FigureCanvas):
    """Base class for a matplotlib canvas widget to embed in PyQt."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class GraphCanvas(MplCanvas):
    """A canvas for visualizing the graph with clickable nodes."""
    nodeSelected = pyqtSignal(int)

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.node_positions = None
        self.graph = None

    def plot(self, graph_data):
        self.axes.cla()
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(graph_data.num_nodes))
        self.graph.add_edges_from(graph_data.edge_index.t().tolist())
        node_colors = ['#4CAF50' if label == 1.0 else '#F44336' for label in graph_data.y]
        self.node_positions = nx.spring_layout(self.graph, seed=42)
        nodes = nx.draw_networkx_nodes(
            self.graph, self.node_positions, ax=self.axes, node_color=node_colors, node_size=150
        )
        nodes.set_picker(5)
        nx.draw_networkx_edges(self.graph, self.node_positions, ax=self.axes, alpha=0.6)
        self.axes.set_title("Candidate Mask Graph")
        self.axes.set_xticks([]);
        self.axes.set_yticks([])
        self.draw()

    def on_pick(self, event):
        if event.artist and hasattr(event.artist, 'get_offsets') and len(event.ind) > 0:
            self.nodeSelected.emit(event.ind[0])


class MaskCanvas(MplCanvas):
    """A simple canvas for displaying a single mask image."""

    def plot_mask(self, mask_np, title="Mask"):
        self.axes.cla()
        if mask_np is not None:
            self.axes.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        self.axes.set_title(title)
        self.axes.set_xticks([]);
        self.axes.set_yticks([])
        self.fig.tight_layout()
        self.draw()


class HistogramCanvas(MplCanvas):
    """A canvas for displaying a feature vector as a histogram."""

    def plot_histogram(self, data, title="Feature Distribution", bins=50):
        self.axes.cla()
        if data is not None:
            self.axes.hist(data, bins=bins, color='#007ACC', alpha=0.75)
        self.axes.set_title(title)
        self.axes.set_xlabel("Feature Value");
        self.axes.set_ylabel("Frequency")
        self.fig.tight_layout()
        self.draw()


# --- NEW CANVAS FOR FEATURE IMAGE ---
class FeatureImageCanvas(MplCanvas):
    """A canvas for displaying a 1D feature vector as a 2D image."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.cbar = None  # To hold the colorbar object

    def _get_image_shape(self, n):
        """Helper to find a near-square shape for a 1D vector of length n."""
        sqrt_n = int(math.sqrt(n))
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return (i, n // i)
        return (1, n)  # Fallback

    def plot_feature_image(self, data, title="Feature as Image"):
        """Reshapes and plots a 1D feature vector."""
        # Clear the entire figure to handle the colorbar correctly on updates
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)

        if data is not None:
            shape = self._get_image_shape(len(data))
            image_data = data.reshape(shape)
            im = self.axes.imshow(image_data, cmap='viridis', aspect='auto')
            # Add a colorbar to show the feature value scale
            self.cbar = self.fig.colorbar(im, ax=self.axes)
            self.cbar.set_label('Feature Value')

        self.axes.set_title(title)
        self.axes.set_xticks([]);
        self.axes.set_yticks([])
        self.fig.tight_layout()
        self.draw()


# --- Main Application Window ---

class GraphViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyG Graph Data Viewer")
        self.setGeometry(100, 100, 1400, 800)
        self.graph_data = None
        self.init_ui()
        self.create_actions()
        self.create_menus()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        h_splitter = QSplitter(Qt.Horizontal)

        left_panel = QSplitter(Qt.Vertical)
        self.graph_canvas = GraphCanvas(self)
        self.info_panel = QTextEdit("Load a graph file to see details...")
        self.info_panel.setReadOnly(True)
        left_panel.addWidget(self.graph_canvas)
        left_panel.addWidget(self.info_panel)
        left_panel.setSizes([600, 200])

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        mask_layout = QHBoxLayout()
        self.gt_mask_canvas = MaskCanvas(self)
        self.candidate_mask_canvas = MaskCanvas(self)
        mask_layout.addWidget(self.gt_mask_canvas)
        mask_layout.addWidget(self.candidate_mask_canvas)

        self.inspector_label = QLabel("Node Inspector (Click a node on the graph)")
        self.inspector_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.feature_tabs = QTabWidget()

        # Tab 1: Raw Data
        raw_data_widget = QWidget()
        raw_data_layout = QVBoxLayout(raw_data_widget)
        self.raw_feature_display = QTextEdit("Node features will be shown here.")
        self.raw_feature_display.setReadOnly(True)
        self.raw_feature_display.setFontFamily("Courier New")
        raw_data_layout.addWidget(self.raw_feature_display)
        self.feature_tabs.addTab(raw_data_widget, "Raw Feature Data")

        # Tab 2: Visualized Features
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        self.histogram_canvas = HistogramCanvas(self)
        self.labeled_feature_display = QTextEdit("Labeled features and stats will be shown here.")
        self.labeled_feature_display.setReadOnly(True)
        viz_layout.addWidget(self.histogram_canvas)
        viz_layout.addWidget(self.labeled_feature_display)
        self.feature_tabs.addTab(viz_widget, "Visualized Features")

        # --- NEW Tab 3: Feature as Image ---
        img_feat_widget = QWidget()
        img_feat_layout = QVBoxLayout(img_feat_widget)
        self.feature_image_canvas = FeatureImageCanvas(self)
        img_feat_layout.addWidget(self.feature_image_canvas)
        self.feature_tabs.addTab(img_feat_widget, "Feature as Image")
        # --- END NEW TAB ---

        right_layout.addWidget(self.inspector_label)
        right_layout.addLayout(mask_layout)
        right_layout.addWidget(self.feature_tabs)
        right_panel.setLayout(right_layout)

        h_splitter.addWidget(left_panel)
        h_splitter.addWidget(right_panel)
        h_splitter.setSizes([800, 600])

        main_layout.addWidget(h_splitter)
        self.graph_canvas.nodeSelected.connect(self.update_node_inspector)

    def create_actions(self):
        self.open_action = QAction("&Open Graph File...", self)
        self.open_action.triggered.connect(self.open_graph_file)
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)

    def create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

    def open_graph_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Precomputed Graph File", "", "PyTorch Files (*.pt)")
        if not file_name: return
        try:
            self.graph_data = torch.load(file_name, map_location='cpu', weights_only=False)
            self.statusBar().showMessage(f"Successfully loaded {file_name}", 5000)
            self.update_all_views()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {e}", 5000)
            self.graph_data = None

    def update_all_views(self):
        if not self.graph_data: return

        self.graph_canvas.plot(self.graph_data)
        self.info_panel.setHtml(f"""<b>Graph Information</b><br>
            -------------------<br>
            <b>Query Class:</b> {self.graph_data.query_class}<br>
            <b>Query Index:</b> {self.graph_data.query_index}<br>
            <b>Image Size:</b> {self.graph_data.image_size}<br>
            <b>Num Nodes:</b> {self.graph_data.num_nodes}<br>
            <b>Num Edges:</b> {self.graph_data.edge_index.shape[1]}<br>""")
        self.gt_mask_canvas.plot_mask(
            self.graph_data.gt_mask_np.astype(float), title="Query Ground Truth Mask"
        )

        self.candidate_mask_canvas.plot_mask(None, "Candidate Mask")
        self.inspector_label.setText("Node Inspector (Click a node on the graph)")
        self.raw_feature_display.setText("Node features will be shown here.")
        self.labeled_feature_display.setText("Labeled features and stats will be shown here.")
        self.histogram_canvas.plot_histogram(None, "Feature Distribution")
        # Clear the new canvas as well
        self.feature_image_canvas.plot_feature_image(None, "Feature as Image")

    def update_node_inspector(self, node_id):
        if not self.graph_data or node_id >= self.graph_data.num_nodes: return

        node_label_text = "Positive" if self.graph_data.y[node_id] == 1.0 else "Negative"
        self.inspector_label.setText(f"Node Inspector: Node {node_id} (Label: {node_label_text})")
        self.candidate_mask_canvas.plot_mask(
            self.graph_data.node_masks_np[node_id].astype(float), title=f"Candidate Mask for Node {node_id}"
        )

        x_base_vec = self.graph_data.x_base[node_id].numpy()
        x_geo_vec = self.graph_data.x_geo[node_id].numpy()
        x_sam_vec = self.graph_data.x_sam[node_id].numpy()

        # Update Tab 1: Raw Data
        self.raw_feature_display.setText(f"""<font face="Courier New">
            <b>x_base (Shape: {x_base_vec.shape}):</b><br>{np.array2string(x_base_vec, precision=4, max_line_width=100)}<br><br>
            <b>x_geo (Shape: {x_geo_vec.shape}):</b><br>{np.array2string(x_geo_vec, precision=4)}<br><br>
            <b>x_sam (Shape: {x_sam_vec.shape}):</b><br>{np.array2string(x_sam_vec, precision=4)}</font>""")

        # Update Tab 2: Visualized Features
        self.histogram_canvas.plot_histogram(x_base_vec, title=f"x_base Distribution for Node {node_id}")
        geo_labels = ["Normalized Area", "Center Y", "Center X", "Aspect Ratio"]
        sam_labels = ["Predicted IoU", "Stability Score"]
        viz_html = "<b>x_base Statistical Summary:</b><br>"
        viz_html += f"- Mean: {x_base_vec.mean():.4f}<br>- Std Dev: {x_base_vec.std():.4f}<br>"
        viz_html += f"- Min: {x_base_vec.min():.4f}<br>- Max: {x_base_vec.max():.4f}<br><hr>"
        viz_html += "<b>Geometric Features (x_geo):</b><br>"
        for label, val in zip(geo_labels, x_geo_vec): viz_html += f"- {label}: {val:.4f}<br>"
        viz_html += "<hr><b>SAM Features (x_sam):</b><br>"
        for label, val in zip(sam_labels, x_sam_vec): viz_html += f"- {label}: {val:.4f}<br>"
        self.labeled_feature_display.setHtml(viz_html)

        # --- UPDATE NEW TAB 3: FEATURE AS IMAGE ---
        self.feature_image_canvas.plot_feature_image(
            x_base_vec, title=f"x_base Feature Image for Node {node_id}"
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = GraphViewerApp()
    main_win.show()
    sys.exit(app.exec_())