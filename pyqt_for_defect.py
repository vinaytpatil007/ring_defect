import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMainWindow
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from visualize_defect_on_ring import visualize_ring

# Set the environment variable within the Python script
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vinay/ring_defect/defect/lib/python3.11/site-packages/cv2/qt/plugins/platform/'

model_path = "C:/ring_defect/Models/patch_stack_seg_18sept_150epoch.pth" 

class ImageVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Ring Defect Visualizer')
        self.setGeometry(100, 100, 800, 600)  

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setGeometry(350, 10, 100, 30)  

        # Labels to display the text above the images
        self.label_original_text = QLabel("Original Image", self)
        self.label_original_text.setFont(QFont('Arial', 12))
        self.label_original_text.setAlignment(Qt.AlignCenter)
        self.label_original_text.setGeometry(50, 50, 300, 30)  

        self.label_predicted_text = QLabel("Predicted Image", self)
        self.label_predicted_text.setFont(QFont('Arial', 12))
        self.label_predicted_text.setAlignment(Qt.AlignCenter)
        self.label_predicted_text.setGeometry(450, 50, 300, 30) 

        # Labels to display the images
        self.label_original = QLabel(self)
        self.label_original.setGeometry(10, 100, 385, 385)  

        self.label_predicted = QLabel(self)
        self.label_predicted.setGeometry(405, 100, 385, 385) 

    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if image_path:
            image = cv2.imread(image_path)
            bb_image, bb_image1 = visualize_ring(image, model_path)

            self.display_image(self.label_original, bb_image)
            self.display_image(self.label_predicted, bb_image1)

    def display_image(self, label, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageVisualizerApp()
    window.show()
    sys.exit(app.exec_())