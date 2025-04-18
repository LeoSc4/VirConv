import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QMessageBox


from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPainterPath, QBrush
from PyQt5.QtCore import Qt, QPointF
import json
import math

def get_pix_coords(width, height, x, y, theta):

    # Calculate the coordinates of the four corners of the rotated rectangle
    angle_rad = math.radians(theta)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    x1 = x - (width / 2) * cos_theta - (height / 2) * sin_theta
    y1 = y - (width / 2) * sin_theta + (height / 2) * cos_theta

    x2 = x + (width / 2) * cos_theta - (height / 2) * sin_theta
    y2 = y + (width / 2) * sin_theta + (height / 2) * cos_theta

    x3 = x + (width / 2) * cos_theta + (height / 2) * sin_theta
    y3 = y + (width / 2) * sin_theta - (height / 2) * cos_theta

    x4 = x - (width / 2) * cos_theta + (height / 2) * sin_theta
    y4 = y - (width / 2) * sin_theta - (height / 2) * cos_theta

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]


class GraphEditor(QWidget):
    def __init__(self, map_scale=0.1):
        super().__init__()
        self.map_scale = map_scale
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ADTC - Sensor graph definition')
        self.setGeometry(1000, 100, 1000, 800)

        self.image_path = None
        self.draw_mode = False
        self.drawn_curves = []
        self.current_curve = []

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.loadImage)
        self.layout.addWidget(self.load_button)

        self.draw_button = QPushButton('Draw Mode', self)
        self.draw_button.clicked.connect(self.toggleDrawMode)
        self.layout.addWidget(self.draw_button)

        self.clear_button = QPushButton('Clear All Curves', self)
        self.clear_button.clicked.connect(self.clearAllCurves)
        self.layout.addWidget(self.clear_button)

        self.save_button = QPushButton('Save Curves', self)
        self.save_button.clicked.connect(self.saveCurves)  # Connect to saveCurves method
        self.layout.addWidget(self.save_button)


        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.start_pose = None

        self.cursor_position_label = QLabel('Cursor Position: (0, 0)', self)

        cursor_pos_layout = QHBoxLayout()
        cursor_pos_layout.addWidget(self.cursor_position_label)
        self.layout.addLayout(cursor_pos_layout)

        self.setLayout(self.layout)

        # Create an off-screen buffer for drawing
        self.offscreen_pixmap = None
        self.view.mouseMoveEvent = self.updateCursorPosition

    def updateCursorPosition(self, event):
        # Get the cursor position in view coordinates
        cursor_pos_view = event.pos()
        # Map the cursor position to image coordinates
        cursor_pos_image = self.view.mapToScene(cursor_pos_view)
        # Display the cursor position in pixels
        self.cursor_position_label.setText(f'Cursor Position: ({cursor_pos_image.x():.2f}, {cursor_pos_image.y():.2f})')

    def loadImage(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp *.gif);;All Files (*)", options=options)

        if file_path:
            self.image_path = file_path
            self.pixmap = QPixmap(file_path)
            self.scene.clear()
            self.scene.setSceneRect(0, 0, self.pixmap.width(), self.pixmap.height())
            self.scene.addPixmap(self.pixmap)
            self.offscreen_pixmap = QPixmap(self.pixmap.size())
            self.offscreen_pixmap.fill(Qt.transparent)

    def toggleDrawMode(self):
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.label.setText('Draw Mode: ON')
        else:
            self.label.setText('Draw Mode: OFF')

    def clearAllCurves(self):
        self.drawn_curves, self.current_curve = [], []
        self.offscreen_pixmap.fill(Qt.transparent)
        self.scene.clear()
        self.scene.addPixmap(self.pixmap)
        self.label.setText('Draw Mode: OFF')

    def drawPoints(self, event):
        if self.draw_mode and self.image_path:
            painter = QPainter(self.offscreen_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw path line in red
            painter.setPen(QPen(QColor(255, 30, 30), 2))
            path = QPainterPath()
            path.moveTo(QPointF(self.current_curve[0][0], self.current_curve[0][1]))
            for point in self.current_curve[1:]:
                path.lineTo(QPointF(point[0], point[1]))
            painter.drawPath(path)

            # Draw each node as blue circle and add index number in white
            radius = 5
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            for idx, (x, y, _) in enumerate(self.current_curve):
                # Draw blue node
                painter.setBrush(QBrush(Qt.blue))
                painter.setPen(QPen(Qt.blue))
                painter.drawEllipse(QPointF(x, y), radius, radius)

                # Draw index number (white text)
                painter.setPen(Qt.blue)
                painter.drawText(QPointF(x + 6, y - 6), str(idx))

            painter.end()

            # Update scene
            self.scene.clear()
            self.scene.addPixmap(self.pixmap)
            self.scene.addPixmap(self.offscreen_pixmap)


         
    def mousePressEvent(self, event):
        if self.draw_mode and self.image_path:
            if event.button() == Qt.LeftButton:
                view_pos = self.view.mapFromGlobal(event.globalPos())
                scene_pos = self.view.mapToScene(view_pos)
                x2, y2 = scene_pos.x(), scene_pos.y()

                if not self.current_curve:
                    # First click -> set start pose
                    self.start_pose = (x2, y2, 0)  # default theta = 0
                    self.current_curve.append(self.start_pose)

                    # Visualize small green point for start
                    r = 4
                    self.scene.addEllipse(x2 - r, y2 - r, 2*r, 2*r, QPen(Qt.green), QBrush(Qt.green))
                    return

                # Compute orientation based on previous point
                x1, y1, theta_1 = self.current_curve[-1]
                dy = y1 - y2
                dx = x2 - x1
                theta_2 = 180 - math.degrees(math.atan2(dx, dy))

                # Add the point to the current curve
                self.current_curve.append((x2, y2, theta_2))

                # Visualize clicked node as red circle
                r = 4
                self.scene.addEllipse(x2 - r, y2 - r, 2*r, 2*r, QPen(Qt.blue), QBrush(Qt.blue))

                # Draw the path
                self.drawPoints(event)

                # Save the curve to the list
                self.drawn_curves.append(self.current_curve.copy())

            

    def createPoly(self, rec_x, rec_y, rec_theta, rec_width, rec_height):
        polygon = QtGui.QPolygonF() 

        points = get_pix_coords(rec_width, rec_height, rec_x, rec_y, rec_theta)
        for point in points:
            x, y = point
            polygon.append(QPointF(x, y))  

        return polygon
    

    def saveCurves(self):
        print("Start saving graph...")

        if not self.current_curve:
            QMessageBox.warning(self, "Error", "No graph defined!")
            return

        # Create graph_output directory if it doesn't exist
        # output_dir = os.path.join(os.getcwd(), "graph_output")
        output_dir = os.path.join('/workspace/graph_definition_UI', "graph_output")
        os.makedirs(output_dir, exist_ok=True)

        # Suggest default file name
        default_filename = os.path.join(output_dir, "graph")
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Graph JSON", default_filename, "JSON Files (*.json)"
        )

        if not file_path:
            return

        if not file_path.endswith(".json"):
            file_path += ".json"

        # Count of nodes
        node_count = len(self.current_curve)

        # Calculate graph length (in meters)
        graph_length = 0.0
        for i in range(1, node_count):
            x1, y1, _ = self.current_curve[i - 1]
            x2, y2, _ = self.current_curve[i]
            dist_px = math.hypot(x2 - x1, y2 - y1)
            graph_length += dist_px * self.map_scale

        # Build list of nodes
        nodes = [{"x_pixel": int(x), "y_pixel": int(y)} for x, y, _ in self.current_curve]

        # JSON structure
        graph_data = {
            "output_dir": output_dir,
            "node_count": node_count,
            "graph_length_meters": round(graph_length, 2),
            "nodes": nodes
        }

        # Save file
        try:
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=4)
            QMessageBox.information(self, "Success", f"Graph saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Saving failed:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    ex = GraphEditor(map_scale=0.05)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()