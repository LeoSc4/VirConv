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
    def __init__(self, map_scale=0.05):
        super().__init__()
        self.map_scale = map_scale
        self.roi_mode = False
        self.roi_top_left = None 
        self.roi_bottom_right = None
        self.roi_rect_item = None

        self.roi_preview_rect_item = None  
        self.is_setting_top_left = True     # control for top-left and bottom-right points of roi rect

        self.reference_mode = False         #for "Set Reference Point"
        self.reference_point = None 

        self.initUI()

    def initUI(self):
        self.setWindowTitle('ADTC - Sensor Graph Trajectory Definition')
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

        self.roi_button = QPushButton('Set RoI for ADTC', self)
        self.roi_button.clicked.connect(self.toggleRoiMode)
        self.layout.addWidget(self.roi_button)

        self.reference_button = QPushButton('Set Reference Point', self)
        self.reference_button.clicked.connect(self.toggleReferenceMode)
        self.layout.addWidget(self.reference_button)

        self.draw_button = QPushButton('Set Graph for Camera Trajectory', self)
        self.draw_button.clicked.connect(self.toggleDrawMode)
        self.layout.addWidget(self.draw_button)

        self.clear_button = QPushButton('Clear All Curves', self)
        self.clear_button.clicked.connect(self.clearAllCurves)
        self.layout.addWidget(self.clear_button)

        self.save_button = QPushButton('Save Graph, RoI and Reference Point', self)
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

        self.view.setMouseTracking(True)  # <-- Neu: Mausbewegung aktivieren
        self.setMouseTracking(True)       # <-- Falls du auch im ganzen Fenster Mausbewegung tracken willst


        # Create an off-screen buffer for drawing
        self.offscreen_pixmap = None
        self.view.mouseMoveEvent = self.updateCursorPosition

        self.view.mousePressEvent = self.handleMousePressInView



    def updateCursorPosition(self, event):
        # Get the cursor position in view coordinates
        cursor_pos_view = event.pos()
        # Map the cursor position to image coordinates
        cursor_pos_image = self.view.mapToScene(cursor_pos_view)
        # Display the cursor position in pixels
        self.cursor_position_label.setText(f'Cursor Position: ({cursor_pos_image.x():.2f}, {cursor_pos_image.y():.2f})')
        
    def mouseMoveEvent(self, event):
        cursor_pos_view = event.pos()
        cursor_pos_image = self.view.mapToScene(cursor_pos_view)
        self.cursor_position_label.setText(
            f'Cursor Position: ({int(cursor_pos_image.x())}, {int(cursor_pos_image.y())})'
        )

        # --- Draw dynamic ROI Preview ---
        if self.roi_mode and self.roi_top_left:
            x1, y1 = self.roi_top_left
            x2, y2 = int(cursor_pos_image.x()), int(cursor_pos_image.y())

            rect = QtGui.QPolygonF([
                QPointF(x1, y1), QPointF(x2, y1),
                QPointF(x2, y2), QPointF(x1, y2),
                QPointF(x1, y1)
            ])

            if self.roi_preview_rect_item:
                self.scene.removeItem(self.roi_preview_rect_item)

            roi_pen = QPen(Qt.green)
            roi_pen.setStyle(Qt.DashLine)  # show preview in dashed line
            roi_pen.setWidth(2)
            self.roi_preview_rect_item = self.scene.addPolygon(rect, roi_pen)
        
        if self.reference_mode: 
            if event.button() == Qt.LeftButton:
                view_pos = self.view.mapFromGlobal(event.globalPos())
                scene_pos = self.view.mapToScene(view_pos)
                x, y = int(scene_pos.x()), int(scene_pos.y())

                self.reference_point = {"x_pixel": x, "y_pixel": y, "z": 0}
                self.reference_mode = False
                self.label.setText(f"Reference Point set at ({x}, {y}, 0)")

                # Draw a small black circle at reference point
                radius = 5
                self.scene.addEllipse(x - radius, y - radius, 2 * radius, 2 * radius,
                                    QPen(Qt.black), QBrush(Qt.black))
                return

    def handleMousePressInView(self, event):
        if self.image_path:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())

            if self.reference_mode:
                self.reference_point = {"x_pixel": x, "y_pixel": y, "z": 0}
                self.reference_mode = False
                self.label.setText(f"Reference Point set at ({x}, {y}, 0)")

                # Draw a small black circle at the reference point
                radius = 6
                ellipse_item = self.scene.addEllipse(x - radius, y - radius, 2 * radius, 2 * radius,
                                                    QPen(Qt.magenta), QBrush(Qt.magenta))
                # Draw "Ref" label next to it
                text_item = self.scene.addText("Reference Point")
                text_item.setDefaultTextColor(Qt.magenta)
                text_item.setPos(x + 8, y - 10)

                self.scene.addItem(ellipse_item)
                return

            elif self.roi_mode:
                if self.is_setting_top_left:
                    self.roi_top_left = (x, y)
                    self.is_setting_top_left = False
                    self.label.setText('ROI Mode: Click Bottom Right Corner')
                    print(f"Top-left of ROI set at ({x}, {y})")
                else:
                    self.roi_bottom_right = (x, y)
                    print(f"Bottom-right of ROI set at ({x}, {y})")

                    if self.roi_rect_item:
                        self.scene.removeItem(self.roi_rect_item)
                    if self.roi_preview_rect_item:
                        self.scene.removeItem(self.roi_preview_rect_item)

                    x1, y1 = self.roi_top_left
                    x2, y2 = self.roi_bottom_right

                    rect = QtGui.QPolygonF([
                        QPointF(x1, y1), QPointF(x2, y1),
                        QPointF(x2, y2), QPointF(x1, y2),
                        QPointF(x1, y1)
                    ])

                    roi_pen = QPen(Qt.green)
                    roi_pen.setWidth(2)
                    self.roi_rect_item = self.scene.addPolygon(rect, roi_pen)

                    self.roi_mode = False
                    self.is_setting_top_left = True
                    self.label.setText('ROI Mode: OFF')
                    self.roi_preview_rect_item = None

            elif self.draw_mode:
                if not self.current_curve:
                    # First click -> set start pose
                    self.start_pose = (x, y, 0)  # default theta = 0
                    self.current_curve.append(self.start_pose)

                    # Visualize small green point for start
                    r = 4
                    self.scene.addEllipse(x - r, y - r, 2 * r, 2 * r, QPen(Qt.green), QBrush(Qt.green))
                    return

                # Compute orientation based on previous point
                x1, y1, theta_1 = self.current_curve[-1]
                dy = y1 - y
                dx = x - x1
                theta_2 = 180 - math.degrees(math.atan2(dx, dy))

                # Add the point to the current curve
                self.current_curve.append((x, y, theta_2))

                # Visualize clicked node as blue circle
                r = 4
                self.scene.addEllipse(x - r, y - r, 2 * r, 2 * r, QPen(Qt.blue), QBrush(Qt.blue))

                # Draw the path
                self.drawPoints(None)

                # Save the curve to the list
                self.drawn_curves.append(self.current_curve.copy())



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
            self.label.setText('Set Graph for Camera Trajectory:: ON')
        else:
            self.label.setText('Set Graph for Camera Trajectory:: OFF')

    def clearAllCurves(self):
        self.drawn_curves, self.current_curve = [], []
        self.offscreen_pixmap.fill(Qt.transparent)

        # Keep reference before clearing
        if self.roi_rect_item:
            self.scene.removeItem(self.roi_rect_item)
            self.roi_rect_item = None  # prevent segmentation fault

        self.scene.clear()
        self.scene.addPixmap(self.pixmap)

        # Draw RoI new if existent
        self.restoreRoIifExists()

        self.label.setText('Set Graph for Camera Trajectory: OFF')

    def drawPoints(self, event):
        self.restoreReferencePointIfExists()

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

            self.restoreRoIifExists()
            self.restoreReferencePointIfExists()
         
    def mousePressEvent(self, event):
        if self.image_path:
            if self.roi_mode:
                if event.button() == Qt.LeftButton:
                    view_pos = self.view.mapFromGlobal(event.globalPos())
                    scene_pos = self.view.mapToScene(view_pos)
                    x, y = scene_pos.x(), scene_pos.y()

                    if self.is_setting_top_left:
                        # Setze Top Left
                        self.roi_top_left = (int(x), int(y))
                        self.is_setting_top_left = False
                        self.label.setText('ROI Mode: Click Bottom Right Corner')
                        print(f"Top-left of ROI set at ({x}, {y})")
                    else:
                        # Setze Bottom Right
                        self.roi_bottom_right = (int(x), int(y))
                        print(f"Bottom-right of ROI set at ({x}, {y})")

                        # Vollständiges Rechteck zeichnen
                        if self.roi_rect_item:
                            self.scene.removeItem(self.roi_rect_item)
                        if self.roi_preview_rect_item:
                            self.scene.removeItem(self.roi_preview_rect_item)

                        x1, y1 = self.roi_top_left
                        x2, y2 = self.roi_bottom_right

                        rect = QtGui.QPolygonF([
                            QPointF(x1, y1), QPointF(x2, y1),
                            QPointF(x2, y2), QPointF(x1, y2),
                            QPointF(x1, y1)
                        ])

                        roi_pen = QPen(Qt.green)
                        roi_pen.setWidth(2)
                        self.roi_rect_item = self.scene.addPolygon(rect, roi_pen)

                        self.roi_mode = False
                        self.is_setting_top_left = True
                        self.label.setText('ROI Mode: OFF')

                        self.roi_preview_rect_item = None

            elif self.draw_mode:
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

    def toggleRoiMode(self):
        self.roi_mode = not self.roi_mode
        self.is_setting_top_left = True  # Immer neu anfangen
        if self.roi_mode:
            self.label.setText('ROI Mode: Click Top Left Corner')
        else:
            self.label.setText('ROI Mode: OFF')

    def toggleReferenceMode(self): 
        self.reference_mode = not self.reference_mode
        if self.reference_mode:
            self.label.setText('Reference Mode: Click on reference infrastructure point that equals the coordinate frame in simulation stage.')
        else: 
            self.label.setText('Reference Mode: OFF')

    def restoreRoIifExists(self):
        if self.roi_top_left and self.roi_bottom_right:
            x1, y1 = self.roi_top_left
            x2, y2 = self.roi_bottom_right

            rect = QtGui.QPolygonF([
                QPointF(x1, y1), QPointF(x2, y1),
                QPointF(x2, y2), QPointF(x1, y2),
                QPointF(x1, y1)
            ])

            roi_pen = QPen(Qt.green)
            roi_pen.setWidth(2)

            self.roi_rect_item = self.scene.addPolygon(rect, roi_pen)
    
    def restoreReferencePointIfExists(self):
        if self.reference_point:
            x = self.reference_point["x_pixel"]
            y = self.reference_point["y_pixel"]

            # Redraw magenta reference point
            radius = 6
            self.scene.addEllipse(
                x - radius, y - radius, 2 * radius, 2 * radius,
                QPen(Qt.magenta), QBrush(Qt.magenta)
            )

            # Redraw label
            text_item = self.scene.addText("Reference Point")
            text_item.setDefaultTextColor(Qt.magenta)
            text_item.setPos(x + 8, y - 10)

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
        base_dir = os.path.join(os.getcwd(), "graph_output")
        os.makedirs(base_dir, exist_ok=True)

        output_dir = os.path.join(base_dir, "base_graph_from_human_input")
        os.makedirs(output_dir, exist_ok=True)

        # Suggest default file name
        default_filename = os.path.join(output_dir, "graph_and_roi")
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

        if self.roi_top_left and self.roi_bottom_right:
            x1, y1 = self.roi_top_left
            x2, y2 = self.roi_bottom_right

            top_left = [min(x1, x2), min(y1, y2)]
            bottom_right = [max(x1, x2), max(y1, y2)]
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            area = width * height

            roi_data = {
                "map_relative_path": "map_coverage_calculator/occupancy_grid.png",
                "top_left": top_left,
                "bottom_right": bottom_right,
                "area": area
            }
        else:
            QMessageBox.warning(self, "Error", "No ROI defined!")
            return

        # Add reference point to graph
        if not self.reference_point:
            QMessageBox.warning(self, "Error", "No reference point defined!")
            return

        # JSON structure
        graph_data = {
            "output_dir": output_dir,
            "node_count": node_count,
            "graph_length_meters": round(graph_length, 2),
            "nodes": nodes,
            "roi": roi_data,
            "reference_point": self.reference_point  
        }

        # Save .json file
        try:
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            QMessageBox.information(self, "Success", f"Graph saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Saving failed:\n{str(e)}")

        # Save a PNG of the current scene
        image_path = file_path.replace(".json", ".png")
        image = QtGui.QImage(self.scene.sceneRect().size().toSize(), QtGui.QImage.Format_ARGB32)
        image.fill(Qt.transparent)

        painter = QtGui.QPainter(image)
        self.scene.render(painter)
        painter.end()

        try:
            image.save(image_path)
            print(f"Saved scene image to: {image_path}")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Image saving failed: {e}")

def get_graph_file_path():
    app = QApplication(sys.argv)
    editor = GraphEditor(map_scale=0.05)
    editor.show()
    app.exec_()

    # Load latest saved file from graph_output     
    output_dir = os.path.join(os.getcwd(), "graph_output")
    latest_file = max(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".json")],
        key=os.path.getmtime
    )
    return latest_file

def main():
    
    app = QApplication(sys.argv)
    ex = GraphEditor(map_scale=0.05)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()