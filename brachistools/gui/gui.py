
import warnings
import os, sys

import numpy as np

from PyQt6 import QtCore, QtGui
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QMainWindow,
    QApplication,
    QGridLayout,
    QCheckBox,
    QPushButton,
    QPlainTextEdit,
    QLabel,
    QProgressBar,
    QProgressDialog,
    QFileDialog,
    QMessageBox,
    QListView,
    QWidget,
    QAction)

from brachistools.gui.io import load_folder, imread, abbrev_path

from brachistools.version import brachistools_version
from brachistools.segmentation import segmentation_pipeline, default_segmentation_params, label2rgb_bbox
from brachistools.classification import classification_pipeline

global logger
def run():
    from brachistools.io import logger_setup

    global logger
    logger, _ = logger_setup()
    warnings.filterwarnings("ignore")

    app = QApplication(sys.argv)
    mainw = MainWindow()
    mainw.show()
    retv = app.exec()
    sys.exit(retv)

class Segment1Thread(QtCore.QThread):
    MAX_PROGRESS = 11

    update_progress = QtCore.Signal(int)

    def __init__(self, input_image, params, parent=None):
        super().__init__(parent)

        self.input_image = input_image
        self.params = params

        from brachistools.segmentation import vahadane
        self.vahadane_transform = vahadane(**params['vahadane'])

    def run(self):
        from skimage.util import img_as_ubyte
        from brachistools.segmentation import (
            vahadane, equalize_adapthist,
            inverted_gray_scale, threshold_otsu,
            remove_small_objects, remove_small_holes,
            distance_transform_edt, peak_local_max,
            peaks_to_markers, watershed,
            merge_small_labels
        )

        input_image = img_as_ubyte(self.input_image)
        image_H = self.vahadane_transform(input_image)
        self.update_progress.emit(1)

        image_H = equalize_adapthist(image_H, **self.params['equalize_adapthist'])
        self.update_progress.emit(2)

        image_H = img_as_ubyte(input_image)
        image_H = inverted_gray_scale(image_H)
        self.update_progress.emit(3)

        nuclei = image_H > threshold_otsu(image_H)
        self.update_progress.emit(4)

        nuclei = remove_small_objects(nuclei, **self.params['remove_small_objects'])
        self.update_progress.emit(5)
        nuclei = remove_small_holes(nuclei, **self.params['remove_small_holes'])
        self.update_progress.emit(6)

        distances = distance_transform_edt(nuclei)
        self.update_progress.emit(7)
        local_maxima_idx = peak_local_max(distances, **self.params['peak_local_max'])
        self.update_progress.emit(8)
        markers = peaks_to_markers(local_maxima_idx, shape=nuclei.shape)
        self.update_progress.emit(9)
        labeled_nuclei = watershed(-distances, markers, mask=nuclei)
        self.update_progress.emit(10)
        labeled_nuclei = merge_small_labels(labeled_nuclei, **self.params['merge_small_labels'])
        self.update_progress.emit(11)

        self.nuclei = nuclei
        self.labeled_nuclei = labeled_nuclei

class Classify1Thread(QtCore.QThread):
    MAX_PROGRESS = 11

    update_progress = QtCore.Signal(int)

    def __init__(self, input_image, parent=None):
        super().__init__(parent)

        self.input_image = input_image


    def run(self):
        import cv2
        import numpy as np
        from brachistools.classification import (
            load_classification_model
        )

        img = cv2.resize(self.input_image, (224, 224))
        img = np.array(img)
        predict_class, confidence_score = load_classification_model(img)
        self.predict_class = predict_class
        self.confidence_score = confidence_score


class SegmentationWindow(QMainWindow):
    def __init__(self, parent, img_fn) -> None:
        super().__init__(parent)

        self.setGeometry(50, 50, 1500, 500)
        self.setWindowTitle("Segmentation results of " + img_fn)

        self.cwidget = QWidget(self)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(6)

        self.OrigImgLabel = QLabel()
        self.InstanceSegLabel = QLabel()
        self.BinaryMaskLabel = QLabel()

        self._init_ui_components()

        self._orig_img = None
        self._binary_mask = None
        self._instance_seg = None

    def _init_ui_components(self):
        self.OrigImgLabel.setFixedSize(500, 500)
        self.l0.addWidget(self.OrigImgLabel, 0, 0)

        self.InstanceSegLabel.setFixedSize(500, 500)
        self.l0.addWidget(self.InstanceSegLabel, 0, 1)

        self.BinaryMaskLabel.setFixedSize(500, 500)
        self.l0.addWidget(self.BinaryMaskLabel, 0, 2)

    def _set_img_for_label(self, lab: QLabel, imarr):
        from skimage.util import img_as_ubyte

        imarr = img_as_ubyte(imarr)
        if len(imarr.shape) < 3:
            # Single-channel
            qimg = QtGui.QImage(imarr.data,
                imarr.shape[1], imarr.shape[0], imarr.shape[1],
                QtGui.QImage.Format.Format_Grayscale8)
        else:
            qimg = QtGui.QImage(imarr.data,
                imarr.shape[1], imarr.shape[0], imarr.shape[1]*3,
                QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap(qimg)
        lab.setPixmap(pixmap)
        lab.setScaledContents(True)
        lab.update()

    def set_orig_img(self, imarr):
        self._orig_img = imarr
        self._set_img_for_label(self.OrigImgLabel, imarr)

    def set_instance_seg(self, imarr):
        self._instance_seg = imarr
        self._set_img_for_label(self.InstanceSegLabel, label2rgb_bbox(imarr, self._orig_img))

    def set_binary_mask(self, imarr):
        self._binary_mask = imarr
        self._set_img_for_label(self.BinaryMaskLabel, imarr)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(50, 50, 1200, 700)
        self.setWindowTitle(f"brachistools v{brachistools_version}")

        # Set central widget for layout control
        self.cwidget = QWidget(self)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(6)

        # Add UI components
        self.SelectInputButton = QPushButton('Select input')
        self.BatchCheckBox = QCheckBox('Select all')
        self.SegmentationButton = QPushButton('Segmentation')
        self.DiagnosisButton = QPushButton('Suggest diagnosis')
        self.SaveButton = QPushButton('Save')
        self.InputFolderLabel = QLabel()
        self.PrevImgButton = QPushButton('Previous')
        self.ImgFileLabel = QLabel()
        self.NextImgButton = QPushButton('Next')
        self.InputListView = QListView()
        self.ImgDisplayLabel = QLabel()
        self.CellCountLabel = QLabel('Cell counts:')
        self.CellCountTextEdit = QPlainTextEdit()
        self.DiagnosisLabel = QLabel('Suggested diagnosis:')
        self.DiagnosisTextEdit = QPlainTextEdit()

        # Initialize UI components
        self._init_ui_components()

        # Set param slots
        self._segmentation_params = default_segmentation_params

        # Set data slots
        self._input_filenames = []
        self._input_folder_path = None
        self._curr_index = None
        self._curr_img = None

    def help_window(self):
        ...

    def segmentation_window(self, nuclei, labeled_nuclei):
        segwindow = SegmentationWindow(
            parent=self, img_fn=self._input_filenames[self._curr_index])
        segwindow.set_orig_img(self._curr_img)
        segwindow.set_binary_mask(nuclei)
        segwindow.set_instance_seg(labeled_nuclei)
        segwindow.show()

    def segment_current(self):
        if self._curr_img is None:
            QMessageBox.critical(self, "Invalid operation", "Please select an image")
            return

        segment_thread = Segment1Thread(
            input_image=self._curr_img,
            params=self._segmentation_params,
            parent=self)

        segment_progress = QProgressDialog(self)
        segment_progress.setWindowTitle("Segmentation in progress...")
        segment_progress.setLabelText("Please wait")
        segment_progress.setRange(0, Segment1Thread.MAX_PROGRESS)
        segment_progress.setModal(True)
        segment_progress.canceled.connect(segment_thread.quit)
        segment_thread.finished.connect(segment_progress.accept)
        segment_thread.update_progress.connect(segment_progress.setValue)
        segment_thread.start()
        segment_progress.exec()

        nuclei, labeled_nuclei = segment_thread.nuclei, segment_thread.labeled_nuclei
        self.CellCountTextEdit.setPlainText(str(len(set(labeled_nuclei.flat))))
        self.segmentation_window(nuclei=nuclei, labeled_nuclei=labeled_nuclei)

    def classify_current(self):
        if self._curr_img is None:
            QMessageBox.critical(self, "Invalid operation", "Please select an image")
            return

        classify_thread = Classify1Thread(
            input_image=self._curr_img,
            parent=self
        )

        classify_progress = QProgressDialog(self)
        classify_progress.setWindowTitle("Classification in progress...")
        classify_progress.setLabelText("Please wait")
        classify_progress.setRange(0, Classify1Thread.MAX_PROGRESS)
        classify_progress.setModal(True)
        classify_progress.canceled.connect(classify_thread.quit)
        classify_thread.finished.connect(classify_progress.accept)
        classify_thread.update_progress.connect(classify_progress.setValue)
        classify_thread.start()
        classify_progress.exec()

        predict_class, confidence_score = classify_thread.predict_class, classify_thread.confidence_score
        self.DiagnosisTextEdit.setPlainText(f"Predict : {predict_class}\nConfidence : {confidence_score}")

    def segment_all(self):
        QMessageBox.critical(self, "Not supported", "Sorry, this option is under development")

    def classify_all(self):
        QMessageBox.critical(self, "Not supported", "Sorry, this option is under development")

    def do_segment(self):
        if self.BatchCheckBox.isChecked():
            self.segment_all()
        else:
            self.segment_current()

    def do_classify(self):
        if self.BatchCheckBox.isChecked():
            self.classify_all()
        else:
            self.classify_current()

    def select_image(self, index):
        if not self._input_filenames:
            QMessageBox.critical(self,
                                "Invalid operation",
                                "Input file list is empty")
            return

        self.CellCountTextEdit.clear()
        self.DiagnosisTextEdit.clear()
        selected_img_fn = self._input_filenames[index]
        try:
            self._curr_img = imread(os.path.join(self._input_folder_path, selected_img_fn))
        except:
            QMessageBox.critical(self, "Invalid operation", "Failed to load image")
            return

        self._curr_index = index
        self.ImgFileLabel.setText(selected_img_fn)
        self.load_image(selected_img_fn)

    def prev_image(self):
        if self._curr_index is not None and self._input_filenames:
            self.select_image((self._curr_index - 1) % len(self._input_filenames))

    def next_image(self):
        if self._curr_index is not None and self._input_filenames:
            self.select_image((self._curr_index + 1) % len(self._input_filenames))

    def load_image(self, im_fn):
        pixmap = QtGui.QPixmap(os.path.join(self._input_folder_path, im_fn))
        self.ImgDisplayLabel.setPixmap(pixmap)
        self.ImgDisplayLabel.setScaledContents(True)
        self.ImgDisplayLabel.update()

    def clear_panel(self):
        self.CellCountTextEdit.clear()
        self.DiagnosisTextEdit.clear()
        self._curr_img = None
        self._curr_index = None
        self.ImgFileLabel.clear()
        self.ImgDisplayLabel.clear()

    def select_input(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select input folder")
        if not folder_path:
            return

        self.clear_panel()

        input_filenames = load_folder(folder_path, ['.PNG', '.JPG', '.JPEG'])
        if not input_filenames:
            QMessageBox.critical(self, "Invalid operation", "Input folder does not contain any PNG/JPG files")
        else:
            self._input_folder_path = folder_path
            self.InputFolderLabel.setText(abbrev_path(folder_path))
            self._input_filenames = input_filenames
            self.InputListViewModel.setStringList(
                abbrev_path(fn) for fn in self._input_filenames)

    def keyPressEvent(self, event):
        modifiers = (event.modifiers() & (
                QtCore.Qt.ControlModifier |
                QtCore.Qt.ShiftModifier |
                QtCore.Qt.AltModifier
            ))
        if not modifiers:
            if event.key() == QtCore.Qt.Key_Left:
                self.prev_image()
            if event.key() == QtCore.Qt.Key_Right:
                self.next_image()

    def _init_ui_components(self):
        self._init_slots()
        self._init_layout()

    def _init_layout(self):
        self.l0.addWidget(self.SelectInputButton, 0, 0, 1, 3)
        woff = 3
        self.l0.addWidget(self.BatchCheckBox, 0, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.SegmentationButton, 0, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.DiagnosisButton, 0, woff, 1, 2)
        woff += 2

        self.l0.addWidget(self.InputFolderLabel, 1, 0, 1, 3)
        woff = 3
        self.l0.addWidget(self.PrevImgButton, 1, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.ImgFileLabel, 1, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.NextImgButton, 1, woff, 1, 2)

        self.InputListViewModel = QtCore.QStringListModel()
        # self.InputListViewModel.resetInternalData()
        self.InputListView.setModel(self.InputListViewModel)
        self.l0.addWidget(self.InputListView, 2, 0, 7, 3)
        woff = 3
        self.ImgDisplayLabel.setFixedSize(600, 600)
        self.l0.addWidget(self.ImgDisplayLabel, 2, woff, 6, 6,
                          QtCore.Qt.AlignmentFlag.AlignCenter)
        self.l0.addWidget(self.CellCountLabel, 8, woff, 1, 1,
                          QtCore.Qt.AlignmentFlag.AlignRight)
        woff += 1
        self.l0.addWidget(self.CellCountTextEdit, 8, woff, 1, 2)
        woff += 2
        self.l0.addWidget(self.DiagnosisLabel, 8, woff, 1, 1,
                          QtCore.Qt.AlignmentFlag.AlignRight)
        woff += 1
        self.l0.addWidget(self.DiagnosisTextEdit, 8, woff, 1, 2)

    def _init_slots(self):
        self.SelectInputButton.clicked.connect(self.select_input)

        self.SegmentationButton.clicked.connect(self.do_segment)
        self.DiagnosisButton.clicked.connect(self.do_classify)

        self.InputListView.clicked.connect(lambda index: self.select_image(index.row()))
        self.PrevImgButton.clicked.connect(self.prev_image)
        self.NextImgButton.clicked.connect(self.next_image)
