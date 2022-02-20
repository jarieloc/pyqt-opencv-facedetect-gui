import sys 
from PyQt6 import QtCore, QtGui, QtWidgets
from ui_mainwindow import Ui_Form
import cv2
import numpy as np
import os.path

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.comboBox.addItem("No Filter")
        self.ui.comboBox.addItem("Canny Edge Filter")
        self.ui.comboBox.addItem("Blur Filter")
        self.ui.comboBox.addItem("Sobel Edge Filter")
        self.ui.comboBox.addItem("Face Detect")
        self.ui.comboBox.addItem("Face Detect + Blur")
        self.ui.comboBox.currentIndexChanged.connect(self.selectionchange)

        self.img = cv2.imread('01.jpg')
        self.faceImg = self.img
        self.imgProcessed = cv2.imread('processed.jpg')
        self.dispState = False
        self.tempImg = '01.jpg'
        self.imgPath = '.'
        self.setImage(self.img)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.ui.inputFileButton.clicked.connect(self.onInputFileButtonClicked)
        self.ui.updateImage.clicked.connect(self.updateImageClicked)

    def faceDetect(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        facesDetected = self.face_cascade.detectMultiScale(grayscale, 1.1, 4)
        for (x, y, w, h) in facesDetected:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv2.imwrite("face_detected.png", img)
        self.ui.faceDetectLabel.setText("Faces Detected: " + str(len(facesDetected)))
        return img

    def updateImageClicked(self):
        self.img = cv2.imread(self.tempImg)
        self.dispState = False
        self.setImage(self.img)
        self.ui.comboBox.setCurrentIndex(0)
        self.ui.faceDetectLabel.setText("Faces Detected: 0")

    def onInputFileButtonClicked(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', self.imgPath, 'All Files (*.*)')
        if filename:
            self.ui.inputFileLineEdit.setText(filename)
            self.tempImg = filename
            self.imgPath = os.path.dirname(filename)

    def setImage(self, img):
        if self.dispState:
            img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        else:
            img = QtGui.QImage(self.tempImg)
        img_pix = QtGui.QPixmap.fromImage(img)
        if img_pix.height() < img_pix.width():
            self.ui.img_frame.setPixmap(img_pix.scaledToHeight(self.ui.img_frame.width(), QtCore.Qt.TransformationMode.SmoothTransformation))
        elif img_pix.height() > img_pix.width():
            self.ui.img_frame.setPixmap(img_pix.scaledToWidth(self.ui.img_frame.width(), QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            self.ui.img_frame.setPixmap(img_pix.scaled(self.ui.img_frame.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))

    def imageBlurCompute(self, img):
        kernel = np.ones((10, 10), np.float32)/25
        output = cv2.filter2D(img, -1, kernel)
        return output

    def sobelEdgeCompute(self, img, x, y):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        output = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=x, dy=y, ksize=5) # X-axis
        cv2.imwrite("processed.jpg", output)

    def cannyEdgeCompute(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        output = cv2.Canny(img_gray, 100, 200) # X-axis
        cv2.imwrite("processed.jpg", output)

    def selectionchange(self, i):
        if i == 0:
            print("no filter")
            self.dispState = False
            self.setImage(self.img)
        elif i == 1:
            print("Canny Edge filter")
            self.dispState = True
            self.cannyEdgeCompute(self.img)
            cannyEdge = cv2.imread('processed.jpg')
            self.setImage(cannyEdge)
        elif i == 2:
            print("Blur filter")
            self.dispState = True
            blurredImg = cv2.blur(self.img, (5, 5))
            self.setImage(blurredImg)

        elif i == 3:
            print("Sobel filter")
            self.dispState = True
            self.sobelEdgeCompute(self.img, 0, 1)
            sobelEdge = cv2.imread('processed.jpg')
            self.setImage(sobelEdge)

        elif i == 4:
            print("Face Detect")
            self.dispState = True
            faceDetected = self.faceDetect(self.img)
            self.setImage(faceDetected)
        else:
            self.dispState = True
            blurredImg = cv2.blur(self.img, (50, 50))
            # blurredImg = cv2.GaussianBlur(self.img, (3, 3), 0)
            print("Face Detect + Blur")
            self.dispState = True
            faceDetected = self.faceDetect(blurredImg)
            self.setImage(faceDetected)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
