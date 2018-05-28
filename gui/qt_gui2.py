import sys
sys.path.append("..")

import os
import shutil

from PyQt4 import QtGui, QtCore
import interface.model_interface as mi
import interface.ensemble_interface as ei
import tensorflow as tf
# use opencv to cut big image
import cv2


class DefectClassWindow:

    def __init__(self):
        # main widget
        self.widget = QtGui.QWidget()
        self.widget.setGeometry(100, 100, 400, 700)

        # vertical layout
        self.layout = QtGui.QVBoxLayout()

        # button to open image
        self.open_button = QtGui.QPushButton()
        self.open_button.setText("Open")
        self.open_button.clicked.connect(self.openButtonClicked)
        self.layout.addWidget(self.open_button)

        # label to show image
        self.image_label = QtGui.QLabel()
        self.image_label.resize(400, 600)
        self.image_label.setPixmap(QtGui.QPixmap(400, 600))
        self.layout.addWidget(self.image_label)

        # create hint label
        self.label = QtGui.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setFrameShape(QtGui.QFrame.Panel)
        self.label.setFrameShadow(QtGui.QFrame.Plain)
        self.label.setText("Please select an image.")
        self.layout.addWidget(self.label)

        self.widget.setWindowTitle("Defect Classification")
        self.widget.setLayout(self.layout)
        self.widget.show()

        # load model interface
        self.model = ei.EnsembleModel()
        self.graph1 = self.model.model_inception.modelGraph()
        self.graph2 = self.model.model_mobilenet.modelGraph()

    def openButtonClicked(self):

        file = QtGui.QFileDialog.getOpenFileName(self.widget, caption='Please select an image')
        print 'Open: ', file

        if file:
            pixmap = QtGui.QPixmap(file).scaled(400, 600)
            self.image_label.setPixmap(pixmap)

            # cut the large pictures into square blocks, then use tensorflow API to detect defect
            # in each blocks. Blocks with defects will be marked in the large pictures.

            # read image and get its size
            img = cv2.imread(str(file))
            img2 = cv2.imread(str(file))
            size = img.shape  # rows and cols
            # print 'size:'
            # print size[0], size[1], size[2]

            # loop, cut blocks and detect
            irow = 0
            icol = 0
            side_length = 300
            overlap = 100

            move_row = False
            last_block = False
            count = 0

            # create temp directory for blocks
            tmp_dir = os.getcwd()+'/../tmp'
            if os.path.exists(tmp_dir):
                # os.removedirs(tmp_dir)
                # os.rmdir(tmp_dir)
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

            # main loop
            with tf.Session(graph=self.graph1) as sess1:
                with tf.Session(graph=self.graph2) as sess2:

                    while True:
                        # cut
                        block = img[irow:irow+side_length, icol:icol+side_length]
                        count += 1
                        img_name = tmp_dir + '/' + str(count)+'.png'
                        cv2.imwrite(img_name, block)
                        cv2.imshow("block", block)

                        # show the block in gui
                        img3 = cv2.imread(str(file))
                        cv2.rectangle(img3, (icol, irow), (icol+side_length,
                                                           irow+side_length), (0, 255, 0), 10)
                        mark_name = tmp_dir + '/marked.png'
                        cv2.imwrite(mark_name, img3)
                        pixmap = QtGui.QPixmap(mark_name).scaled(400, 600)
                        self.image_label.setPixmap(pixmap)

                        # call tensorflow API
                        defect = self.model.predict(img_name, sess1, sess2)

                        # mark on origin image when defect detected
                        if defect:
                            cv2.rectangle(img2, (icol, irow), (icol+side_length,
                                                               irow+side_length), (0, 0, 255), 10)

                        if last_block:
                            break

                        # move the sliding windows
                        if move_row:
                            next_down = irow + 2*side_length - overlap
                            if (next_down) > size[0]:
                                irow = size[0]-side_length
                            else:
                                irow += (side_length-overlap)
                            icol = 0
                            move_row = False
                        else:
                            next_right = icol + 2*side_length - overlap
                            if (next_right) > size[1]:
                                icol = size[1]-side_length
                                move_row = True
                            else:
                                icol += (side_length-overlap)
                        print('next row and col: ', irow, icol)

                        if irow == size[0]-side_length and icol == size[1]-side_length:
                            last_block = True

                        cv2.waitKey(1)
            # main loop end

            # save the marked image and show it
            mark_name = tmp_dir + '/marked.png'
            cv2.imwrite(mark_name, img2)
            pixmap = QtGui.QPixmap(mark_name).scaled(400, 600)
            self.image_label.setPixmap(pixmap)

            shutil.rmtree(tmp_dir)
            print 'Loop end'

            self.label.setText('Classification finished! Select another image to continue.')

        else:
            self.label.setText('Image not opened, try another image.')


def main():
    app = QtGui.QApplication(sys.argv)

    dfcw = DefectClassWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
