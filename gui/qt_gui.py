import sys
sys.path.append("..")

from PyQt4 import QtGui, QtCore
import interface.model_interface as mi


class DefectClassWindow:

    def __init__(self):
        # main widget
        self.widget = QtGui.QWidget()
        self.widget.setGeometry(100, 100, 400, 500)

        # vertical layout
        self.layout = QtGui.QVBoxLayout()

        # button to open image
        self.open_button = QtGui.QPushButton()
        self.open_button.setText("Open")
        self.open_button.clicked.connect(self.openButtonClicked)
        self.layout.addWidget(self.open_button)

        # label to show image
        self.image_label = QtGui.QLabel()
        self.image_label.resize(400, 400)
        self.image_label.setPixmap(QtGui.QPixmap(400, 400))
        self.layout.addWidget(self.image_label)

        # frame and labels to show classification result
        class_num = 5

        # create frame and grid layout to contains all labels
        frame = QtGui.QFrame()
        grid = QtGui.QGridLayout()

        # create labels
        self.class_labels = []
        self.prob_labels = []
        for i in range(class_num):
            lb1 = QtGui.QLabel()
            lb1.setAlignment(QtCore.Qt.AlignCenter)
            lb1.setFrameShadow(QtGui.QFrame.Sunken)
            lb1.setFrameStyle(QtGui.QFrame.StyledPanel)

            lb2 = QtGui.QLabel()
            lb2.setAlignment(QtCore.Qt.AlignCenter)
            lb2.setFrameShadow(QtGui.QFrame.Sunken)
            lb2.setFrameStyle(QtGui.QFrame.StyledPanel)

            grid.addWidget(lb1, i, 0)
            grid.addWidget(lb2, i, 1)

            self.class_labels.append(lb1)
            self.prob_labels.append(lb2)

        frame.setLayout(grid)
        frame.setFrameShape(QtGui.QFrame.Panel)
        frame.setFrameShadow(QtGui.QFrame.Plain)
        self.layout.addWidget(frame)

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
        self.model = mi.ModelInterface("Inception")

    def openButtonClicked(self):

        file = QtGui.QFileDialog.getOpenFileName(self.widget, caption='Please select an image')
        print 'Open: ', file

        if file:
            pixmap = QtGui.QPixmap(file).scaled(400, 400)
            self.image_label.setPixmap(pixmap)

            # call tensorflow API to get classification result
            res_label, res_prob = self.model.predict(str(file))

            # update label text
            for i in range(len(res_label)):
                self.class_labels[i].setText(str(res_label[i]))
                self.prob_labels[i].setText(str(res_prob[i]))

            self.label.setText('Classification finished! Select another image to continue.')

        else:
            self.label.setText('Image not opened, try another image.')


def main():
    app = QtGui.QApplication(sys.argv)

    dfcw = DefectClassWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

