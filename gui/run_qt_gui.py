import sys
from PyQt4 import QtGui, QtCore
import model_interface as mi


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

        # label to show classification result
        self.label = QtGui.QLabel(self.widget)
        self.label.setText("Please select an image.")
        self.layout.addWidget(self.label)

        self.widget.setWindowTitle("Defect Classification")
        self.widget.setLayout(self.layout)
        self.widget.show()

        # load model interface
        self.model = mi.ModelInterface()

    def openButtonClicked(self):

        file = QtGui.QFileDialog.getOpenFileName(self.widget, caption='Please select an image')
        print 'Open: ', file

        if file:
            pixmap = QtGui.QPixmap(file).scaled(400, 400)
            self.image_label.setPixmap(pixmap)

        # call tensorflow API to get classification result
        res_label, res_prob = self.model.predict(str(file))

        # update label text
        res_str = ''
        for i in range(len(res_label)):
            res_str += str(res_label[i])+': '+str(res_prob[i])+'\n'
        print res_str

        self.label.setText(res_str)


def main():
    app = QtGui.QApplication(sys.argv)

    dfcw = DefectClassWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


# python tensorflow/examples/label_image/label_image.py \
# --graph=/home/zby/workspaces/grad_thesis_ws/trained_model/inception_v3/output_graph.pb \
# --labels=/home/zby/workspaces/grad_thesis_ws/trained_model/inception_v3/output_labels.txt \
# --input_layer=Mul \
# --output_layer=final_result \
# --input_mean=128 --input_std=128 \
# --image=/home/zby/workspaces/grad_thesis_ws/generated_images/image_0311/image_0311/abrasion/abrasion_1_2.png
