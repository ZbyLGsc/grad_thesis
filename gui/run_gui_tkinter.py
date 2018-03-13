import tkinter as tk
import tkFileDialog as tkfd
import time
from PIL import ImageTk

_image = ''
# Gui to select image and show classification result


class DefectClassWindow:

    def __init__(self):

        self.top = tk.Tk()

        # create button to select image
        self.button = tk.Button(self.top, text='Open', command=self.openButtonCallback)
        self.button.pack()

        # create label to show selected image
        self.image_label = tk.Label(self.top, width=40, height=40)
        self.image_label.pack()

        # create label to show classification result
        self.label = tk.Label(self.top, height=4)
        self.label.pack()

        self.top.mainloop()

    def openButtonCallback(self):

        # Open a new window for selecting image
        self.file_name = tkfd.askopenfilename()
        print 'file:', self.file_name

        # show the selected image on canvas
        _image = tk.PhotoImage(name=self.file_name)
        self.image_label.configure(image=_image)

        # Call tensorflow API to classify image
        # ...

        # show classification result on label
        self.label["text"] = 'Normal'
        


def main():
    dcw = DefectClassWindow()


if __name__ == '__main__':
    main()
