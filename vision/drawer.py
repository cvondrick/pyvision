"""
A toolkit to mimic imrect in MATLAB. Upon calling getcoords(image), a window
will appear prompting the user to draw a bounding box. Once they have drawn
the box, it returns with the coordinates.

Usage is simple:

>> import Image, vision.drawer
>> image = Image.open("/path/to/image.jpg")
>> xmin, ymin, xmax, ymax = getcoords(image)

The only dependencies are Tkinter and PIL (it does not depend on pyvision).
"""

from Tkinter import *
from PIL import Image
from PIL import ImageTk

try:
    import vision
except ImportError:
    pass
else:
    def getbox(image, frame = None):
        print "get box"
        if frame:
            box = vision.Box(*getcoords(image[frame]))
            box.frame = frame
            return box
        else:
            return vision.Box(*getcoords(image))

def getcoords(image):
    return Drawer(image).coordinates

class Drawer(object):
    def __init__(self, image):
        self.lastclick = None
        self.rectangle = None
        self.coordinates = None

        self.root = Tk()
        self.root.wm_title("Draw a bounding box")

        self.image = image
        self.tkimage = ImageTk.PhotoImage(image)

        width, height = image.size

        self.w = Canvas(self.root, width = width, height = height)
        self.w.create_image((width / 2, height / 2), image = self.tkimage)
        self.w.image = self.tkimage
        self.w.bind("<Button-1>", self.clickcanvas)
        self.w.bind("<Button-3>", self.cancelcanvas)
        self.w.bind("<Motion>", self.movecanvas)
        self.w.pack()

        self.root.mainloop()

    def movecanvas(self, event):
        if self.rectangle:
            self.w.delete(self.rectangle)
            self.rectangle = None
        if self.lastclick:
            xmin, ymin, xmax, ymax = self.calccoords(event, self.lastclick)
            self.rectangle = self.w.create_rectangle((xmin, ymin, xmax, ymax), width=5, outline = "red")

    def clickcanvas(self, event):
        if self.lastclick:
            xmin, ymin, xmax, ymax = self.calccoords(event, self.lastclick)

            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, self.image.size[0])
            ymax = min(ymax, self.image.size[1])

            self.coordinates = (xmin, ymin, xmax, ymax)

            self.root.destroy()
        else:
            self.lastclick = event

    def cancelcanvas(self, event):
        self.lastclick = None

    def calccoords(self, a, b):
        xmin = min(a.x, b.x)
        ymin = min(a.y, b.y)
        xmax = max(a.x, b.x)
        ymax = max(a.y, b.y)

        return xmin, ymin, xmax, ymax

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print "usage: python drawer.py /path/to/image.jpg"
    else:
        print getcoords(Image.open(sys.argv[1]))
