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
import Image
import ImageTk

def getcoords(image):
    d = Drawer(image)
    d.run()
    return d.coordinates

class Drawer(object):
    def __init__(self, image):
        self.lastclick = None
        self.rectangle = None
        self.coordinates = None

        self.root = Tk()
        self.image = image
        self.tkimage = ImageTk.PhotoImage(image)

        width, height = image.size

        self.w = Canvas(self.root, width = width, height = height)
        self.w.create_image((width / 2, height / 2), image = self.tkimage)
        self.w.image = self.tkimage
        self.w.bind("<Button-1>", self.clickcanvas)
        self.w.bind("<ButtonRelease-1>", self.clickcanvas)
        self.w.bind("<Motion>", self.movecanvas)
        self.w.pack()

    def movecanvas(self, event):
        if self.lastclick:
            xmin, ymin, xmax, ymax = self.calccoords(event, self.lastclick)
            self.w.delete(self.rectangle)
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

    def calccoords(self, a, b):
        xmin = min(a.x, b.x)
        ymin = min(a.y, b.y)
        xmax = max(a.x, b.x)
        ymax = max(a.y, b.y)

        return xmin, ymin, xmax, ymax

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    im = Image.open("/csail/vision-videolabelme/databases/video_adapt/kitchen_carl_c/frames/0/00001.jpg")
    print getcoords(im)
