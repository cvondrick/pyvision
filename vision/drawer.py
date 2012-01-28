from Tkinter import *
import Image
import ImageTk

def draw(image):
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
        self.w.bind("<Motion>", self.movecanvas)
        self.w.pack()

    def movecanvas(self, event):
        if self.lastclick:
            xmin, ymin, xmax, ymax = self.calccoords(event, self.lastclick)
            self.w.delete(self.rectangle)
            self.rectangle = self.w.create_rectangle((xmin, ymin, xmax, ymax), width=5, outline = "red")

    def clickcanvas(self, event):
        if self.lastclick:
            self.coordinates = self.calccoords(event, self.lastclick)
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
    print draw(im)
