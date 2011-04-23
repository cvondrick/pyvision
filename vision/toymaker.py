import Image
from ImageDraw import Draw
import random
from vision.annotations import *

class Geppetto(object):
    """
    Facilitates making toy tracking data. Geppetto manages the toys and renders the
    frames to produce the actual data.
    """
    def __init__(self, size = (720, 480), background = (255, 255, 255), cap = -1):
        self.size = size
        self.background = background
        self.toys = []
        self.frames = 0
        self.cap = cap

    def add(self, toy):
        """
        Adds a toy that will be rendered.
        """
        self.toys.append(toy)
        self.frames = max(self.frames, toy.frames)

    def render(self, frame):
        """
        Renders a frame and returns an PIL instance.
        """
        if frame >= self.frames:
            raise ValueError("Requested frame {0}, but there are only {1}".format(frame, self.frames))
        canvas = Image.new("RGB", self.size, self.background)
        for toy in self.toys:
            toy.render(frame, canvas)
        return canvas

    def write(self, frame, location):
        """
        Writes a rendered frame to disk.
        """
        self.render(frame).save(location)

    def export(self, location, format = "jpg"):
        """
        Exports all rendered frames to disk.
        """
        self.frames = max(x.frames for x in self.toys)
        if self.cap > 0:
            self.frames = min(self.cap, self.frames)
        for frame in range(self.frames):
            self.write(frame, "{location}/{frame}.{format}".format(location=location, frame=frame, format=format))

    def __getitem__(self, frame):
        """
        Alias to render a frame.
        """
        return self.render(frame)

    def __len__(self):
        return self.frames

class Toy(object):
    """
    An abstract toy class.
    """
    def __init__(self, position = (0,0), size = (100,100), color = "black"):
        self.size = size
        self.color = color
        self.positions = [position]
        self.lastposition = position
        self.frames = 1

    def linear(self, position, frame, chaos = 0):
        """
        Moves the object to a new location using linear interpolation.
        If chaos is nonzero, then the object will jiggle on its way there.
        """
        if frame <= self.frames:
            raise ValueError("Target frame is behind current time index")
        fdiff = float(frame - self.frames)
        rx = (position[0] - self.lastposition[0]) / fdiff
        ry = (position[1] - self.lastposition[1]) / fdiff
        for i in range(self.frames + 1, frame):
            x = self.lastposition[0] + rx * (i - self.frames)
            x += random.randint(-chaos, chaos)
            y = self.lastposition[1] + ry * (i - self.frames)
            y += random.randint(-chaos, chaos)
            y = max(y, 0)
            x = max(x, 0)
            self.positions.append((int(x),int(y)))
        self.positions.append(position)
        self.frames = frame
        self.lastposition = position
        return self

    def stationary(self, frame):
        """
        Causes the object to remain still until the specified frame.
        """
        for i in range(frame - self.frames):
            self.positions.append(self.lastposition)
        self.frames = frame
        return self

    def disappear(self, frame, reappear = True):
        """
        Causes the object to disappear until specified frame.
        """
        if frame < self.frames:
            raise ValueError("Target frame is behind current time index")
        amount = frame - self.frames
        if reappear:
            amount -= 1
        self.positions.extend([None] * amount)
        if reappear:
            self.positions.append(self.lastposition)
        self.frames = frame
        return self

    def random(self, frame, estate = (720, 480)):
        """
        Causes the object to randomly teleport around the screen.
        """
        for _ in range(frame - self.frames):
            self.positions.append((random.randint(0, estate[0] - self.size[0]), random.randint(0, estate[1] - self.size[1])))
        self.frames = frame
        return self

    def set(self, position):
        """
        Moves the object to a new location by one frame only.
        """
        self.positions.append(position)
        self.lastposition = position
        self.frames += 1
        return self

    def render(self, frame, canvas):
        """
        Renders the specified frame to the canvas.
        """
        if frame < self.frames and self.positions[frame]:
            self.draw(frame, canvas)
        
    def draw(self, canvas):
        raise NotImplementedError()

    def __getitem__(self, frame):
        """
        Gets the bounding for this toy at a certain frame.
        """
        if frame < 0:
            frame = len(self) + frame
        pos = self.positions[frame]
        if not pos:
            return Box(0, 0, 1, 1, frame, 1)
        return Box(pos[0], pos[1],
                   pos[0] + self.size[0],
                   pos[1] + self.size[1], frame, 0)
    
    def __len__(self):
        return len(self.positions)

    def groundtruth(self):
        return list(self)

class Rectangle(Toy):
    """
    Produces a rectangle as the toy.
    """
    def draw(self, frame, canvas):
        p = self.positions[frame]
        Draw(canvas).rectangle((p[0], p[1], p[0] + self.size[0], p[1] + self.size[1]), fill = self.color)

class Ellipse(Toy):
    """
    Produces an ellipsis as the toy.
    """
    def draw(self, frame, canvas):
        p = self.positions[frame]
        Draw(canvas).ellipse((p[0], p[1], p[0] + self.size[0], p[1] + self.size[1]), fill = self.color)

class Bitmap(Toy):
    """
    Draws a bitmap instead of any vector graph.
    """
    def __init__(self, image, positions = (0,0)):
        Toy.__init__(positions, size = image.size)
        self.image = image

    def draw(self, frame, canvas):
        p = self.positions[frame]
        canvas.paste(image, p)

def seed(s = 0):
    """
    Allows changing the random seed so that the same path is generated repeatedly.
    """
    random.seed(s)
