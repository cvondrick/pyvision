from PIL import Image
import os

class frameiterator(object):
    """
    A simple iterator to produce frames.
    """
    def __init__(self, base, start = 0, skip = 1):
        self.base = base
        self.start = start
        self.skip = skip
        self.length = None

    def path(self, frame):
        l1 = frame / 10000
        l2 = frame / 100
        path = "{0}/{1}/{2}.jpg".format(l1, l2, frame)
        path = "{0}/{1}".format(self.base, path)
        return path

    def __len__(self):
        if not self.length:
            i = self.start
            counter = 0
            while True:
                if not os.path.exists(self.path(i)):
                    self.length = counter - 1
                    break
                counter += 1
                i += self.skip
        return self.length

    def __getitem__(self, frame):
        if frame < 0:
            raise RuntimeError("Frame {0} is before start of video".format(frame))
        return Image.open(self.path((frame * self.skip) + self.start))

    def __iter__(self):
        i = 0
        while True:
            try:
                yield self[i]
            except IOError:
                raise
            else:
                i += self.skip

class flatframeiterator(frameiterator):
    def path(self, frame):
        return "{0}/{1:05d}.jpg".format(self.base, frame)
