"""
Reads in the annotations and images for the PASCAL data format.

Example usage:

>>> p = PascalDataset(".")
>>> for anotation in p.annotations():
...     im = p.image(annotation.image)
...     do_something(im, annotation)
"""

from vision import Box
import os
from PIL import Image
import logging
from xml.etree import ElementTree

logger = logging.getLogger("vision.pascal")

class PascalDataset(object):
    def __init__(self, root):
        self.root = root

    def annotations(self, imageset = None, classes = None, nodifficult = False):
        if imageset is None:
            imageset = self.imageset()
        elif isinstance(imageset, str):
            imageset = self.imageset(imageset)
        for file in imageset:
            logger.info("Reading {0}".format(file))
            file = os.path.join(self.root, "Annotations", "{0}.xml".format(file))

            if not file.endswith(".xml"):
                continue
            
            tree = ElementTree.parse(file)
            filename = tree.find("filename").text.strip()
                
            for object in tree.findall("object"):
                label = object.find("name").text.strip()
                if classes and label not in classes:
                    continue

                if nodifficult:
                    difficult = object.find("difficult").text
                    difficult = bool(int(difficult))
                    if difficult:
                        continue

                xtl = int(object.find("bndbox/xmin").text)
                ytl = int(object.find("bndbox/ymin").text)
                xbr = int(object.find("bndbox/xmax").text)
                ybr = int(object.find("bndbox/ymax").text)

                yield Box(xtl, ytl, xbr, ybr, label = label, image = filename)

    def imageset(self, imageset = "trainval"):
        """
        imageset should be: trainval, train, or val
        """
        imageset = "{0}.txt".format(imageset)
        path = os.path.join(self.root, "ImageSets", "Main", imageset)
        for line in open(path):
            yield line.strip()

    def find(self, has = [], missing = [], imageset = "trainval"):
        answer = set()
        for multi, type in [(1, has), (-1, missing)]:
            for like in type:
                path = "{0}_{1}.txt".format(like, imageset)
                path = os.path.join(self.root, "ImageSets", "Main", path)
                for line in open(path):
                    image, indicator = line.split()
                    if int(indicator) * multi > 0 and image not in answer:
                        yield image
                        answer.add(image)

    def image(self, image):
        path = os.path.join(self.root, "JPEGImages", image)
        return Image.open(path)

    def __getitem__(self, image):
        return self.image(image)

    def __iter__(self):
        return self.annotations()
