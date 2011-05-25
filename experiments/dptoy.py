from vision import *
from vision import model
from vision.track import dp
from vision import visualize
from vision.toymaker import *
import os
import logging

logging.basicConfig(level = logging.INFO)

root = os.path.dirname(os.path.abspath(__file__))

g = Geppetto()
b = Rectangle()
b = b.linear((300,300), 10)
b = b.linear((0,300), 20)
b = b.linear((300,0), 30)
g.add(b)

path = b.groundtruth()
pathdict = dict((x.frame, x) for x in path)

start = 0
stop = len(g) - 1
given = [pathdict[start], pathdict[stop]]

svm = model.PathModel(g, given)

predicted = dp.track(given[0], given[-1], svm, g, pairwisecost = 0.000001)

vit = visualize.highlight_path(g, predicted)
base = "tmp"

try:
    os.makedirs(base)
except:
    pass
visualize.save(vit, lambda x: "{0}/{1}.jpg".format(base, x))
