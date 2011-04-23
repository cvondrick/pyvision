from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging

logging.basicConfig(level = logging.INFO)

g = Geppetto()
b = Rectangle((0, 0))
b = b.linear((300, 0), 10)
b = b.linear((120, 150), 20)
b = b.linear((300, 300), 30)
b = b.linear((0, 350), 40)
b = b.linear((0, 0), 50)
g.add(b)

pool = multiprocessing.Pool(24)
svm = model.PathModel(g, [b[0], b[-1]])
frame, score, path = marginals.pick(b[0], b[-1], svm, g, pool = pool,
                                    pairwisecost = .01)

visualize.save(visualize.highlight_path(g, path), lambda x: "tmp/path{0}.jpg".format(x))

print frame
