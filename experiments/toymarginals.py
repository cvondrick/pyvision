from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging

logging.basicConfig(level = logging.INFO)

g = Geppetto()
b = Rectangle((100, 100))
b.linear((300, 100), 15)
b.linear((300, 300), 30)
g.add(b)

o = Rectangle((100, 100))
o.linear((100, 300), 15)
o.linear((300, 300), 30)
g.add(o)

f = Rectangle((600, 350))
f.stationary(30)
g.add(f)

c = Rectangle((100, 100))
c.linear((10, 10), 30)
g.add(c)

pool = multiprocessing.Pool(24)
svm = model.PathModel(g, [b[0], b[-1]])
frame, score, path = marginals.pick(b[0], b[-1], svm, g, pool = pool,
                                    pairwisecost = .0001)

visualize.save(visualize.highlight_paths(g, [path, b]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
