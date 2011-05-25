from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
import ImageColor

logging.basicConfig(level = logging.INFO)

g = Geppetto()

b = Rectangle((100, 100))
b.linear((300, 100), 10)
b.linear((300, 300), 20)
b.linear((100, 300), 30)
b.linear((100, 100), 40)
g.add(b)

o = Rectangle((100, 100))
o.linear((100, 300), 10)
o.linear((300, 300), 20)
o.linear((300, 100), 30)
o.linear((100, 100), 40)
g.add(o)

pool = multiprocessing.Pool(24)
frame, score, path = marginals.pick([b[0]], g, pool = pool,
                                    last = b[-1].frame,
                                    pairwisecost = .0001,
                                    erroroverlap = 0.5)

visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
