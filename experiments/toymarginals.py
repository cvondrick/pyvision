from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
from PIL import ImageColor
import pylab
import pickle

logging.basicConfig(level = logging.INFO)

g = Geppetto((720,480))

b = Rectangle((400, 100), color="white")
b.linear((400, 800), 20)
g.add(b)

#g = Geppetto()
#b = Rectangle((100, 100))
#b.linear((600, 100), 100)
#g.add(b)
#
#o = Rectangle((100, 350))
#o.linear((600, 350), 100)
#g.add(o)

pool = multiprocessing.Pool(24)
frame, score, path, m = marginals.pick([b[0], b[-1]], g, pool = pool,
                                    pairwisecost = .001,
                                    sigma = .1,
                                    erroroverlap = 0.5)


#visualize.save(visualize.highlight_paths(g, [path, b], width = 3, colors = ["red", "green"]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
