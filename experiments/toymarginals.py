from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
import ImageColor
import pylab
import pickle

logging.basicConfig(level = logging.INFO)

g = Geppetto()

b = Rectangle((100, 100))
b.linear((300, 100), 10)
b.linear((300, 300), 20)
b.linear((100, 300), 30)
b.linear((100, 100), 40)
b.linear((300, 100), 50)
b.linear((300, 300), 60)
b.linear((100, 300), 70)
b.linear((100, 100), 80)
b.linear((300, 100), 90)
b.linear((300, 300), 100)
b.linear((100, 300), 110)
b.linear((100, 100), 120)
g.add(b)

o = Rectangle((100, 100))
o.linear((100, 300), 10)
o.linear((300, 300), 20)
o.linear((300, 100), 30)
o.linear((100, 100), 40)
o.linear((100, 300), 50)
o.linear((300, 300), 60)
o.linear((300, 100), 70)
o.linear((100, 100), 80)
o.linear((100, 300), 90)
o.linear((300, 300), 100)
o.linear((300, 100), 110)
o.linear((100, 100), 120)
g.add(o)

pool = multiprocessing.Pool(24)
frame, score, path, m = marginals.pick([b[0], b[6]], g, pool = pool,
                                    last = b[-1].frame,
                                    pairwisecost = .001,
                                    erroroverlap = 0.5)


visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

pickle.dump(m, open("bounce.pkl", "w"))

print "frame {0} with score {1}".format(frame, score)

pylab.plot(m.keys(), m.values())
pylab.show()
