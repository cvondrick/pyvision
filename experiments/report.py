from vision.reporting.track import *
import multiprocessing
import cPickle as pickle
from pprint import pprint

file = "uci-basketball"

import logging
logging.basicConfig(level = logging.INFO, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(12)
engines = [LinearInterpolationEngine(),
           DynamicProgrammingEngine(pairwisecost = 0.1, hogbin = 4),
           ]#ActiveLearnDPEngine(pairwisecost = 0.1, sigma = 10, hogbin = 4)]
cpfs = [.01, 0.02, 0.05, 0.1]
frames = filetovideo("/scratch/vatic")

data = load(["{0}.txt".format(file)], frames, toframe = 10100)

data = build(data, cpfs, engines, pool = pool)

#visualizepaths(data, "tmp/")

print "saving to data"
pickle.dump(data, open("{0}.pkl".format(file), "w"))

plotperformance(data, PercentOverlap(0.5))
