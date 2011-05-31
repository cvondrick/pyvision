from vision.reporting.track import *
import multiprocessing
import cPickle as pickle
from pprint import pprint

file = "VIRAT_S_040305_02_001463_001550"

import logging
logging.basicConfig(level = logging.INFO, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(24)
engines = [#LinearInterpolationEngine(),
           DynamicProgrammingEngine(pairwisecost = 0.1, hogbin = 4),
           ActiveLearnDPEngine(pairwisecost = 0.1, sigma = 10, hogbin = 4)]
cpfs = [.002, 0.0025, 0.003, 0.0035, 0.004]
frames = filetovideo("/scratch/virat/frames")

data = load(["{0}.txt".format(file)], frames, ["Car"])

data = build(data, cpfs, engines, pool = pool)

#visualizepaths(data, "tmp/")

print "saving to data"
pickle.dump(data, open("{0}.pkl".format(file), "w"))

plotperformance(data, PercentOverlap(0.5))
