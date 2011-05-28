from vision.reporting.track import *
import multiprocessing
import cPickle as pickle
from pprint import pprint

import logging
logging.basicConfig(level = logging.INFO, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(24)
engines = [#LinearInterpolationEngine(),
           DynamicProgrammingEngine(pairwisecost = 0.1, hogbin = 4),
           ActiveLearnDPEngine(pairwisecost = 0.1, sigma = 10, hogbin = 4)]
cpfs = [.002, 0.0025, 0.003, 0.0035, 0.004]
frames = filetovideo("/scratch/virat/frames")

data = load(["VIRAT_S_040104_05_000939_001116.txt"], frames, ["Car"])

data = build(data, cpfs, engines, pool = pool)

#visualizepaths(data, "tmp/")

print "saving to data"
pickle.dump(data, open("data.pkl", "w"))

plotperformance(data, PercentOverlap(0.5))
