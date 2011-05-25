from vision.reporting.track import *
import multiprocessing
import cPickle as pickle
from pprint import pprint

import logging
logging.basicConfig(level = logging.INFO, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(24)
engines = [#LinearInterpolationEngine(),
           ActiveLearnDPEngine(pairwisecost = 0.1, sigma = 1000),
           DynamicProgrammingEngine(pairwisecost = 0.1)]
cpfs = [.01]
frames = filetovideo("/scratch/virat/frames")

data = load(["VIRAT_S_040104_05_000939_001116.txt"], frames, ["Car"], toframe = 2000)

data = build(data, cpfs, engines, pool = pool)

visualizepaths(data, "tmp/")

print "saving to data"
pickle.dump(data, open("data.pkl", "w"))

plotperformance(data, Intersection())
