from vision.reporting.track import *
import multiprocessing
import cPickle as pickle
from pprint import pprint

file = "virat-sample-allfeat-1pair"

import logging
logging.basicConfig(level = logging.INFO, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(24)
engines = [LinearInterpolationEngine(),
           DynamicProgrammingEngine(pairwisecost = 1, hogbin = 4),
           ]#ActiveLearnDPEngine(pairwisecost = 0.1, sigma = 10, hogbin = 4)]
cpfs = [.001, .002, 0.005, 0.008, 0.01]
frames = filetovideo("/scratch/virat/frames/")

data = load(["VIRAT_S_010101_04_000534_000616.txt",
#             "VIRAT_S_040103_06_000836_000909.txt",
#             "VIRAT_S_040305_02_001463_001550.txt",
#             "VIRAT_S_040104_05_000939_001116.txt"],
             ], frames, ["Car"])

data = build(data, cpfs, engines, pool = pool)

#visualizepaths(data, "tmp/")

print "saving to data"
pickle.dump(data, open("{0}.pkl".format(file), "w"))

plotperformance(data, PercentOverlap(0.3))
