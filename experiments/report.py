from vision.reporting.track import *
import multiprocessing
import cPickle as pickle

import logging
logging.basicConfig(level = logging.WARNING, format = "%(process)d: %(message)s")

pool = multiprocessing.Pool(24)
engines = [LinearInterpolationEngine(),
#           ActiveLearnLinearEngine(),
           DynamicProgrammingEngine()]
cpfs = [.01, 0.012, 0.015, 0.02]
frames = filetovideo("/scratch/virat/frames")

data = load(["VIRAT_S_040104_05_000939_001116.txt"], frames, ["Car"])
data = build(data, cpfs, engines, pool = pool)
plotperformance(data, PercentOverlap(0.5))
