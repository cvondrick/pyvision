from vision.reporting.track import *
import multiprocessing
import cPickle as pickle

import logging
logging.basicConfig(level = logging.INFO, format = "%(message)s")

engines = [LinearInterpolationEngine(), ActiveLearnLinearEngine()]
cpfs = [.002]
frames = filetovideo("/scratch/virat/frames")
#pool = multiprocessing.Pool(24)

data = load(["VIRAT_S_040104_05_000939_001116.txt"], frames, ["Car"])
