from vision import *
from vision.track import dp
from vision import visualize
import os
import logging
import multiprocessing

logging.basicConfig(level = logging.INFO)

name = "VIRAT_S_040104_05_000939_001116"
root = os.path.dirname(os.path.abspath(__file__))
#iter = frameiterator("/scratch/virat/frames/{0}".format(name))
iter = frameiterator("/scratch/vatic/syn-bounce-level")

#start = Box(234, 115, 234 + 72, 115 + 44, 0)
#start = Box(434, 184, 434 + 112, 184 + 75, 0)
start = Box(576, 63, 576 + 81, 63 + 297, 0)
stop  = 1000

given = [start]

pool = multiprocessing.Pool(24)
predicted = dp.fill(given, iter, last = stop, pool = pool, pairwisecost = .1)

vit = visualize.highlight_path(iter, predicted)
visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
