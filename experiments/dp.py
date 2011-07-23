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
iter = frameiterator("/scratch/vatic/uci-basketball")

#start = Box(234, 115, 234 + 72, 115 + 44, 0)
#start = Box(434, 184, 434 + 112, 184 + 75, 0)
start = Box(492, 254, 492 + 16, 254 + 18, 20315)
stop = Box(510, 270, 510 + 16, 270 + 18, 20385)
#stop  = 20385

given = [start, stop]

pool = multiprocessing.Pool(24)
predicted = dp.fill(given, iter, pool = pool, pairwisecost = 1.0, c = 100)

vit = visualize.highlight_path(iter, predicted)
visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
