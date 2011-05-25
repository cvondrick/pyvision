from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
import ImageColor

logging.basicConfig(level = logging.INFO)

#g = frameiterator("/scratch/vatic/uci-basketball")
#b = [Box(39, 215, 39 + 38, 215 + 95, 15744),
#     #Box(303, 223, 303 + 38, 223 + 95, 15793)]
#     Box(488, 247, 488 + 38, 247 + 95, 15895)]

#g = frameiterator("/scratch/vatic/uci-basketball")
#b = [Box(226, 230, 226 + 51, 230 + 97, 20290),
#     Box(46, 211, 46 + 51, 211 + 97, 20355)]

#g = frameiterator("/scratch/virat/frames/VIRAT_S_000300_01_000055_000218")
#b = [Box(183, 91, 183 + 54, 91 + 30, 625),
#     Box(504, 71, 504 + 54, 71 + 30, 720)]

g = frameiterator("/scratch/vatic/syn-yi-levels")
b = [Box(153, 124, 153 + 61, 124 + 148, 0)]
stop = 1349

g = frameiterator("/scratch/vatic/syn-many")
b = [Box(461, 131, 461 + 67, 131 + 101, 0)]
stop = 900

pool = multiprocessing.Pool(24)

frame, score, path = marginals.pick(b, g,
                                    last = stop,
                                    pool = pool,
                                    pairwisecost = .001,
                                    dim = (40, 40),
                                    sigma = 1000,
                                    erroroverlap = 0.5,
                                    rgbbin = 16,
                                    hogbin = 8,
                                    c = 0.1)

visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
