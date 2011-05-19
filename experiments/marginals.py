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

g = frameiterator("/scratch/virat/frames/VIRAT_S_050102_02_000526_000593")
b = [Box(342, 235, 342 + 67, 235 + 55, 429)]
stop = 675

pool = multiprocessing.Pool(24)
svm = model.PathModel(g, b)
frame, score, path = marginals.pick(b[0], stop, svm, g, pool = pool,
                                    pairwisecost = .01,
                                    erroroverlap = 0.5)

visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
