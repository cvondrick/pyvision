from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
import ImageColor
import pickle

logging.basicConfig(level = logging.INFO)

#g = frameiterator("/scratch/vatic/uci-basketball")
#b = [Box(39, 215, 39 + 38, 215 + 95, 15744),
#     #Box(303, 223, 303 + 38, 223 + 95, 15793)]
#     Box(488, 247, 488 + 38, 247 + 95, 15895)]

#g = frameiterator("/scratch/vatic/uci-basketball")
#b = [Box(283, 259, 283 + 57, 259 + 99, 42785),
#     Box(489, 243, 489 + 47, 243 + 114, 42897)]
#stop = 42907

#g = frameiterator("/scratch/virat/frames/VIRAT_S_000003")
#b = [Box(346, 170, 17 + 346, 170 + 42, 275)]
#stop = 450

#g = frameiterator("/scratch/vatic/syn-yi-levels")
#b = [Box(153, 124, 153 + 61, 124 + 148, 0)]
#b2 = [Box(153, 124, 153 + 61, 124 + 148, 0),
#     Box(550, 82, 550 + 61, 82 + 148, 756)]
#stop = 1349

#g = frameiterator("/scratch/vatic/syn-bounce-level")
#b = [Box(365, 70, 365 + 63, 70 + 297, 278)]
#b = [Box(365, 70, 365 + 63, 70 + 297, 278),
#     Box(624, 56, 624 + 66, 56 + 318, 1655)]
#
#stop = 1768

g = frameiterator("/scratch/vatic/syn-occlusion2")
b = [Box(592, 48, 592 + 103, 48 + 326, 20)]
stop = 22

g = frameiterator("/scratch/virat/frames/VIRAT_S_000302_04_000453_000484")
b = [Box(156, 96, 156 + 50, 96 + 24, 270),
     Box(391, 83, 391 + 48, 83 + 22, 459)]
stop = 500

pool = multiprocessing.Pool(24)

frame, score, path, marginals = marginals.pick(b, g,
                                               last = stop,
                                               pool = pool,
                                               pairwisecost = .01,
                                               dim = (40, 40),
                                               sigma = 1,
                                               erroroverlap = 0.5,
                                               hogbin = 8,
                                               clickradius = 10,
                                               c = 1)

pickle.dump(marginals, open("occlusion.pkl", "w"))

visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
