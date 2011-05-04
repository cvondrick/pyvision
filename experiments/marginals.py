from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging

logging.basicConfig(level = logging.INFO)

name = "VIRAT_S_040104_05_000939_001116"
root = os.path.dirname(os.path.abspath(__file__))
data = readpaths(open("{0}/{1}.txt".format(root, name)))
iter = frameiterator("/scratch/virat/frames/{0}".format(name))

for id, (label, _) in enumerate(data):
    print id, label

id = 23
label, path = data[id]

print label

start = min(x.frame for x in path if not x.lost)
stop  = max(x.frame for x in path if not x.lost)

print start, stop

#162 and stop = 1670

#start = 300
#stop = 1500

pathdict = dict((x.frame, x) for x in path)

given = [pathdict[start], pathdict[start+(stop-start)/2], pathdict[stop]]

pool = multiprocessing.Pool(24)
svm = model.PathModel(iter, given)
frame, score, path = marginals.pick(given[0], given[-1], svm, iter, pool = pool,
                                    pairwisecost = .0001)

visualize.save(visualize.highlight_paths(path, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
