from vision import *
from vision.track import dp
from vision import visualize
import os
import logging

logging.basicConfig(level = logging.INFO)

name = "VIRAT_S_040104_05_000939_001116"
root = os.path.dirname(os.path.abspath(__file__))
data = readpaths(open("{0}/{1}.txt".format(root, name)))
iter = frameiterator("/scratch/virat/frames/{0}".format(name))

for id, (label, _) in enumerate(data):
    print id, label

id = 30
label, path = data[id]

print label

start = min(x.frame for x in path if not x.lost)
stop  = max(x.frame for x in path if not x.lost)

#162 and stop = 1670

#start = 300
#stop = 1500

pathdict = dict((x.frame, x) for x in path)

given = [pathdict[start], pathdict[start+(stop-start)/2], pathdict[stop]]

predicted = dp.fill(given, iter, pairwiseradius = 10,
                    resizestart = 1.0, resizestop = 1.1, resizeincrement = 0.2,
                    c = 0.1)

vit = visualize.highlight_path(iter, predicted)
base = "{0}/dynamic/{1}/clicks{2}/".format(root, id, len(given))
try:
    os.makedirs(base)
except:
    pass

visualize.save(vit, lambda x: "{0}/{1}.jpg".format(base, x))
