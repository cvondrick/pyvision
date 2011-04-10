from vision import *
from vision.track import interpolation
from vision.alearn import linear as alearn
from vision import visualize
import os
import multiprocessing

name = "VIRAT_S_040104_05_000939_001116"
root = os.path.dirname(os.path.abspath(__file__))
data = readpaths(open("{0}/{1}.txt".format(root, name)))
pool = multiprocessing.Pool(24)
iter = frameiterator("/scratch/virat/frames/{0}".format(name))

#for id, (label, path) in enumerate(data):
#    print "Visualizing {0} {1}".format(label, id)
#    vit = visualize.highlight_path(iter, interpolation.LinearFill(path))
#    base = "{0}/visualize/{1}".format(root, id)
#    os.makedirs(base)
#    visualize.save(vit, lambda x: "{0}/{1}.jpg".format(base, x))
#raise SystemExit()

label, path = data[9]
id = 9

print "Processing {0} {1}".format(label, id)

start = min(x.frame for x in path if not x.lost)
stop  = max(x.frame for x in path if not x.lost)

#162 and stop = 1670

start = 200
stop = 1500

print "start = {0} and stop = {1}".format(start, stop)

pathdict = dict((x.frame, x) for x in path)

given = [pathdict[start], pathdict[stop]]

for _ in range(20):
    print "Given frames are:", ", ".join(str(x.frame) for x in given)
    print "Simulating with {0} clicks".format(len(given))
    base = "{0}/visualize/{1}/clicks{2}/tmp".format(root, id, len(given))
    try:
        os.makedirs(base)
    except:
        pass
    askingfor = alearn.pick(iter, given, pool = pool, skip = 1,
                            bgskip = 3, bgsize = 5e5, errortube = 100000,
                            plot = base)
    print "Requested frame {0}".format(askingfor)
    print "Visualizing path with {0} clicks".format(len(given))
    
    vit = visualize.highlight_path(iter, interpolation.LinearFill(given))
    base = "{0}/visualize/{1}/clicks{2}/wants{3}".format(root, id,
                                                            len(given),
                                                            askingfor)
    try:
        os.makedirs(base)
    except:
        pass

    visualize.save(vit, lambda x: "{0}/{1}.jpg".format(base, x))

    given.append(pathdict[askingfor])
    given.sort(key = lambda x: x.frame)
