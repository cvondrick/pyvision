from vision import *
from vision.track import alearn, interpolation
from vision import visualize
from vision.toymaker import *
import os
import multiprocessing

g = Geppetto()
b = Rectangle()
b = b.linear((300,300), 100)
b = b.linear((0,300), 200)
b = b.linear((300,0), 300)
g.add(b)

path = b.groundtruth()
pathdict = dict((x.frame, x) for x in path)

start = 0
stop = 299
given = [pathdict[start], pathdict[stop]]

id = "toy"
pool = multiprocessing.Pool(24)
root = os.path.dirname(os.path.abspath(__file__))

for _ in range(1):
    print "Given frames are:", ", ".join(str(x.frame) for x in given)
    print "Simulating with {0} clicks".format(len(given))
    askingfor = alearn.pick(g, given, pool = pool, skip = 1,
                            bgskip = 10, bgsize = 5e3, plot = "tmp/",
                            errortube = 100000)
    print "Requested frame {0}".format(askingfor)
    print "Visualizing path with {0} clicks".format(len(given))
    
    vit = visualize.highlight_path(g, interpolation.LinearFill(given))
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
