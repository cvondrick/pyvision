from vision.track.realprior import *
import vision.detectionreader
import vision.drawer
import os.path
from vision import visualize
import random

import vision.track.dp

logging.basicConfig(level = logging.INFO)

path = ("/csail/vision-videolabelme/databases/"
        "video_adapt/home_ac_a/frames/0/bundler")
path = "/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/bundler"

video = vision.flatframeiterator(path + "/..", start = 1)

root = os.path.join(path, "pmvs")
patches, projections = pmvs.read(root, start = 1)

seed = vision.Box(173, 30, 173 + 51, 30 + 137, 0 + 1)
seed2 = vision.Box(256, 49, 256 + 58, 49 + 117, 200 + 1)
seed3 = vision.Box(173, 88, 173 + 67, 88 + 89, 600 + 1)
seed4 = vision.Box(156, 57, 156 + 71, 57 + 115, 900 + 1)
seed5 = vision.Box(181, 42, 181 + 51, 42 + 106, 406)
seed6 = vision.Box(199, 54, 199 + 60, 54 + 142, 211)
badseed = vision.Box(358, 12, 358 + 33, 12 + 25, 150 + 1)
seeds = [seed, seed2,  seed5, seed6]

detections = vision.detectionreader.exemplarsvm('/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/pedro-pascal-bottle.mat')

prior = ThreeD(video, patches, projections)
prior.build(detections)
prior.estimate()

#import pylab, numpy, Image
#for frame, nd in prior.scoreall():
#    print frame
#
#    pylab.subplot(211)
#    pylab.set_cmap("gray")
#    pylab.title("min={0}, max={1}".format(nd.min(), nd.max()))
#    pylab.imshow(nd.transpose())
#
#    pylab.subplot(212)
#    pylab.imshow(numpy.asarray(video[frame]))
#
#    pylab.savefig("tmp/out{0}.png".format(frame))
#    pylab.clf()
