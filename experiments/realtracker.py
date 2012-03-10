from vision.track.realprior import *
import vision.detectionreader
import vision.drawer
import os.path
from vision import visualize
import random
import multiprocessing

pool = multiprocessing.Pool(multiprocessing.cpu_count())

import vision.track.dp

logging.basicConfig(level = logging.DEBUG)

path = "/csail/vision-videolabelme/databases/video_adapt/home_ac_a/frames/5/bundler"

video = vision.flatframeiterator(path + "/..", start = 1)

root = os.path.join(path, "pmvs")
patches, projections = pmvs.read(root, start = 1)

seed = vision.Box(173, 30, 173 + 51, 30 + 137, 0)
seed2 = vision.Box(256, 49, 256 + 58, 49 + 117, 200)
seed3 = vision.Box(173, 88, 173 + 67, 88 + 89, 600)
seed4 = vision.Box(156, 57, 156 + 71, 57 + 115, 900)
seed5 = vision.Box(181, 42, 181 + 51, 42 + 106, 405)
seed6 = vision.Box(199, 54, 199 + 60, 54 + 142, 210)
badseed = vision.Box(358, 12, 358 + 33, 12 + 25, 150)
seeds = [seed, seed2, seed5, seed6]

category = 'sofa'
detfile = '/csail/vision-videolabelme/databases/video_adapt/home_ac_a/frames/5/dets-sun11-dpm-{0}.mat'.format(category)
detections = vision.detectionreader.exemplarsvm(detfile)
detections = list(detections)
detections.sort(key = lambda x: -x.score)

print detections[0]

realprior = ThreeD(video, patches, projections).build(detections)
predicted = vision.track.dp.fill([detections[0]], video, last = detections[0].frame + 1000, pool = pool, hogbin = 4, pairwisecost = 0.01, c = 0.1, realprior = realprior)

#path = ("/csail/vision-videolabelme/databases/"
#        "video_adapt/home_ac_a/frames/5/bundler-5")
#
#video = vision.flatframeiterator(path, 1, 5)
#
#root = os.path.join(path, "pmvs")
#patches, projections = pmvs.read(root)
#
#seed = vision.Box(55, 39, 55 + 270, 39 + 136, 145)
#predicted = track(video, [seed], patches, projections)
#
vit = visualize.highlight_paths(video, [predicted])
visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
