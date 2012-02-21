from vision.track.realcoords import *
import vision.drawer
import os.path
from vision import visualize
import random
import multiprocessing

pool = multiprocessing.Pool(multiprocessing.cpu_count())

import vision.track.dp

logging.basicConfig(level = logging.INFO)

path = ("/csail/vision-videolabelme/databases/"
        "video_adapt/home_ac_a/frames/0/bundler")
path = "/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/bundler"

video = vision.flatframeiterator(path + "/..")

root = os.path.join(path, "pmvs")
patches, projections = pmvs.read(root)

seed = vision.Box(173, 30, 173 + 51, 30 + 137, 0 + 1)
seed2 = vision.Box(256, 49, 256 + 58, 49 + 117, 200 + 1)
seed3 = vision.Box(173, 88, 173 + 67, 88 + 89, 600 + 1)
seed4 = vision.Box(156, 57, 156 + 71, 57 + 115, 900 + 1)
seed5 = vision.Box(181, 42, 181 + 51, 42 + 106, 406)
seed6 = vision.Box(199, 54, 199 + 60, 54 + 142, 211)
badseed = vision.Box(358, 12, 358 + 33, 12 + 25, 150 + 1)
seeds = [seed, seed2,  seed5, seed6]
predicted = track(video, seeds, patches, projections)

predicted = vision.track.dp.fill(predicted, video, last = len(video), pool = pool, hogbin = 4, c = 0.001)

justdp = vision.track.dp.fill(seeds, video, last = len(video), pool = pool, hogbin = 4, c = .1)

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
vit = visualize.highlight_paths(video, [predicted, justdp])
visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
