from vision.track.realcoords import *
import vision.drawer
import os.path
from vision import visualize
import random

logging.basicConfig(level = logging.DEBUG)

path = ("/csail/vision-videolabelme/databases/"
        "video_adapt/home_ac_a/frames/0/bundler")
path = "/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/bundler"

video = vision.flatframeiterator(path, 1, 5)

root = os.path.join(path, "pmvs")
patches, projections = pmvs.read(root)

seed = vision.Box(173, 30, 173 + 51, 30 + 137, 0)
predicted = track(video, [seed], patches, projections)

vit = visualize.highlight_path(video, predicted)
visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
