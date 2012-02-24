from vision import Box
from scipy.io import loadmat
import logging

logger = logging.getLogger("vision.detectionreader")

def exemplarsvm(filename):
    logger.info("Reading detections from {0}".format(filename))
    data = loadmat(filename)
    data = data['ds']
    for frame, detections in enumerate(data):
        detections = detections[0][0,0][0]
        for i in range(detections.shape[0]):
            d = detections[i,:]
            xtl = max(0, d[0])
            ytl = max(0, d[1])
            xbr = d[2]
            ybr = d[3]
            score = d[-1]
            yield Box(xtl, ytl, xbr, ybr, frame, score = score)

#boxes = exemplarsvm('/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/pedro-pascal-bottle.mat')
#print boxes.next()
