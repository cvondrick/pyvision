from vision import Box
from scipy.io import loadmat

def exemplarsvm(filename):
    data = loadmat(filename)
    data = data['ds']

    boxes = []

    for frame, detections in enumerate(data):
        detections = detections[0][0,0][0]
        for i in range(detections.shape[0]):
            d = detections[i,:]
            xtl = max(0, d[0])
            ytl = max(0, d[1])
            xbr = d[2]
            ybr = d[3]
            score = d[-1]
            boxes.append(Box(xtl, ytl, xbr, ybr, frame, score = score))
    return boxes

boxes = exemplarsvm('/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table/pedro-pascal-bottle.mat')

print boxes[-1]
