import logging
from vision.reconstruction import pmvs
import vision

logger = logging.getLogger("vision.track.realcoords")

def track(video, seed, mapping, pool = None):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """

    logger.debug("Calculating active region in 3-space")
    xtl, ytl, xbr, ybr = seed[0:4]
    projection = mapping.projections[seed.image]
    activeregion = []
    for x in range(xtl, xbr):
        for y in range(ytl, ybr):
            activeregion.append(mapping.imagetoreal(projection, (x, y)))

    logger.debug("Projecting")
    resp = []
    for image in range(len(video)):
        resp.append(track_project(image, video, mapping, activeregion))

    return resp

def track_project(image, video, mapping, activeregion):
    logger.debug("Projecting points into {0}".format(image))
    coords = []
    for real in activeregion:
        images = mapping.realtoimages(real)
        if image in images:
            coord = images[image]
            coords.append(coord)

    if len(coords) == 0:
        logger.debug("Point did not map, so assume lost")
        box = vision.Box(0, 0, 1, 1, image, lost = 1)
    else:
        bxtl = int(min(i[0] for i in coords))
        bytl = int(min(i[1] for i in coords))
        bxbr = int(max(i[0] for i in coords))
        bybr = int(max(i[1] for i in coords))

        w, h = video[image].size

        if bxtl < 0:
            bxtl = 0
        if bytl < 0:
            bytl = 0
        if bxbr >= w:
            bxbr = w
        if bybr >= h:
            bybr = h

        if bxbr < 0 or bybr < 0 or bxtl >= w or bytl > h:
            logger.debug("Object is lost")
            box = vision.Box(0, 0, 1, 1, image, lost = 1)
        else:
            if bxtl >= bxbr:
                logger.warning("Real X coord bounds is a point, adjusting")
                bxbr = bxtl + 1
            if bytl >= bybr:
                logger.warning("Real Y coord bounds is a point, adjusting")
                bybr = bytl + 1

            box = vision.Box(bxtl, bytl, bxbr, bybr, image)

    return box

if __name__ == "__main__":
    import vision.drawer
    import os.path
    from vision import visualize
    import multiprocessing

    pool = multiprocessing.Pool(16)
    #pool = None

    logging.basicConfig(level = logging.DEBUG)

    path = ("/csail/vision-videolabelme/databases/"
            "video_adapt/home_ac_a/frames/0/bundler")

    video = vision.flatframeiterator(path, 1, 5)

    root = os.path.join(path, "pmvs")
    mapping = pmvs.RealWorldMap(*pmvs.read(root))

    seed = vision.Box(100, 100, 120, 120, 1)
    seed.image = 1
    predicted = track(video, seed, mapping, pool)

    vit = visualize.highlight_path(video, predicted)
    visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
