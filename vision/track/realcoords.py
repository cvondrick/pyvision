import logging
from vision.reconstruction.pmvs import *
import vision

logger = logging.getLogger("vision.track.realcoords")

def track(video, seed, mapping):
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

    resp = []

    for image, frame in enumerate(video):
        logger.debug("Projecting points into {0}".format(image))
        coords = []
        for real in activeregion:
            coord = mapping.realtoimages(real)[image]
            coords.append(coord)

        bxtl = int(min(i[0] for i in coords))
        bytl = int(min(i[1] for i in coords))
        bxbr = int(max(i[0] for i in coords))
        bybr = int(max(i[1] for i in coords))

        resp.append(vision.Box(bxtl, bytl, bxbr, bybr, image))

    return resp

if __name__ == "__main__":
    import vision.drawer
    import os.path
    from vision import visualize

    logging.basicConfig(level = logging.DEBUG)

    path = ("/csail/vision-videolabelme/databases/"
            "video_adapt/home_ac_a/frames/0/bundler")

    patches, projections = read(os.path.join(path, "pmvs"))

    mapping = RealWorldMap(patches[0:100], projections)
    video = vision.flatframeiterator(path, 1, 5)

    seed = vision.Box(100, 100, 120, 120, 1)
    seed.image = 1
    predicted = track(video, seed, mapping)

    vit = visualize.highlight_path(video, predicted)
    visualize.save(vit, lambda x: "tmp/path{0}.jpg".format(x))
