import logging
from vision.reconstruction import pmvs
import numpy
import vision
import matplotlib.pyplot as plt
from vision.reconstruction import plywriter

cimport numpy
from vision cimport annotations

logger = logging.getLogger("vision.track.realcoords")

cdef extern from "math.h":
    float exp(float n)

def track(video, seeds, patches, projections):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """
    cdef double x, y, z
    cdef int pxi, pxii, pyi, pyii
    cdef int width, height
    cdef int xmin, xmax, ymin, ymax
    cdef double normalizer, normalizer2d, score, prob3d
    cdef double px, py, pn
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
    cdef numpy.ndarray[numpy.double_t, ndim=2] prob2map

    logger.info("Cleaning seeds")
    useseeds = []
    for seed in seeds:
        if seed.frame not in projections:
            logger.warning("Dropping seed {0} because no projection".format(seed.frame))
        else:
            useseeds.append(seed)
    seeds = useseeds

    if not seeds:
        raise RuntimeError("No seeds")

    logger.info("Using seeds in {0}".format(", ".join(str(x.frame) for x in seeds)))

    logger.info("Voting in 3-space")
    mapping = {}
    intersection = set()
    for patch in patches:
        score = 0
        for seed in seeds:
            matrix = projections[seed.frame].matrix
            x, y, z, _ = patch.realcoords
            pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
            if pn < 0:
                continue
            px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
            py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn
            if seed.xtl <= px and seed.xbr >= px and seed.ytl <= py and seed.ybr >= py:
                score += 1
        if score > 0:
            normalizer += score
            mapping[x, y, z] = score
        if score == len(seeds):
            intersection.add((x, y, z))

    cdef int radius = 10
    boxes = []
    logger.info("Projecting votes into 2-space")
    for frame, projection in sorted(projections.items()):
        matrix = projection.matrix
        videoframe = video[frame]
        prob2map = numpy.zeros(videoframe.size)
        width, height = video[frame].size
        points = []
        normalizer2d = 0

        for (x, y, z), prob3d in mapping.iteritems():
            prob3d = prob3d / normalizer
            pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
            if pn < 0:
                continue
            pxi = <int>((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]*z+matrix[0,3])/pn)
            pyi = <int>((matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]*z+matrix[1,3])/pn)
            for pxii in range(pxi - radius, pxi + radius + 1):
                for pyii in range(pyi - radius, pyi + radius + 1):
                    if pxii < 0 or pyii < 0 or pxii >= width or pyii >= height:
                        continue
                    prob2map[pxii, pyii] += prob3d
                    normalizer2d += prob3d
        prob2map /= normalizer2d


        boxes.append(vision.Box(xmin, ymin, xmax, ymax, frame))
    return boxes
