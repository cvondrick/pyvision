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

def track(video, seeds, patches, projections, double delta = 10e-3):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """
    cdef double xmin, xmax, ymin, ymax, zmin, zmax
    cdef double x, y, z
    cdef int xi, yi, zi
    cdef int xs, ys, zs
    cdef int width, height
    cdef double normalizer, prob3d
    cdef double px, py, pn
    cdef int pxi, pyi, pxii, pyii
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=3] mapping, intersection
    cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
    cdef numpy.ndarray[numpy.double_t, ndim=2] prob2map, intersection2map

    bounds = pmvs.get_patch_bounds(patches)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    logger.debug("x-bounds are: {0} thru {1}".format(xmin, xmax))
    logger.debug("y-bounds are: {0} thru {1}".format(ymin, ymax))
    logger.debug("z-bounds are: {0} thru {1}".format(zmin, zmax))

    logger.debug("Building 3D density with delta={0}".format(delta))
    xs = <int> ((xmax - xmin) / delta)
    ys = <int> ((ymax - ymin) / delta)
    zs = <int> ((zmax - zmin) / delta)
    logger.debug("Trying to allocate {0}x{1}x{2}={3}".format(xs, ys, zs, 
                                                             xs*ys*zs))
    mapping = numpy.zeros((xs, ys, zs))
    intersection = numpy.ones((xs, ys, zs))
    logger.debug("Allocated")
    normalizer = 0
    for seednum, seed in enumerate(seeds):
        logger.debug("Processing seed {0}".format(seednum))
        matrix = projections[seed.frame].matrix
        for xi in range(xs):
            x = xi * delta + xmin
            for yi in range(ys):
                y = yi * delta + ymin
                for zi in range(zs):
                    z = zi * delta + zmin
                    pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
                    if pn < 0:
                        intersection[xi, yi, zi] = 0
                        continue
                    px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
                    py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn
                    if seed.xtl <= px and seed.xbr >= px and seed.ytl <= py and seed.ybr >= py:
                        mapping[xi, yi, zi] += 1
                        normalizer += 1
                    else:
                        intersection[xi, yi, zi] = 0

    if normalizer > 0:
        mapping = mapping / normalizer
        logger.debug("Normalizer is {0}".format(normalizer))
    else:
        raise RuntimeError("Normalizer is 0, probably no points mapped")
    
    cdef double sigma = 0.0001
    normalizer = 0
    for xi in range(xs):
        for yi in range(ys):
            for zi in range(zs):
                e = exp(mapping[xi, yi, zi] / sigma)
                normalizer += e
                mapping[xi, yi, zi] = e

    cdef int radius = 10
    for frame, projection in enumerate(projections):
        logger.debug("Projecting into {0}".format(frame))
        matrix = projections[frame].matrix
        prob2map = numpy.zeros(video[frame].size)
        intersection2map = numpy.zeros(video[frame].size)
        width, height = video[frame].size
        normalizer = 0
        points = []
        for x from xmin <= x <= xmax by delta: 
            for y from ymin <= y <= ymax by delta:
                for z from zmin <= z <= zmax by delta:
                    xi = <int> ((x - xmin) / delta)
                    yi = <int> ((y - ymin) / delta)
                    zi = <int> ((z - zmin) / delta)
                    prob3d = mapping[xi, yi, zi]
                    pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
                    if pn < 0:
                        continue
                    pxi = <int>((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]*z+matrix[0,3])/pn)
                    pyi = <int>((matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]*z+matrix[1,3])/pn)
                    for pxii in range(pxi - radius, pxi + radius + 1):
                        for pyii in range(pyi - radius, pyi + radius + 1):
                            if pxii < 0 or pyii < 0 or pxii >= width or pyii >= height:
                                continue
                            prob2map[pxii, pyii] += prob3d > 0
                            normalizer += prob3d
                            if intersection[xi, yi, zi]:
                                intersection2map[pxii, pyii] = max(intersection2map[pxii, pyii], 1)
        if normalizer > 0:
            prob2map = prob2map / normalizer

            logger.debug("Ploting")
            plt.subplot(221)
            plt.title("Prob of object")
            plt.set_cmap("gray")
            plt.title("min = {0}, max = {1}".format(prob2map.min(), prob2map.max()))
            plt.imshow(prob2map.transpose())
            plt.subplot(222)
            plt.title("Frame")
            plt.imshow(numpy.asarray(video[frame]))
            plt.subplot(223)
            plt.title("Prob mask")
            plt.imshow(numpy.array(numpy.asarray(video[frame]) * numpy.tile(prob2map / prob2map.max(), (3, 1, 1)).T, numpy.uint8))
            plt.subplot(224)
            plt.title("Intersection mask")
            plt.imshow(numpy.array(numpy.asarray(video[frame]) * numpy.tile(intersection2map, (3, 1, 1)).T, numpy.uint8))
            plt.savefig("tmp/prob{0}.png".format(frame))
            plt.clf()
        else:
            logger.warning("Normalizer for frame {0} is 0".format(frame))

    logger.debug("Writing 3D probability map")
    plywriter.write(open("mapping.ply", "w"), mapping,
        condition = plywriter.filterlower, bounds = bounds)

    logger.debug("Writing intersection map")
    plywriter.write(open("intersection.ply", "w"), intersection,
        condition = plywriter.filterlower, bounds = bounds)
