import logging
from vision.reconstruction import pmvs
import numpy
import vision
from vision.reconstruction import plywriter

cimport numpy
from vision cimport annotations

logger = logging.getLogger("vision.track.realcoords")

def track(video, seeds, patches, projections, double delta = 10e-3 * 2):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """
    cdef double xmin, xmax, ymin, ymax, zmin, zmax
    cdef double x, y, z
    cdef int xi, yi, zi
    cdef int xs, ys, zs
    cdef double normalizer
    cdef double px, py, pn
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=3] mapping 
    cdef numpy.ndarray[numpy.double_t, ndim=2] matrix

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
                    px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
                    py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn
                    if seed.xtl <= px and seed.xbr >= px and seed.ytl <= py and seed.ybr >= py:
                        mapping[xi, yi, zi] += 1
                        normalizer += 1
    if normalizer > 0:
        mapping = mapping / normalizer
        logger.debug("Normalizer is {0}".format(normalizer))
    else:
        raise RuntimeError("Normalizer is 0, probably no points mapped")

#    for frame, projection in enumerate(projections):
#        matrix = projections[frame].matrix
#        for x from xmin <= x <= xmax by delta: 
#            for y from ymin <= y <= ymax by delta:
#                for z from zmin <= z <= zmax by delta:
#                    xi = <int> ((x - xmin) / delta)
#                    yi = <int> ((y - ymin) / delta)
#                    zi = <int> ((z - zmin) / delta)
#                    prob3d = mapping[xi, yi, zi]
#                    pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
#                    px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
#                    py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn

    logger.debug("Writing 3D probability map")
    plywriter.write(open("mapping.ply", "w"), mapping,
        condition = plywriter.filterlower, bounds = bounds)
