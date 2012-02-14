import logging
from vision.reconstruction import pmvs
import numpy
import vision

cimport numpy
from vision cimport annotations

logger = logging.getLogger("vision.track.realcoords")

def track(video, seeds, patches, projections, double delta = 10e-4):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """
    cdef double xmin, xmax, ymin, ymax, zmin, zmax
    cdef double x, y, z
    cdef int xi, yi, zi
    cdef double normalizer
    cdef int contains
    cdef int sxtl, sytl, sxbr, sybr
    cdef double px, py, pn
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=3] mapping
    cdef numpy.ndarray[numpy.double_t, ndim=2] matrix

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = pmvs.get_patch_bounds(patches)

    logger.debug("x-bounds are: {0} thru {1}".format(xmin, xmax))
    logger.debug("y-bounds are: {0} thru {1}".format(ymin, ymax))
    logger.debug("z-bounds are: {0} thru {1}".format(zmin, zmax))

    logger.debug("Building 3D density")
    mapping = numpy.zeros(((xmax - xmin + 1) / delta,
                           (ymax - ymin + 1) / delta,
                           (zmax - zmin + 1) / delta))
    normalizer = 0
    for seed in seeds:
        matrix = projections[seed.frame].matrix
        for x from xmin <= x <= xmax by delta: 
            for y from ymin <= y <= ymax by delta:
                for z from zmin <= z <= zmax by delta:
                    pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
                    px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
                    py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn
                    if seed.xtl <= px and seed.xbr >= px and seed.ytl <= py and seed.ybr >= py:
                        xi = <int> ((x - xmin) / delta)
                        yi = <int> ((y - ymin) / delta)
                        zi = <int> ((z - zmin) / delta)
                        mapping[xi, yi, zi] += 1
                        normalizer += 1
    if normalizer > 0:
        mapping = mapping / normalizer
        logger.debug("Normalizer is {0}".format(normalizer))
    else:
        raise RuntimeError("Normalizer is 0, probably no points mapped")
