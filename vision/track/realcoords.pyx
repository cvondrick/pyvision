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
    cdef double normalizer, points
    cdef int contains
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=3] mapping

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = pmvs.get_patch_bounds(patches)

    logger.debug("x-bounds are: {0} thru {1}".format(xmin, xmax))
    logger.debug("y-bounds are: {0} thru {1}".format(ymin, ymax))
    logger.debug("z-bounds are: {0} thru {1}".format(zmin, zmax))

    logger.debug("Building 3D density")
    mapping = numpy.empty(((xmax - xmin + 1) / delta,
                           (ymax - ymin + 1) / delta,
                           (zmax - zmin + 1) / delta))
    normalizer = 0
    for x from xmin <= x <= xmax by delta: 
        for y from ymin <= y <= ymax by delta:
            for z from zmin <= z <= zmax by delta:
                points = 0
                real = numpy.array([x, y, z, 1])
                for seed in seeds:
                    point = projections[seed.frame].project(real)
                    contains = (seed.xtl >= point[0] and seed.xbr <= point[0] and
                                seed.ytl >= point[1] and seed.ybr <= point[1])
                    if contains:
                        points += 1.0
                xi = <int> ((x - xmin) / delta)
                yi = <int> ((y - ymin) / delta)
                zi = <int> ((z - zmin) / delta)
                mapping[xi, yi, zi] = points
                normalizer += points
    if normalizer > 0:
        mapping = mapping / normalizer
    else:
        raise RuntimeError("Normalizer is 0, probably no points mapped")
