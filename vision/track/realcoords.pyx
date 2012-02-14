import logging
from vision.reconstruction import pmvs
import numpy
import vision

cimport numpy

logger = logging.getLogger("vision.track.realcoords")

def track(video, seeds, patches, projections, double delta = 10e-3):
    """
    This tracker uses a 3D reconstruction of a scene in order to
    localize objects throughout a video sequence.
    """
    cdef int xmin, xmax, ymin, ymax, zmin, zmax
    cdef double x, y, z
    cdef double normalizer
    cdef numpy.ndarray[numpy.double_t, ndim=3] mapping

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = pmvs.get_patch_bounds(patches)

    mapping = numpy.zeros(((xmax - xmin) / delta,
                           (ymax - ymin) / delta,
                           (zmax - zmin) / delta))
    normalizer = 0
    for x from xmin <= x <= xmax by delta: 
        for y from ymin <= y <= ymax by delta:
            for z from zmin <= z <= zmax by delta:
                points = 0
                real = numpy.array([x, y, z, 1])
                for seed in seeds:
                    projection = projections[seed.image]
                    points += seed.contains(projection.project(real))
                mapping[x, y, z] = points
                normalizer += points
    mapping = mapping / normalizer
