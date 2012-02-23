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

class ThreeD(object):
    def __init__(self, video, patches, projections):
        self.video = video
        self.patches = patches
        self.projections = projections

    def build(self, seeds):
        cdef double x, y, z
        cdef double normalizer, score 
        cdef double px, py, pn
        cdef annotations.Box seed
        cdef numpy.ndarray[numpy.double_t, ndim=2] matrix

        logger.info("Building 3D model")

        logger.debug("Cleaning seeds")
        useseeds = []
        for seed in seeds:
            if seed.frame not in self.projections:
                logger.warning("Dropping seed {0} because "
                               "no projection".format(seed.frame))
            else:
                useseeds.append(seed)
        seeds = useseeds

        if not seeds:
            logger.warning("No seeds")

        logger.info("Using seeds in {0}".format(", ".join(str(x.frame) for x in seeds)))

        logger.debug("Voting in 3-space")
        self.mapping = {}
        for patch in self.patches:
            score = 0
            for seed in seeds:
                matrix = self.projections[seed.frame].matrix
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
                self.mapping[x, y, z] = score
        self.normalizer = normalizer

        return self

    def hasprojection(self, frame):
        return frame in self.projections

    def scorelocations(self, frame, int radius = 10):
        cdef numpy.ndarray[numpy.double_t, ndim=2] prob2map
        cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
        cdef int pxi, pxii, pyi, pyii
        cdef double x, y, z
        cdef double normalizer2d, normalizer, prob3d

        videoframe = self.video[frame]
        prob2map = numpy.zeros(videoframe.size)

        if frame not in self.projections:
            return prob2map

        projection = self.projections[frame]
        matrix = projection.matrix
        prob2map = numpy.zeros(videoframe.size)
        width, height = videoframe.size
        points = []
        normalizer2d = 0
        normalizer = self.normalizer
        for (x, y, z), prob3d in self.mapping.iteritems():
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
        return prob2map
