import logging
from vision.reconstruction import pmvs
import numpy
import vision
import matplotlib.pyplot as plt
import vision.convolution
from vision.reconstruction import plywriter

cimport numpy
from vision cimport annotations

logger = logging.getLogger("vision.track.realcoords")

cdef extern from "math.h":
    float exp(float n)

class ThreeD(object):
    def __init__(self, video, patches, projections, sigma = 1):
        self.video = video
        self.patches = patches
        self.projections = projections
        self.built = False
        self.sigma = sigma

    def build(self, seeds, forcescore = None, negatives = [], posprune = 0, negprune = 0.1):
        cdef double x, y, z
        cdef double normalizer, score, mod
        cdef double px, py, pn
        cdef annotations.Box seed
        cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
        cdef double sigma = self.sigma

        logger.info("Building 3D model")

        logger.debug("Cleaning seeds")
        useseeds = []
        for seed in seeds:
            if seed.frame in self.projections:
                if exp(seed.score / sigma) > posprune:
                    useseeds.append((1, seed))
        seeds = useseeds
        logger.info("Using {0} seeds".format(len(seeds)))

        cdef double lower = 0
        if negatives:
            usenegatives = []
            for negative in negatives:
                if negative.frame in self.projections:
                    negscore = exp(negative.score / sigma)             
                    if negscore > negprune:
                        usenegatives.append((-1, negative))
                        if negscore > lower:
                            lower = negscore
            seeds.extend(usenegatives)
            logger.info("Using {0} negatives".format(len(usenegatives)))

        if forcescore is not None:
            for seed in seeds:
                seed.score = forcescore

        if not seeds:
            logger.warning("No usable seeds")

        logger.info("Voting in 3-space")
        self.mapping = {}
        numpatches = len(self.patches)
        for num, patch in enumerate(self.patches):
            if num % 100 == 0 and num > 0:
                logger.debug("Voted for {0} of {1} patches".format(num, numpatches))
            score = 0
            for mod, seed in seeds:
                matrix = self.projections[seed.frame].matrix
                x, y, z, _ = patch.realcoords
                pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
                if pn < 0:
                    continue
                px = (matrix[0,0]*x + matrix[0,1]*y +matrix[0,2]*z + matrix[0,3]) / pn
                py = (matrix[1,0]*x + matrix[1,1]*y +matrix[1,2]*z + matrix[1,3]) / pn
                if seed.xtl <= px and seed.xbr >= px and seed.ytl <= py and seed.ybr >= py:
                    score += mod * exp(seed.score / sigma) + lower
                normalizer += score
                self.mapping[x, y, z] = score
        self.normalizer = normalizer
        self.built = True

        if self.normalizer == 0:
            logger.warning("Normalizer in 3D is 0")
        return self

    def hasprojection(self, frame):
        return frame in self.projections

    def scorelocations(self, frame, int radius = 10):
        cdef numpy.ndarray[numpy.double_t, ndim=2] prob2map
        cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
        cdef long pxi, pxii, pyi, pyii
        cdef double x, y, z
        cdef double px, py, pn
        cdef double normalizer2d, normalizer, prob3d
        cdef int width, height

        videoframe = self.video[frame]
        prob2map = numpy.zeros(videoframe.size)

        if frame not in self.projections:
            logger.warning("Frame {0} cannot project".format(frame))
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
            pxi = <long>((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]*z+matrix[0,3])/pn)
            pyi = <long>((matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]*z+matrix[1,3])/pn)
            for pxii in range(pxi - radius, pxi + radius + 1):
                for pyii in range(pyi - radius, pyi + radius + 1):
                    if pxii < 0 or pyii < 0 or pxii >= width or pyii >= height:
                        continue
                    prob2map[pxii, pyii] += prob3d
                    normalizer2d += prob3d
        if normalizer2d == 0:
            logger.warning("Normalizer for frame {0} is 0".format(frame))
        #prob2map /= normalizer2d
        return prob2map

    def scoreall(self, radius = 10):
        for frame in self.frames():
            yield frame, self.scorelocations(frame, radius)

    def frames(self):
        return sorted(self.projections)

    def boxfit(self, frame, radius = 10, double alpha = 1.0, int skip = 5):
        cdef int xmin, ymin, xmax, ymax
        cdef int bestxmin = 0, bestymin = 0, bestxmax = 0, bestymax = 0
        cdef int width, height
        cdef double score, bestscore = 0
        cdef numpy.ndarray[numpy.double_t, ndim=2] map, sumarea
        map = self.scorelocations(frame, radius)
        sumarea = vision.convolution.sumprob(map)

        videoframe = self.video[frame]
        width, height = videoframe.size

        for xmin in range(0, width - 1, skip):
            for ymin in range(0, height - 1, skip):
                for xmax in range(xmin + 1, width, skip):
                    for ymax in range(ymin + 1, height, skip):
                        score = sumarea[xmin, ymin]
                        score += sumarea[xmax, ymax]
                        score -= sumarea[xmin, ymax]
                        score -= sumarea[xmax, ymin]
                        score -= alpha * (xmax - xmin) * (ymax - ymin)

                        if score > bestscore:
                            bestscore = score
                            bestxmin = xmin
                            bestymin = ymin
                            bestxmax = xmax
                            bestymax = ymax
        if bestxmax <= bestxmin and bestymax <= bestymin:
            return None
        return vision.Box(bestxmin, bestymin,
                          bestxmax, bestymax,
                          frame, score = bestscore)
