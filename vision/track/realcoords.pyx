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
    cdef double normalizer, normalizer2d, score, prob3d
    cdef double px, py, pn
    cdef annotations.Box seed
    cdef numpy.ndarray[numpy.double_t, ndim=2] matrix
    cdef numpy.ndarray[numpy.double_t, ndim=2] prob2map, intersection2map

    logger.debug("Cleaning seeds")
    useseeds = []
    for seed in seeds:
        if seed.frame not in projections:
            logger.warning("Dropping seed {0} because no projection".format(seed.frame))
        else:
            useseeds.append(seed)
    seeds = useseeds

    if not seeds:
        raise RuntimeError("No seeds")

    logger.debug("Using seeds in {0}".format(", ".join(str(x.frame) for x in seeds)))

    logger.debug("Voting in 3-space")
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
    for frame, projection in sorted(projections.items()):
        logger.debug("Projecting into {0}".format(frame))
        matrix = projection.matrix
        videoframe = video[frame]
        prob2map = numpy.zeros(videoframe.size)
        intersection2map = numpy.zeros(videoframe.size)
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

        for x, y, z in intersection:
            pn = matrix[2,0]*x + matrix[2,1]*y +matrix[2,2]*z + matrix[2,3]
            if pn < 0:
                continue
            pxi = <int>((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2]*z+matrix[0,3])/pn)
            pyi = <int>((matrix[1,0]*x+matrix[1,1]*y+matrix[1,2]*z+matrix[1,3])/pn)
            for pxii in range(pxi - radius, pxi + radius + 1):
                for pyii in range(pyi - radius, pyi + radius + 1):
                    if pxii < 0 or pyii < 0 or pxii >= width or pyii >= height:
                        continue
                    intersection2map[pxii, pyii] = 1

        if normalizer2d > 0:
            prob2map = prob2map  / normalizer2d

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

    #logger.debug("Writing 3D probability map")
    #plywriter.write(open("mapping.ply", "w"), mapping,
    #    condition = plywriter.filterlower, bounds = pmvs.get_patch_bounds(patches))

   # logger.debug("Writing intersection map")
   # plywriter.write(open("intersection.ply", "w"), intersection,
   #     condition = plywriter.filterlower, bounds = bounds)
