import math, logging, multiprocessing, numpy
from vision import annotations, convolution, model
import interpolation

cimport numpy
from vision cimport annotations
import matplotlib.pyplot as plt

cdef extern from "math.h":
    float exp(float n)

log = logging.getLogger("alearn")

def pick(images, path, dim = (40, 40), errortube = 100,
         double sigma = 0.1, plot = False):
    """
    Given a path, picks the most informative frame that we currently lack.
    """
    log.info("Picking most informative frame through active learning")
    svm = model.PathModel(images, path, dim = dim)
    scores = []

    log.info("Scoring frames")
    for prev, cur in zip(path, path[1:]):
        lpath = interpolation.Linear(prev, cur)
        for box in lpath[1:-1]:
            framescore = score_frame(box, images, svm, prev, cur, \
                dim, errortube, sigma, plot)
            scores.append((framescore, box.frame))
    best = max(scores)[1]

    if plot:
        log.info("Saving score plot")
        plt.close()
        plt.plot([x[1] for x in scores], [x[0] for x in scores])
        plt.grid()
        plt.savefig("tmp/scoreplot.png")

    return best

def score_frame(annotations.Box linearbox, images, svm,
                annotations.Box previous, annotations.Box current, dim,
                int errortube, double sigma = 10, plot = False):
    """
    Scores an individual frame to determine its usefulness. A higher score
    is more useful than a lower score from a different frame.
    """
    log.info("Scoring frame {0}".format(linearbox.frame))

    wr = (<float>dim[0]) / linearbox.get_width()
    hr = (<float>dim[1]) / linearbox.get_height()

    im = images[linearbox.frame]
    im = im.resize((int(im.size[0] * wr), int(im.size[1] * hr)))

    cdef int w = im.size[0] - dim[0]
    cdef int h = im.size[1] - dim[1]

    cdef numpy.ndarray[numpy.double_t, ndim=2] costim
    costim = convolution.hogrgb(im, dim, svm.hogweights(), svm.rgbweights())

    pstartx = linearbox.xtl - errortube
    pstopx  = linearbox.xtl + errortube
    pstarty = linearbox.ytl - errortube
    pstopy  = linearbox.ytl + errortube

    if pstartx < 0:
        pstartx = 0
    if pstopx > w:
        pstopx = w
    if pstarty < 0:
        pstarty = 0
    if pstopy > h:
        pstopy = h

    framearea = (w/wr) * (h/hr)
    score = 0
    normalizer = 0

    cdef numpy.ndarray[numpy.double_t, ndim=2] dlinearim = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] dprobim = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] dscoreim = numpy.zeros((w, h))

    for i in range(pstartx, pstopx):
        for j in range(pstarty, pstopy): 
            # calculate area difference through cross product
            lineardiffy  = (current.xtl - previous.xtl) * \
                (linearbox.frame - previous.frame)
            lineardiffy -= (i / wr      - previous.xtl) * \
                (current.frame   - previous.frame)
            lineardiffx  = (current.ytl - previous.ytl) * \
                (linearbox.frame - previous.frame)
            lineardiffx -= (j / hr      - previous.ytl) * \
                (current.frame   - previous.frame)
            lineardiff   = lineardiffx * lineardiffx + \
                lineardiffy * lineardiffy 

            # compute local score
            matchscore   = exp(-costim[i,j] / sigma)
            localscore   = matchscore * lineardiff
            # store for total score
            score       += localscore 
            normalizer  += matchscore

            dlinearim[i,j] = lineardiff
            dprobim[i,j] = matchscore
            dscoreim[i,j] = localscore

    if plot:
        plt.subplot(221)
        plt.set_cmap("gray")
        plt.imshow(dprobim.transpose() / normalizer)
        plt.title("probability")

        plt.subplot(222)
        plt.set_cmap("gray")
        plt.imshow(dlinearim.transpose() / normalizer)
        plt.title("linear diff")

        plt.subplot(223)
        plt.set_cmap("gray")
        plt.imshow(dscoreim.transpose() / normalizer)
        plt.title("expected reduction")

        plt.subplot(224)
        plt.imshow(numpy.asarray(im))
        plt.title("data")
        plt.savefig("tmp/prob{0}.png".format(linearbox.frame))
        plt.clf()

    return score / (normalizer * framearea)
