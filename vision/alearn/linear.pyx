import math, logging, multiprocessing, numpy
from vision import annotations, convolution, model
from vision.track import interpolation

cimport numpy
from vision cimport annotations

cdef extern from "math.h":
    float exp(float n)

log = logging.getLogger("vision.track.alearn")

def pick(images, path, dim = (40, 40), errortube = 100,
         double sigma = 0.1, bgskip = 4, bgsize = 5e4,
         skip = 1, plot = False, pool = None):
    """
    Given a path, picks the most informative frame that we currently lack.
    """
    log.info("Picking most informative frame through active learning")
    svm = model.PathModel(images, path, dim = dim,
                          bgskip = bgskip, bgsize = bgsize)
    scores = []

    log.info("Scoring frames")
    for prev, cur in zip(path, path[1:]):
        lpath = interpolation.Linear(prev, cur)[1:-1:skip]
        workorders = [(x, images, svm, prev, cur, 
                       dim, errortube, sigma, plot) for x in lpath]
        if pool:
            scores.extend(pool.map(score_frame_do, workorders))
        else:
            for workorder in workorders:
                scores.append(score_frame_do(workorder))

    best = max([min(x) for x in zip(*[scores[y:] for y in range(25)])])[1]

    if plot:
        import matplotlib.pyplot as plt
        log.info("Saving score plot")
        plt.close()
        plt.plot([x[1] for x in scores], [x[0] for x in scores])
        plt.grid()
        plt.savefig("{0}/scoreplot.png".format(plot))

    return best

def score_frame_do(workorder):
    return score_frame(*workorder), workorder[0].frame
    
def score_frame(annotations.Box linearbox, images, svm,
                annotations.Box previous, annotations.Box current, dim,
                int errortube, double sigma = 10, plot = False):
    """
    Scores an individual frame to determine its usefulness. A higher score
    is more useful than a lower score from a different frame.
    """
    log.info("Scoring frame {0}".format(linearbox.frame))

    cdef double wr = (<float>dim[0]) / linearbox.width
    cdef double hr = (<float>dim[1]) / linearbox.height

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

    cdef double framearea = (w/wr) * (h/hr)
    cdef int framedifference = current.frame - previous.frame
    score = 0
    normalizer = 0

    cdef numpy.ndarray[numpy.double_t, ndim=2] dlinearim = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] dprobim = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] dscoreim = numpy.zeros((w, h))

    cdef double lineardiffy, lineardiffx, lineardiff
    cdef int i, j
    for i in range(pstartx, pstopx):
        for j in range(pstarty, pstopy): 
            # calculate area difference through cross product
            lineardiffy  = current.xtl - previous.xtl
            lineardiffy *= linearbox.frame - previous.frame
            lineardiffy -= (i / wr - previous.xtl) * framedifference

            lineardiffx  = current.ytl - previous.ytl
            lineardiffx *= linearbox.frame - previous.frame
            lineardiffx -= (j / hr - previous.ytl) * framedifference

            lineardiff   = lineardiffx * lineardiffx
            lineardiff  += lineardiffy * lineardiffy 

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
        import matplotlib.pyplot as plt
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
        plt.savefig("{0}/prob{1}.png".format(plot, linearbox.frame))
        plt.clf()

    return score / (normalizer * framearea)
