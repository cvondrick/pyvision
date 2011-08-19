from vision import convolution
from vision import Box
from vision.track import interpolation
from vision.track import dp
from vision.model import PathModel
import logging
import numpy

cimport numpy

from vision cimport annotations

cdef int debug = 1

if debug:
    import pylab

logger = logging.getLogger("vision.alearn.marginals")

cdef extern from "math.h":
    float exp(float n)

cdef double Infinity = 1e300

def pick(givens, images, last = None, 
         pairwisecost = 0.001, upperthreshold = 10, lowerthreshold = -100,
         sigma = .1, erroroverlap = 0.5, skip = 3, dim = (40, 40), 
         rgbbin = 8, hogbin = 8, c = 1, clickradius = 10, pool = None):

    givens.sort(key = lambda x: x.frame)

    model = PathModel(images, givens, rgbbin = rgbbin, hogbin = hogbin, c = c)
    
    fullpath = []
    fullmarginals = []
    for x, y in zip(givens, givens[1:]):
        if x.frame == y.frame:
            raise RuntimeError("Frame {0} appears twice".format(x.frame))
        logger.info("Scoring {0} to {1}".format(x.frame, y.frame))
        marginals, path = picksegment(x, y, model, images, pairwisecost,
                                        upperthreshold, lowerthreshold,
                                        sigma, erroroverlap,
                                        skip, clickradius, pool)
        fullpath.extend(path[:-1])
        fullmarginals.extend(marginals[:-1])

    if last is not None and last >= givens[-1].frame:
        logger.info("Scoring {0} to last at {1}".format(givens[-1].frame, last))
        marginals, path = picksegment(givens[-1], last, model, images,
                                      pairwisecost, upperthreshold,
                                      lowerthreshold, sigma,
                                      erroroverlap, skip, clickradius, pool)
        fullpath.extend(path[:-1])
        fullmarginals.extend(marginals)

    if debug:
        pylab.plot([x[1] for x in fullmarginals],
                   [x[0] for x in fullmarginals])
        pylab.grid()
        pylab.savefig("tmp/scoreplot.png")

    best = max(fullmarginals)
    marginals = dict((x[1], x[0]) for x in fullmarginals)

    return best[1], best[0], fullpath, marginals

def picksegment(start, stop, model, images,
               pairwisecost = 0.001, upperthreshold = 10,
               lowerthreshold = -100, sigma = .1, erroroverlap = 0.5, skip = 3,
               clickradius = 10, pool = None):

    if pool:
        logger.info("Found a process pool, so attempting to parallelize")
        mapper = pool.map
    else:
        logger.info("No process pool found, so remaining single threaded")
        mapper = map

    imagesize = images[0].size

    try:
        stopframe = stop.frame
    except:
        stopframe = stop
        constrained = False
        constraints = [start]
    else:
        constrained = True
        # adjust stop for scaling reasons
        if stop.xtl + start.width >= imagesize[0]:
            stopxtl = imagesize[0] - start.width - 1
        else:
            stopxtl = stop.xtl
        if stop.ytl + start.height >= imagesize[1]:
            stopytl = imagesize[1] - start.height - 1
        else:
            stopytl = stop.ytl
        stop = Box(stopxtl, stopytl, stopxtl + start.width,
                   stopytl + start.height, stop.frame)
        constraints = [start, stop]

    if start.frame == stopframe:
        return [(0, start.frame)], [start] 
    elif start.frame + 1 == stopframe:
        if not constrained:
            stop = Box(*start)
            stop.frame = stopframe
        return [(0, start.frame), (0, stopframe)], [start, stop] 

    frames = range(start.frame, stopframe + 1)

    # build dictionary of local scores
    # if there is a pool, this will happen in parallel
    logger.info("Scoring frames")
    orders = [(images, start, x, model) for x in frames]
    costs = dict(mapper(dp.scoreframe, orders))

    # forward and backwards passes
    # if there is a pool, this will use up to 2 cores
    forwardsargs  = [frames, imagesize, model, costs, pairwisecost,
                     upperthreshold, lowerthreshold, skip, constraints]
    backwardsargs = list(forwardsargs)
    backwardsargs[0] = list(reversed(frames))
    if False and pool:
        logger.info("Building forwards and backwards graphs")
        forwards  = pool.apply_async(dp.buildgraph, forwardsargs)
        backwards = dp.buildgraph(*backwardsargs)
        forwards  = forwards.get() # blocks until forwards graph done
    else:
        logger.info("Building forwards graph")
        forwards  = dp.buildgraph(*forwardsargs)
        logger.info("Building backwards graph")
        backwards = dp.buildgraph(*backwardsargs)
    
    # backtrack
    logger.info("Backtracking")
    if constrained:
        x, y, frame = stop.xtl, stop.ytl, stop.frame
    else:
        x, y = numpy.unravel_index(numpy.argmin(forwards[stopframe][0]),
                                                forwards[stopframe][0].shape)
        x = x * skip
        y = y * skip
        frame = stopframe
    path = []
    while frame > start.frame:
        path.append(annotations.Box(x, y,
                                    x + start.width,
                                    y + start.height,
                                    frame))
        x, y = (forwards[frame][1][x // skip, y // skip] * skip,
                forwards[frame][2][x // skip, y // skip] * skip)
        frame = frame - 1
    path.append(start)
    path.reverse()
    pathdict = dict((x.frame, x) for x in path)

    # calculating error
    # if there is a pool, this will use up to 2 cores
    if False and pool:
        logger.info("Calculating forward and backwards errors")
        forwarderror  = pool.apply_async(calcerror, (pathdict,
                                                     forwards,
                                                     erroroverlap,
                                                     skip,
                                                     frames))
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  erroroverlap,
                                  skip,
                                  list(reversed(frames)))
        forwarderror  = forwarderror.get() # blocks until forwards is done
    else:
        logger.info("Calculating forward error")
        forwarderror  = calcerror(pathdict,
                                  forwards,
                                  erroroverlap,
                                  skip,
                                  frames)
        logger.info("Calculating backward error")
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  erroroverlap,
                                  skip,
                                  list(reversed(frames)))

    # score marginals on the frames
    # if there is a pool, this will happen in parallel
    logger.info("Computing marginals")
    orders = [(forwards[x], backwards[x],
               forwarderror[x], backwarderror[x],
               costs[x], sigma, model.dim, start, skip, clickradius, x)
               for x in frames[1:-1]]
    marginals = mapper(scoremarginals, orders)
    marginals = [(0, start.frame)] + marginals + [(0, stopframe)]

    return marginals, path

cdef double calcerroroverlap(int i, int j, annotations.Box box, double thres,
                             int skip):
    boxw = (box.xbr - box.xtl) // skip
    boxh = (box.ybr - box.ytl) // skip

    xdiff = min(i + boxw, box.xbr // skip) - max(i, box.xtl // skip) 
    ydiff = min(j + boxh, box.ybr // skip) - max(j, box.ytl // skip) 

    if xdiff <= 0 or ydiff <= 0:
        error = 1.
    else:
        uni = boxw * boxh * 2 - xdiff * ydiff
        error = xdiff * ydiff / float(uni)
        if error >= thres:
            error = 0.
        else:
            error = 1.
    return error

def calcerror(pathdict, pointers, double erroroverlap, int skip, frames):
    """
    Calculates the error going through a path at a certain point.

    We use dynamic programming to calculate error here in order to speed up
    the error calculation. We simply use the pairwise results from the
    previous results.
    """
    cdef int frame = frames[0]
    cdef annotations.Box box = pathdict[frame]
    cdef numpy.ndarray[numpy.double_t, ndim=2] previous, current, local
    cdef numpy.ndarray[numpy.int_t, ndim=2] xpointer, ypointer
    size = pointers[frames[-1]][1].shape
    cdef int w = size[0], h = size[1]
    cdef int xp, yp
    cdef double message

    graph = {}

    # setup the base case
    previous = numpy.zeros(size, dtype = numpy.double)
    for i in range(w):
        for j in range(h):
            previous[i, j] = calcerroroverlap(i, j, box, erroroverlap, skip) 
    graph[frame] = previous, previous

    # do the inductive steps
    for frame in frames[1:]:
        current = numpy.zeros(size, dtype = numpy.double)
        local = numpy.zeros(size, dtype = numpy.double)
        box = pathdict[frame]
        xpointer = pointers[frame][1]
        ypointer = pointers[frame][2]

        for i in range(w):
            for j in range(h):
                error = calcerroroverlap(i, j, box, erroroverlap, skip)
                xp = xpointer[i, j]
                yp = ypointer[i, j]
                message = previous[xp, yp]
                local[i, j] = error
                current[i, j] = error + message
        graph[frame] = current, local
        previous = current
    return graph

def scoremarginals(workorder):
    cdef double sigma
    cdef int frame, radius
    cdef annotations.Box start
    (forw, backw, forwerr, backwerr, costs, sigma, dim, start, skip, radius, frame) = workorder

    cdef double score = 0, normalizer = 0, matchscore, error, localscore

    cdef numpy.ndarray[numpy.double_t, ndim=2] forwrt = forw[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] backwrt = backw[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] costrt = costs
    cdef numpy.ndarray[numpy.double_t, ndim=2] forwe = forwerr[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] backwe = backwerr[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] forwloce = forwerr[1]

    cdef int w = forw[1].shape[0]
    cdef int h = forw[1].shape[1]

    cdef double wr = dim[0] / (<double>start.width)
    cdef double hr = dim[1] / (<double>start.height)

    cdef numpy.ndarray[numpy.double_t, ndim=2] gprob = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] greduct = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] gerrors = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] gmargin = numpy.zeros((w, h))

    cdef numpy.ndarray[numpy.double_t, ndim=2] matchscores
    matchscores = numpy.zeros((w, h))

    cdef numpy.ndarray[numpy.double_t, ndim=2] errors, errorsvert
    errors = numpy.zeros((w, h))
    errorsvert = numpy.zeros((w, h))

    cdef double maxmatchscore = Infinity

    # for numerical reasons, we want to subtract the most best score
    for i in range(w):
        for j in range(h):
            matchscore = forwrt[i, j] + backwrt[i, j]
            matchscore = matchscore - costrt[<int>(i * wr), <int>(j * hr)]
            matchscores[i, j] = matchscore
            if matchscore < maxmatchscore:
                maxmatchscore = matchscore

    # compute error image
    for i in range(w):
        for j in range(h):
            errors[i, j] = forwe[i, j] + backwe[i, j] - forwloce[i, j]

    # vertical pass on min
    for i in range(w):
        for j in range(h):
            errorsvert[i, j] = errors[i, j]
            for y in range(max(0, j - radius), min(h - 1, j + radius)):
                errorsvert[i, j] = min(errorsvert[i, j], errors[i, y])

    # horizontal pass on min
    for i in range(w):
        for j in range(h):
            errors[i, j] = errorsvert[i, j]
            for x in range(max(0, i - radius), min(w - 1, i + radius)):
                errors[i, j] = min(errors[i, j], errorsvert[x, j])
            
    for i in range(w):
        for j in range(h):
            matchscore = matchscores[i, j] - maxmatchscore

            gmargin[i, j] = matchscore

            matchscore = exp(-matchscore / sigma)

            localscore = matchscore * errors[i, j]

            gprob[i, j] = matchscore
            greduct[i, j] = localscore
            gerrors[i, j] = errors[i, j]

            score += localscore
            normalizer += matchscore

    if debug:
        gmargin = -gmargin
        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(gmargin.min(), gmargin.max()))
        pylab.imshow(gmargin.transpose())
        pylab.savefig("tmp/margin{0}.png".format(frame))
        pylab.clf()

        pylab.set_cmap("gray")
        gprob = gprob / normalizer
        pylab.title("min = {0}, max = {1}".format(gprob.min(), gprob.max()))
        pylab.imshow(gprob.transpose())
        pylab.savefig("tmp/prob{0}.png".format(frame))
        pylab.clf()

        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(greduct.min(), greduct.max()))
        pylab.imshow(greduct.transpose())
        pylab.savefig("tmp/reduct{0}.png".format(frame))
        pylab.clf()

        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(gerrors.min(), gerrors.max()))
        pylab.imshow(gerrors.transpose())
        pylab.savefig("tmp/errors{0}.png".format(frame))
        pylab.clf()

    return score / normalizer, frame

