from vision import convolution
from vision.track import interpolation
import logging
import numpy

cimport numpy

from vision cimport annotations

import pylab

logger = logging.getLogger("vision.alearn.marginals")

cdef extern from "math.h":
    float exp(float n)

def pick(start, stop, model, images,
         pairwisecost = 0.1, lineardeviation = 0.0,
         upperthreshold = 10, sigma = .1, pool = None):

    mapper = pool.map if pool else map
    linearpath = interpolation.Linear(start, stop)

    # build dictionary of local scores
    # if there is a pool, this will happen in parallel
    logger.info("Scoring frames")
    orders = [(images, x, model) for x in linearpath]
    costs = dict(mapper(scoreframe, orders))

    # forward and backwards passes
    # if there is a pool, this will use up to 2 cores
    logger.info("Building forwards and backwards graphs")
    forwardsargs  = [linearpath, images, model, costs, pairwisecost,
                     upperthreshold, lineardeviation]
    backwardsargs = list(forwardsargs)
    backwardsargs[0] = list(reversed(linearpath))
    if pool:
        forwards  = pool.apply_async(buildgraph, forwardsargs)
        backwards = buildgraph(*backwardsargs)
        forwards  = forwards.get() # blocks until forwards graph done
    else:
        forwards  = buildgraph(*forwardsargs)
        backwards = buildgraph(*backwardsargs)
    
    # backtrack
    logger.info("Backtracking")
    x, y, frame = stop.xtl, stop.ytl, stop.frame
    path = []
    while frame > start.frame:
        path.append(annotations.Box(x, y,
                                    x + stop.width,
                                    y + stop.height,
                                    frame))
        x, y = forwards[frame][1][x, y], forwards[frame][2][x, y]
        frame = frame - 1
    path.append(start)
    path.reverse()
    pathdict = dict((x.frame, x) for x in path)

    # calculating error
    # if there is a pool, this will use up to 2 cores
    logger.info("Calculating errors from the predicted path")
    if pool:
        forwarderror  = pool.apply_async(calcerror, (pathdict,
                                                     forwards,
                                                     linearpath))
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  list(reversed(linearpath)))
        forwarderror  = forwarderror.get() # blocks until forwards is done
    else:
        forwarderror  = calcerror(pathdict,
                                  forwards,
                                  linearpath)
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  list(reversed(linearpath)))

    # score marginals on the frames
    # if there is a pool, this will happen in parallel
    logger.info("Computing marginals")
    orders = [(forwards[x.frame], backwards[x.frame],
               forwarderror[x.frame], backwarderror[x.frame],
               costs[x.frame], sigma, model.dim, x) for x in linearpath[1:-1]]
    marginals = mapper(scoremarginals, orders)

    pylab.close()
    pylab.plot([x[1] for x in marginals], [x[0] for x in marginals])
    pylab.grid()
    pylab.savefig("tmp/scoreplot.png")

    # find minimum cost
    best = max(marginals)
    return best[1], best[0], path

def calcerror(pathdict, pointers, linearpath):
    """
    Calculates the error going through a path at a certain point.

    We use dynamic programming to calculate error here in order to speed up
    the error calculation. We simply use the pairwise results from the
    previous results.
    """
    cdef int frame = linearpath[0].frame
    cdef annotations.Box box = pathdict[frame]
    cdef numpy.ndarray[numpy.double_t, ndim=2] previous, current, local
    cdef numpy.ndarray[numpy.int_t, ndim=2] xpointer, ypointer
    size = pointers[linearpath[-1].frame][1].shape
    cdef int w = size[0], h = size[1]
    cdef int xp, yp
    cdef double message

    graph = {}

    # setup the base case
    previous = numpy.zeros(size, dtype = numpy.double)
    for i in range(w):
        for j in range(h):
            previous[i, j] = (i - box.xtl) ** 2 + (j - box.ytl) ** 2
    graph[frame] = previous, previous

    # do the inductive steps
    for linearbox in linearpath[1:]:
        current = numpy.zeros(size, dtype = numpy.double)
        local = numpy.zeros(size, dtype = numpy.double)
        frame = linearbox.frame
        box = pathdict[frame]
        xpointer = pointers[frame][1]
        ypointer = pointers[frame][2]

        for i in range(w):
            for j in range(h):
                error = (i - box.xtl) ** 2 + (j - box.ytl) ** 2
                xp = xpointer[i, j]
                yp = ypointer[i, j]
                message = previous[xp, yp]
                local[i, j] = error
                current[i, j] = error + message
        graph[frame] = current, local
        previous = current

        pylab.set_cmap("gray")
        pylab.imshow(current.transpose())
        pylab.title("min = {0}, max = {1}".format(current.min(), current.max()))
        pylab.savefig("tmp/error{0}-{1}.png".format(linearpath[0].frame,
                                                    linearbox.frame))
        pylab.clf()
    return graph

def scoremarginals(workorder):
    cdef double sigma
    (forw, backw, forwerr, backwerr, costs, sigma, dim, linearbox) = workorder

    cdef int frame = linearbox.frame
    cdef double score = 0, normalizer = 0, matchscore, error, localscore

    cdef numpy.ndarray[numpy.double_t, ndim=2] forwrt = forw[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] backwrt = backw[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] costrt = costs
    cdef numpy.ndarray[numpy.double_t, ndim=2] forwe = forwerr[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] backwe = backwerr[0]
    cdef numpy.ndarray[numpy.double_t, ndim=2] forwloce = forwerr[1]

    cdef int w = forw[1].shape[0]
    cdef int h = forw[1].shape[1]

    cdef double wr = dim[0] / (<double>linearbox.width)
    cdef double hr = dim[1] / (<double>linearbox.height)

    cdef numpy.ndarray[numpy.double_t, ndim=2] prob = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] reduct = numpy.zeros((w, h))

    for i in range(w):
        for j in range(h):
            matchscore = forwrt[i, j] + backwrt[i, j]
            matchscore = matchscore - costrt[<int>(i * wr), <int>(j * hr)]
            matchscore = exp(-matchscore / sigma)

            error = forwe[i, j] + backwe[i, j] - forwloce[i, j]

            localscore = matchscore * error

            prob[i, j] = matchscore
            reduct[i, j] = localscore

            score += localscore
            normalizer += matchscore

    pylab.set_cmap("gray")
    pylab.title("min = {0}, max = {1}".format(prob.min(), prob.max()))
    pylab.imshow(prob.transpose())
    pylab.savefig("tmp/prob{0}.png".format(linearbox.frame))
    pylab.clf()

    pylab.set_cmap("gray")
    pylab.title("min = {0}, max = {1}".format(reduct.min(), reduct.max()))
    pylab.imshow(reduct.transpose())
    pylab.savefig("tmp/reduct{0}.png".format(linearbox.frame))
    pylab.clf()

    return score / normalizer, linearbox.frame

def buildgraph(linearpath, images, model, costs,
               double pairwisecost, double upperthreshold,
               double lineardeviation):

    cdef double cost, wr, hr
    cdef int width, height, usablewidth, usableheight
    cdef numpy.ndarray[numpy.double_t, ndim=2] relevantcosts
    cdef numpy.ndarray[numpy.double_t, ndim=2] current
    cdef numpy.ndarray[numpy.int_t, ndim=2] xpointer, ypointer
    cdef annotations.Box linearbox, start = linearpath[0]

    width, height = images[0].size

    usablewidth = width - start.width
    usableheight = height - start.height

    current = numpy.ones((usablewidth, usableheight), dtype = numpy.double)
    current = current * float("infinity")
    current[<int>start.xtl, <int>start.ytl] = 0

    graph = {}
    graph[start.frame] = current, None, None

    # walk along linear path
    for linearbox in linearpath[1:]:
        current, xpointer, ypointer = pairwise_manhattan(current, pairwisecost)
        #current, xpointer, ypointer = pairwise_quadratic(current, pairwisecost)

#        pylab.imshow(current.transpose())
#        pylab.savefig("tmp/pairwise{0}.png".format(linearbox.frame))
#        pylab.clf()

        wr = model.dim[0] / (<double>linearbox.width)
        hr = model.dim[1] / (<double>linearbox.height)

        usablewidth = width - linearbox.width
        usableheight = height - linearbox.height

        relevantcosts = costs[linearbox.frame]

        for x in range(0, usablewidth):
            for y in range(0, usableheight):
                cost  = relevantcosts[<int>(x*wr), <int>(y*hr)]
                cost  = min(cost, upperthreshold)
                cost += (lineardeviation * 
                            (x - linearbox.xtl) * 
                            (x - linearbox.xtl))
                cost += (lineardeviation * 
                            (y - linearbox.ytl) * 
                            (y - linearbox.ytl))
                current[x, y] += cost

        graph[linearbox.frame] = current, xpointer, ypointer
    return graph

cdef double Infinity = float("infinity")

# see Pedro Felzenszwalb et. al
cdef pairwise_quadratic_1d(numpy.ndarray[numpy.double_t, ndim=1] src,
                           numpy.ndarray[numpy.double_t, ndim=1] dst,
                           numpy.ndarray[numpy.double_t, ndim=1] ptr,
                           int step, double a, double b):
    cdef int n = src.shape[0]
    cdef numpy.ndarray[numpy.int_t, ndim=1] v = numpy.zeros(n, dtype=numpy.int)
    cdef numpy.ndarray[numpy.double_t, ndim=1] z = numpy.zeros(n+1)
    cdef int k = 0
    cdef int q
    cdef double s

    v[0] = 0
    z[0] = -Infinity
    z[1] = Infinity

    for q in range(1, n): 
        s = (src[q*step]-src[v[k]*step])-b*(q-v[k])+a*(q*q-v[k]*v[k])
        s = s / (2*a*(q-v[k]))

        while s <= z[k]:
            k = k - 1
            s = (src[q*step]-src[v[k]*step])-b*(q-v[k])+a*(q*q-v[k]*v[k])
            s = s / (2*a*(q-v[k]))
        k = k + 1
        v[k] = q
        z[k] = s
        z[k+1] = Infinity

    k = 0
    for q in range(0, n): 
        while z[k+1] < q:
            k = k + 1
        dst[q*step] = a*(q-v[k])*(q-v[k]) + b*(q-v[k]) + src[v[k]*step]
        ptr[q*step] = v[k]
    return dst, ptr

## see Pedro Felzenszwalb et. al
#def pairwise_quadratic(inscores, double cost):
#    cdef int w, h, i, j, ri, rj
#    w, h = inscores.shape
#
#    cdef numpy.ndarray[numpy.double_t, ndim=1] src = scores.flatten()
#    cdef numpy.ndarray[numpy.double_t, ndim=1] dst, ptr
#    dst = numpy.zeros(scores.shape, dtype = numpy.double)
#    ptr = numpy.zeros(scores.shape, dtype = numpy.double)
#
#    # transform along columns
#    for i in range(w):
#        pairwise_quadratic_1d(
#        cols[i, :], colsp[i, :] = pairwise_quadratic_1d(scores[i, :], cost)
#
#    # transform along rows
#    for j in range(h):
#        rows[:, j], rowsp[:, j] = pairwise_quadratic_1d(cols[:, j], cost)
#
#    return rows, rowsp, colsp

def pairwise_manhattan(inscores, incost):
    cdef int w, h, i, j, ri, rj
    w, h = inscores.shape

    cdef double cost = incost
    cdef numpy.ndarray[numpy.double_t, ndim=2] scores = inscores

    cdef numpy.ndarray[numpy.double_t, ndim=2] forward, backward
    cdef numpy.ndarray[numpy.int_t, ndim=2] forwardxp, forwardyp
    cdef numpy.ndarray[numpy.int_t, ndim=2] backwardxp, backwardyp

    forward = numpy.zeros((w,h), dtype = numpy.double)
    forwardxp = numpy.zeros((w,h), dtype = numpy.int)
    forwardyp = numpy.zeros((w,h), dtype = numpy.int)
    backward = numpy.zeros((w,h), dtype = numpy.double)
    backwardxp = numpy.zeros((w,h), dtype = numpy.int)
    backwardyp = numpy.zeros((w,h), dtype = numpy.int)

    # forward pass
    for i in range(w):
        for j in range(h):
            forward[i, j] = scores[i, j]
            forwardxp[i, j] = i
            forwardyp[i, j] = j

            if j-1 >= 0 and forward[i, j-1] + cost < forward[i, j]:
                forward[i, j] = forward[i, j-1] + cost
                forwardxp[i, j] = forwardxp[i, j-1]
                forwardyp[i, j] = forwardyp[i, j-1]
            if i-1 >= 0 and forward[i-1, j] + cost < forward[i, j]:
                forward[i, j] = forward[i-1, j] + cost
                forwardxp[i, j] = forwardxp[i-1, j]
                forwardyp[i, j] = forwardyp[i-1, j]

    # backwards pass
    for ri in range(w):
        i = w - ri - 1
        for rj in range(h):
            j = h - rj - 1
            backward[i, j] = forward[i, j]
            backwardxp[i, j] = forwardxp[i, j]
            backwardyp[i, j] = forwardyp[i, j]

            if j+1 < h and backward[i, j+1] + cost < backward[i, j]:
                backward[i, j] = backward[i, j+1] + cost
                backwardxp[i, j] = backwardxp[i, j+1]
                backwardyp[i, j] = backwardyp[i, j+1]
            if i+1 < w and backward[i+1, j] + cost < backward[i, j]:
                backward[i, j] = backward[i+1, j] + cost
                backwardxp[i, j] = backwardxp[i+1, j]
                backwardyp[i, j] = backwardyp[i+1, j]

    return backward, backwardxp, backwardyp

def scoreframe(workorder):
    """
    Convolves a learned weight vector against an image. This method
    should take a workorder tuple because it can be used in multiprocessing.
    """
    images, box, model = workorder

    logger.debug("Scoring frame {0}".format(box.frame))

    # resize image to so box has 'dim' in the resized space
    image = images[box.frame]
    width, height = image.size
    wr = model.dim[0] / float(box.width)
    hr = model.dim[1] / float(box.height)
    rimage = image.resize((int(width * wr), int(height * hr)), 2)

    cost = convolution.hogrgb(rimage, model.dim,
                              model.hogweights(),
                              model.rgbweights())

    return box.frame, cost
