from vision import convolution
from vision.track import interpolation
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

def pick(start, stop, model, images,
         pairwisecost = 0.001, lineardeviation = 0.0,
         upperthreshold = 10, sigma = .1, erroroverlap = 0.5, pool = None):

    if pool:
        logger.info("Found a process pool, so attempting to parallelize")
        mapper = pool.map
    else:
        logger.info("No process pool found, so remaining single threaded")
        mapper = map
    linearpath = interpolation.Linear(start, stop)
    imagesize = images[0].size

    # build dictionary of local scores
    # if there is a pool, this will happen in parallel
    logger.info("Scoring frames")
    orders = [(images, x, model) for x in linearpath]
    costs = dict(mapper(scoreframe, orders))

    # forward and backwards passes
    # if there is a pool, this will use up to 2 cores
    forwardsargs  = [linearpath, imagesize, model, costs, pairwisecost,
                     upperthreshold, lineardeviation]
    backwardsargs = list(forwardsargs)
    backwardsargs[0] = list(reversed(linearpath))
    if pool:
        logger.info("Building forwards and backwards graphs")
        forwards  = pool.apply_async(buildgraph, forwardsargs)
        backwards = buildgraph(*backwardsargs)
        forwards  = forwards.get() # blocks until forwards graph done
    else:
        logger.info("Building forwards graph")
        forwards  = buildgraph(*forwardsargs)
        logger.info("Building backwards graph")
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
    if pool:
        logger.info("Calculating forward and backwards errors")
        forwarderror  = pool.apply_async(calcerror, (pathdict,
                                                     forwards,
                                                     erroroverlap,
                                                     linearpath))
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  erroroverlap,
                                  list(reversed(linearpath)))
        forwarderror  = forwarderror.get() # blocks until forwards is done
    else:
        logger.info("Calculating forward error")
        forwarderror  = calcerror(pathdict,
                                  forwards,
                                  erroroverlap,
                                  linearpath)
        logger.info("Calculating backward error")
        backwarderror = calcerror(pathdict,
                                  backwards,
                                  erroroverlap,
                                  list(reversed(linearpath)))

    # score marginals on the frames
    # if there is a pool, this will happen in parallel
    logger.info("Computing marginals")
    orders = [(forwards[x.frame], backwards[x.frame],
               forwarderror[x.frame], backwarderror[x.frame],
               costs[x.frame], sigma, model.dim, x) for x in linearpath[1:-1]]
    marginals = mapper(scoremarginals, orders)

    if debug:
        pylab.close()
        pylab.plot([x[1] for x in marginals], [x[0] for x in marginals])
        pylab.grid()
        pylab.savefig("tmp/scoreplot.png")

    # find minimum cost
    best = max(marginals)
    return best[1], best[0], path

def calcerror(pathdict, pointers, double erroroverlap, linearpath):
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

        boxw = box.xbr - box.xtl
        boxh = box.ybr - box.ytl

        for i in range(w):
            for j in range(h):
                #error = (i - box.xtl) ** 2 + (j - box.ytl) ** 2

                xdiff = min(i + boxw, box.xbr) - max(i, box.xtl) 
                ydiff = min(j + boxh, box.ybr) - max(j, box.ytl) 

                if xdiff <= 0 or ydiff <= 0:
                    error = 1.
                else:
                    uni = boxw * boxh * 2 - xdiff * ydiff
                    error = xdiff * ydiff / float(uni)
                    if error >= erroroverlap:
                        error = 0.
                    else:
                        error = 1.

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

    cdef numpy.ndarray[numpy.double_t, ndim=2] gprob = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] greduct = numpy.zeros((w, h))
    cdef numpy.ndarray[numpy.double_t, ndim=2] gerrors = numpy.zeros((w, h))

    cdef numpy.ndarray[numpy.double_t, ndim=2] matchscores
    matchscores = numpy.zeros((w, h))

    cdef numpy.ndarray[numpy.double_t, ndim=2] errors, errorsvert
    errors = numpy.zeros((w, h))
    errorsvert = numpy.zeros((w, h))

    cdef double maxmatchscore = Infinity

    cdef int radius = 10

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
            matchscore = exp(-matchscore / sigma)

            localscore = matchscore * errors[i, j]

            gprob[i, j] = matchscore
            greduct[i, j] = localscore
            gerrors[i, j] = errors[i, j]

            score += localscore
            normalizer += matchscore

    if debug:
        pylab.set_cmap("gray")
        gprob = gprob / normalizer
        pylab.title("min = {0}, max = {1}".format(gprob.min(), gprob.max()))
        pylab.imshow(gprob.transpose())
        pylab.savefig("tmp/prob{0}.png".format(linearbox.frame))
        pylab.clf()

        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(greduct.min(), greduct.max()))
        pylab.imshow(greduct.transpose())
        pylab.savefig("tmp/reduct{0}.png".format(linearbox.frame))
        pylab.clf()

        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(gerrors.min(), gerrors.max()))
        pylab.imshow(gerrors.transpose())
        pylab.savefig("tmp/errors{0}.png".format(linearbox.frame))
        pylab.clf()

    return score / normalizer, linearbox.frame

def buildgraph(linearpath, imagesize, model, costs,
               double pairwisecost, double upperthreshold,
               double lineardeviation):

    cdef double cost, wr, hr
    cdef int width, height, usablewidth, usableheight
    cdef numpy.ndarray[numpy.double_t, ndim=2] relevantcosts
    cdef numpy.ndarray[numpy.double_t, ndim=2] current
    cdef numpy.ndarray[numpy.int_t, ndim=2] xpointer, ypointer
    cdef annotations.Box linearbox, start = linearpath[0]

    cdef double Huge = 1e200

    width, height = imagesize

    usablewidth = width - start.width
    usableheight = height - start.height

    current = numpy.ones((usablewidth, usableheight), dtype = numpy.double)
    current = current * Huge
    current[<int>start.xtl, <int>start.ytl] = 0

    graph = {}
    graph[start.frame] = current, None, None

    # walk along linear path
    for linearbox in linearpath[1:]:
        wr = model.dim[0] / (<double>linearbox.width)
        hr = model.dim[1] / (<double>linearbox.height)

        usablewidth = width - linearbox.width
        usableheight = height - linearbox.height

        relevantcosts = costs[linearbox.frame]

        # we need to resize current in order to handle resizing paths
        # if we enlarge, we just pad with psuedo-infinity
        # otherwise, we can just truncate
        #current = numpy.resize(current, (usablewidth, usableheight))
        #for x in range(current.shape[0], usablewidth):
        #    for y in range(usableheight):
        #        current[x, y] = Huge
        #for y in range(current.shape[1], usableheight):
        #    for x in range(usablewidth):
        #        current[x, y] = Huge
         
        current, xpointer, ypointer = pairwise_quadratic(current, pairwisecost)
        #current, xpointer, ypointer = pairwise_hinge(current)

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

# see Pedro Felzenszwalb et. al
cpdef pairwise_quadratic_1d(numpy.ndarray[numpy.double_t, ndim=1] src,
                            numpy.ndarray[numpy.double_t, ndim=1] dst,
                            numpy.ndarray[numpy.int_t, ndim=1] ptr,
                            int step, int n, double a, double b, int o):
    
    cdef numpy.ndarray[numpy.int_t, ndim=1] v = numpy.zeros(n, dtype=numpy.int)
    cdef numpy.ndarray[numpy.double_t, ndim=1] z = numpy.zeros(n+1,
                                                   dtype = numpy.double)
    cdef int k = 0
    v[0] = 0
    z[0] = -Infinity
    z[1] = Infinity

    cdef int q
    cdef double s

    for q in range(1, n):
        s = ((src[q*step+o]-src[v[k]*step+o])-b*(q-v[k])+a*(q**2-v[k]**2))
        s = s / (2*a*(q-v[k]))

        while s <= z[k]:
            k = k - 1
            s = ((src[q*step+o]-src[v[k]*step+o])-b*(q-v[k])+a*(q**2-v[k]**2))
            s = s / (2*a*(q-v[k]))

        k = k + 1
        v[k] = q
        z[k] = s
        z[k+1] = Infinity

    k = 0
    for q in range(0, n):
        while z[k+1] < q:
            k = k + 1
        dst[q*step+o] = a*(q-v[k])**2 + b*(q-v[k]) + src[v[k]*step+o]
        ptr[q*step+o] = v[k]

# see Pedro Felzenszwalb et. al
def pairwise_quadratic(numpy.ndarray[numpy.double_t, ndim=2] scores,
                       double cost):
    cdef int w, h, x, y, p
    w = scores.shape[0]
    h = scores.shape[1]

    cdef numpy.ndarray[numpy.double_t, ndim=1] vals = scores.flatten()

    cdef numpy.ndarray[numpy.double_t, ndim=1] M = numpy.zeros(w*h, 
                                                   dtype = numpy.double)
    cdef numpy.ndarray[numpy.int_t, ndim=2] Ix = numpy.zeros((w, h), 
                                                 dtype = numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=2] Iy = numpy.zeros((w, h), 
                                                 dtype = numpy.int)

    cdef numpy.ndarray[numpy.double_t, ndim=1] tmpM = numpy.zeros(w*h, 
                                                      dtype = numpy.double)
    cdef numpy.ndarray[numpy.int_t, ndim=1] tmpIx = numpy.zeros(w*h, 
                                                    dtype = numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=1] tmpIy = numpy.zeros(w*h, 
                                                    dtype = numpy.int)

    for x in range(w):
        pairwise_quadratic_1d(vals, tmpM, tmpIy, 1, h, cost, 0, x * h)

    for y in range(h):
        pairwise_quadratic_1d(tmpM, M, tmpIx, h, w, cost, 0, y)

    for x in range(w):
        for y in range(h):
            p = x * h + y
            Ix[x, y] = tmpIx[p]
            Iy[x, y] = tmpIy[tmpIx[p]*h+y]

    return M.reshape((w,h)), Ix, Iy

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

def pairwise_hinge(inscores, int radius = 30):
    cdef int w, h, i, j, pstart, pstop, x, y 
    w, h = inscores.shape

    cdef numpy.ndarray[numpy.double_t, ndim=2] scores = inscores
    cdef numpy.ndarray[numpy.double_t, ndim=2] vertical, horizontal
    cdef numpy.ndarray[numpy.int_t, ndim=2] xp, yp

    vertical = numpy.zeros((w,h), dtype = numpy.double)
    horizontal = numpy.zeros((w,h), dtype = numpy.double)
    xp = numpy.zeros((w,h), dtype = numpy.int)
    yp = numpy.zeros((w,h), dtype = numpy.int)

    for i in range(w):
        for j in range(h):
            pstart = j - radius
            pstop  = j + radius 
            if pstart < 0:
                pstart = 0
            if pstop > h:
                pstop = h
            vertical[i, j] = scores[i, j]
            yp[i, j] = j
            for y in range(pstart, pstop): 
                if scores[i, y] < vertical[i, j]:
                    vertical[i, j] = scores[i, y]
                    yp[i, j] = y

    for i in range(w):
        for j in range(h):
            pstart = i - radius
            pstop  = i + radius
            if pstart < 0:
                pstart = 0
            if pstop > w:
                pstop = w
            horizontal[i, j] = vertical[i, j]
            xp[i, j] = i
            for x in range(pstart, pstop):
                if vertical[x, j] < horizontal[i, j]:
                    horizontal[i, j] = vertical[x, j]
                    xp[i, j] = x

    return horizontal, xp, yp

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

    if debug:
        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(cost.min(), cost.max()))
        pylab.imshow(cost.transpose())
        pylab.savefig("tmp/cost{0}.png".format(box.frame))
        pylab.clf()

    return box.frame, cost
