import numpy
from vision import convolution
from vision import Box
from vision.track import pairwise
from vision import annotations
from vision.model import PathModel

from math import ceil

import logging

cimport numpy
from vision cimport annotations

cdef int debug = 0

if debug:
    import pylab

logger = logging.getLogger("vision.track.dp")

def fill(givens, images, last = None, 
         pairwisecost = 0.001, upperthreshold = 10, lowerthreshold = -100,
         skip = 3, rgbbin = 8, hogbin = 8, c = 1, realprior = None, pool = None):

    givens.sort(key = lambda x: x.frame)

    model = PathModel(images, givens, rgbbin = rgbbin, hogbin = hogbin, c = c,
                      realprior = realprior)
    
    fullpath = []
    for x, y in zip(givens, givens[1:]):
        path = track(x, y, model, images, pairwisecost,
                    upperthreshold, lowerthreshold, skip, pool)
        fullpath.extend(path[:-1])

    if last is not None and last > givens[-1].frame:
        path = track(givens[-1], last, model, images,
                     pairwisecost, upperthreshold, lowerthreshold, skip, pool)
        fullpath.extend(path[:-1])

    return fullpath

def track(start, stop, model, images,
          pairwisecost = 0.001, upperthreshold = 10, lowerthreshold = -100,
          skip = 3, pool = None):

    imagesize = images[start.frame].size

    if pool:
        mapper = pool.map
    else:
        mapper = map

    try:
        stopframe = stop.frame
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
    except:
        stopframe = stop
        constrained = False
        constraints = [start]

    logger.info("Dynamic programming from {0} to {1}".format(start.frame,
                                                             stopframe))

    frames = range(start.frame, stopframe + 1)

    # build dictionary of local scores
    # if there is a pool, this will happen in parallel
    logger.info("Scoring frames")
    orders = [(images, start, x, model) for x in frames]
    costs = dict(mapper(scoreframe, orders))

    # forward and backwards passes
    # if there is a pool, this will use up to 2 cores
    forwardsargs  = [frames, imagesize, model, costs, pairwisecost,
                     upperthreshold, lowerthreshold, skip, constraints]
    logger.info("Building forwards graph")
    forwards  = buildgraph(*forwardsargs)
    
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
    return path

def buildgraph(frames, imagesize, model, costs,
               double pairwisecost, double upperthreshold, 
               double lowerthreshold, int skip, constraints):

    cdef double cost, wr, hr
    cdef int width, height, usablewidth, usableheight
    cdef numpy.ndarray[numpy.double_t, ndim=2] relevantcosts
    cdef numpy.ndarray[numpy.double_t, ndim=2] current
    cdef numpy.ndarray[numpy.int_t, ndim=2] xpointer, ypointer
    cdef annotations.Box constraint
    cdef annotations.Box start = constraints[0]

    cdef double Huge = 1e200

    width, height = imagesize
    wr = model.dim[0] / (<double>start.width)
    hr = model.dim[1] / (<double>start.height)
    usablewidth = <int>ceil((width - start.width) / <double>(skip))
    usableheight = <int>ceil((height - start.height) / <double>(skip))

    graph = {}

    # walk along linear path
    for frame in frames:
        if frame == frames[0]:
            current = numpy.zeros((usablewidth, usableheight),
                                   dtype = numpy.double)
            xpointer = None
            ypointer = None
        else:
            current, xpointer, ypointer = pairwise.quadratic(current,
                                                             pairwisecost)

        for constraint in constraints:
            if constraint.frame == frame:
                current = numpy.ones((usablewidth, usableheight),
                                      dtype = numpy.double)
                current = current * Huge
                #print "image", width, height
                #print "usable", usablewidth, usableheight
                #print "constraint", constraint.xtl, constraint.ytl, constraint
                #print "start", start.width, start.height, str(start)
                #print "ratio", wr, hr
                #print "skip", skip
                current[constraint.xtl // skip, constraint.ytl // skip] = 0
                break
        else:
            relevantcosts = costs[frame]

            for x in range(0, usablewidth):
                for y in range(0, usableheight):
                    cost  = relevantcosts[<int>(x*wr*skip), <int>(y*hr*skip)]
                    cost  = min(cost, upperthreshold)
                    cost  = max(cost, lowerthreshold)
                    current[x, y] += cost

        graph[frame] = current, xpointer, ypointer
    return graph

def scoreframe(workorder):
    """
    Convolves a learned weight vector against an image. This method
    should take a workorder tuple because it can be used in multiprocessing.
    """
    images, start, frame, model = workorder

    logger.debug("Scoring frame {0}".format(frame))

    cost = model.scoreframe(images[frame], start.size, frame)

    if debug:
        pylab.set_cmap("gray")
        pylab.title("min = {0}, max = {1}".format(cost.min(), cost.max()))
        pylab.imshow(cost.transpose())
        pylab.savefig("tmp/cost{0}.png".format(frame))
        pylab.clf()

    return frame, cost
