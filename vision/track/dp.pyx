from vision import annotations
from vision import convolution
import interpolation 
import logging
import numpy
import Image

cimport numpy
cimport cython
from vision cimport annotations

log = logging.getLogger("track")

cpdef track(annotations.Box start, annotations.Box stop, svm, images,
    int slidingskip          = 3,    int slidingsearchwidth   = 300,
    int pairwiseradius       = 10,   float resizestart        = 1.0,
    float resizestop         = 1.1,  float resizeincrement    = 0.2,
    float lineardeviation    = 0.01, float upperthreshold     = 10,
    dim = (40, 40)):
    """
    Performs dynamic programming via the Viterbi algorithm. Finds the globally
    optimal path from start to stop using the SVM across the frames.

    - slidingskip dictates how many pixels to move the sliding window.
    - slidingsearchwidth dictates how many pixels to deviate from the linear
      path
    - pairwiseradius is the acceptable radius to move between frames
    - search over resizes resizestart <= r < resizestop, skipping by
      resizeincrement
    - lineardeviation is proportionality constant for deviating from linear path
    - upperthreshold is the maximum score
    - dim is the size of the template to score with (probably should be removed)
    """

    cdef NodeMatrix pltable, cltable, pairwiseterms
    cdef annotations.Box slidingbox, resizedbox, linearbox
    cdef int width, height, usedwindows
    cdef double cost, wr, hr, velocity, progress
    cdef numpy.ndarray[numpy.double_t, ndim=2] costim

    linearpath = interpolation.Linear(start, stop)
    width, height = images[start.frame].size

    velocity = linearpath[0].get_distance(linearpath[1])
    if velocity >= pairwiseradius:
        pairwiseradius = int(velocity * 1.5)
        logging.warning("Adjusting pairwise radius")

    root = Node(start)
    pltable = NodeMatrix(width, height, slidingskip)
    pltable.set(start.xtl, start.ytl, root)

    for linearbox in linearpath[1:]:

        cltable = NodeMatrix(width, height, slidingskip)
        pairwiseterms = pltable.pairwise_build(pairwiseradius)
        im = images[linearbox.frame]
        progress = ((linearbox.frame - start.frame) /
                    float(stop.frame - start.frame))

        log.info("Scoring frame {0} ({1}%)".format(linearbox.frame,
                                                   int(progress * 100)))

        for resizeratio in decimal_range(resizestart,
                                         resizestop,
                                         resizeincrement):
            resizedbox = linearbox.resize(resizeratio)
            if resizedbox.xbr > width:
                resizedbox.xbr = width
            if resizedbox.ybr > height:
                resizedbox.ybr = height

            wr = dim[0] / float(resizedbox.get_width())
            hr = dim[1] / float(resizedbox.get_height())
            rimage = im.resize((int(width * wr), int(height * hr)), 2)

            slidingspace = annotations.calculateslidingspace(
                resizedbox, slidingsearchwidth, (width, height))
            slidingwindows = annotations.buildslidingwindows(
                resizedbox, slidingspace, slidingskip)

            costim = convolution.hogrgb(rimage, dim, svm.hogweights(),
                                        svm.rgbweights())

            #plt.set_cmap("gray")
            #plt.imshow(costim.transpose())
            #plt.savefig("tmp/local{0}.png".format(linearbox.frame))
            #plt.clf()

            for slidingbox in slidingwindows:
                pairwisenode = pairwiseterms.get(slidingbox.xtl,
                                                 slidingbox.ytl)
                if pairwisenode:
                    cost = costim[<int>(slidingbox.xtl*wr),
                                  <int>(slidingbox.ytl*hr)]
                    cost += (lineardeviation * 
                             (slidingbox.xtl - linearbox.xtl) * 
                             (slidingbox.xtl - linearbox.xtl))
                    cost += (lineardeviation * 
                             (slidingbox.ytl - linearbox.ytl) * 
                             (slidingbox.ytl - linearbox.ytl))
                    if cost > upperthreshold:
                        cost = upperthreshold

                    if (not cltable.contains(slidingbox.xtl, slidingbox.ytl) or 
                       cltable.get(slidingbox.xtl, slidingbox.ytl).cost > cost):
                        node = Node(slidingbox, cost = cost,
                                    previous = pairwisenode)
                        cltable.set(slidingbox.xtl, slidingbox.ytl, node)
        pltable.dump(linearbox.frame)
        pltable = cltable

    target = pltable.get(stop.xtl, stop.ytl)
    if target is None:
        raise TrackImpossible()
    track = []
    while target is not None:
        track.append(target.box)
        target = target.previous
    track.reverse()
    return track

class TrackImpossible(Exception):
    def __str__(self):
        return "No track found for object given parameters"

cdef class Node(object):
    """A node in the dynamic programming without any pairwise constraints. In
    effect, this is a relation between a box and the cost."""
    
    @cython.profile(False)
    def __init__(self, annotations.Box box, double cost = 0,
                 Node previous = None):
        self.box = box
        self.cost = cost
        self.previous = previous

        self.total_cost = cost
        if previous is not None:
            self.total_cost += previous.total_cost

cdef class NodeMatrix(object):
    def __init__(self, int width, int height, int skip):
        self.width = width
        self.height = height
        self.skip = skip
        self.matrix = [None] * (width // skip)
        for i in range(width // skip):
            self.matrix[i] = [None] * (height // skip)

    @cython.profile(False)
    cdef inline bint contains(self, int x, int y):
        return self.matrix[x // self.skip][y // self.skip] is not None

    @cython.profile(False)
    cdef inline Node get(self, int x, int y):
        return self.matrix[x // self.skip][y // self.skip]

    @cython.profile(False)
    cdef inline set(self, int x, int y, Node node):
        self.matrix[x // self.skip][y // self.skip] = node

    def pairwise_build(self, int radius):
        cdef int i, j, x, y, pstart, pstop
        cdef NodeMatrix fpass, pairwises

        # vertical pass
        fpass = NodeMatrix(self.width, self.height, self.skip)
        for i from 0 <= i < self.width by self.skip:
            for j from 0 <= j < self.height by self.skip:
                pstart = j - radius
                pstop  = j + radius 
                if pstart < 0:
                    pstart = 0
                if pstop > self.height:
                    pstop = self.height
                for y from pstart <= y < pstop by self.skip:
                    if self.contains(i, y):
                        if (not fpass.contains(i, j) or
                            self.get(i, y).total_cost <
                            fpass.get(i, j).total_cost):
                            fpass.set(i, j, self.get(i, y))

        # horizontal pass
        pairwises = NodeMatrix(self.width, self.height, self.skip)
        for i from 0 <= i < self.width by self.skip:
            for j from 0 <= j < self.height by self.skip:
                pstart = i - radius
                pstop  = i + radius
                if pstart < 0:
                    pstart = 0
                if pstop > self.width:
                    pstop = self.width
                for x from pstart <= x < pstop by self.skip:
                    if fpass.contains(x, j):
                        if (not pairwises.contains(i, j) or 
                            fpass.get(x, j).total_cost <
                            pairwises.get(i, j).total_cost):
                            pairwises.set(i, j, fpass.get(x, j))

        return pairwises

    def dump(self, frame):
        import matplotlib.pyplot as plt
        data = numpy.zeros((self.width // self.skip, self.height // self.skip))
        for x, mdata in enumerate(self.matrix):
            for y, v in enumerate(mdata):
                if not v:
                    data[x,y] = 0
                else:
                    data[x,y] = -v.total_cost
        plt.set_cmap("gray")
        plt.imshow(data.transpose())
        plt.title("{0} to {1}".format(data.min(), data.max()))
        plt.savefig("tmp/pairwise{0}.png".format(frame))
        plt.clf()

def decimal_range(a, b, skip):
    start = min(a, b)
    stop = max(a, b)
    list = []
    next = start
    while next < stop:
        list.append(next)
        next += skip
    return list
