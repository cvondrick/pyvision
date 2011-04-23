"""
A module to produce performance plots about tracking approaches.

>>> engines = [LinearInterpolationEngine(),
...            DynamicProgrammingEngine(),
...            ActiveLearnLinearEngine()]
>>> cpfs = [.01, .02, .05, .1]
>>> frames = filetovideo("/scratch/frames/")

>>> data = load(["video.txt", "video2.txt"], frames)
>>> data = build(data, cpfs, engines, multiprocessing.Pool(24))

>>> fig = plotperformance(data, lambda x, y: x.percentoverlap(y) > .5)
>>> fig.show()
"""

from vision.track import interpolation
from vision.track import dp
from vision import frameiterator, readpaths
from vision import alearn
from math import ceil
import pylab
import logging
import os

logger = logging.getLogger("vision.reporting")

class Engine(object):
    """
    An engine to predict given a fixed number of clicks.
    """
    def __call__(self, video, groundtruth, cpfs):
        """
        Returns a dictionary of predicted tracks given the ground truth. Each
        key in the dictionary must be a click-per-frame as specified by the
        cpfs parameter and the value is a predicted path.
        """
        raise NotImplementedError("__call__() must be implemented")

    def __str__(self):
        """
        Returns the name of the engine, for reporting in the graph.
        """
        name = self.__class__.__name__
        if name[-6:] == "Engine":
            name = name[0:-6]
        return name

    def color(self):
        """
        Returns the color used to represent this engine.
        """
        raise NotImplementedError("color() must be implemented")

class FixedRateEngine(Engine):
    """
    An abstract engine that uses a fixed skip interval (e.g., linear
    interpolation).
    """
    def __call__(self, video, groundtruth, cpfs):
        """
        Computes the correct skip for a given cpf and builds the dictionary.
        Child classes should implement predict().
        """
        groundtruth.sort(key = lambda x: x.frame)
        result = {}
        for cpf in cpfs:
            # compute how many clicks we have, but subtract one for end point
            clicks = (groundtruth[-1].frame - groundtruth[0].frame) * cpf - 1
            skip = int(ceil(len(groundtruth) / clicks))
            given = groundtruth[::skip]

            # add end point if it is missing
            if groundtruth[-1].frame != given[-1].frame:
                given.append(groundtruth[-1])

            result[cpf] = self.predict(video, given)
        return result

    def predict(self, video, given):
        """
        Given a video and a sparse path, predict the missing annotations.
        """
        raise NotImplementedError("predict() must be implemented")

class LinearInterpolationEngine(FixedRateEngine):
    """
    Uses linear interpolation to predict missing annotations as a fixed rate
    engine.
    """
    def predict(self, video, given):
        return interpolation.LinearFill(given)

    def color(self):
        return "b"

class DynamicProgrammingEngine(FixedRateEngine):
    """
    Uses a dynamic programming based tracker to predict the missing
    annotations.
    """
    def predict(self, video, given):
        try:
            return dp.fill(given, video)
        except dp.TrackImpossible:
            logger.error("Impossible track! Revert to linear.")
            return interpolation.LinearFill(given)

    def color(self):
        return "r"

class ActiveLearnLinearEngine(Engine):
    """
    Uses an active learning approach to annotate the most informative frames
    with linear interpolation between clicks.
    """
    def __init__(self, pool = None):
        self.pool = pool

    def __call__(self, video, gtruth, cpfs):
        reqclicks = []
        for cpf in cpfs:
            reqclicks.append((int((gtruth[-1].frame - gtruth[0].frame) * cpf),
                              cpf))
        reqclicks.sort()

        gtruth.sort(key = lambda x: x.frame)
        gtruthdict = dict((x.frame, x) for x in gtruth)
        given = [gtruth[0], gtruth[-1]]

        numclicks = max(reqclicks)[0]
        logger.info("Performing active learning for {0} clicks".format(numclicks))

        simulation = {}
        for clicks in range(2, numclicks + 1):
            given.sort(key = lambda x: x.frame)
            simulation[clicks] = interpolation.LinearFill(given)
            wants = alearn.linear.pick(video, given, pool = self.pool)
            given.append(gtruthdict[wants])

        result = {}
        for clicks, cpf in reqclicks:
            if clicks < 2:
                result[cpf] = interpolation.LinearFill([gtruth[0], gtruth[-1]])
            else:
                result[cpf] = simulation[clicks]

        return result

    def color(self):
        return "g"

class PercentOverlap(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x, y):
        if x.frame != y.frame:
            raise RuntimeError("Frames do not match")

        if x.percentoverlap(y) >= self.threshold:
            return 0
        else:
            return 1

    def __str__(self):
        return "Percent Overlap >= {0}".format(self.threshold)

def filetovideo(base):
    def process(filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        return frameiterator(base + "/" + name)
    return process

def load(data, frames, onlylabels = None, breakup = True):
    """
    Produces a list over tracks found in the files for data. frames 
    should be a callable that returns a frame iterator.
    """ 
    result = []
    for file in data:
        video = frames(file)
        paths = readpaths(open(file))
        for label, path in paths:
            if not onlylabels or label in onlylabels:
                if breakup:
                    path.sort(key = lambda x: x.frame)
                    currentpath = []
                    for box in path:
                        if box.lost:
                            if currentpath:
                                result.append((video, label, currentpath))
                                currentpath = []
                        else:
                            currentpath.append(box)
                    if currentpath:
                        result.append((video, label, currentpath))
                else:
                    result.append((video, label, path))
    logger.info("Loaded {0} paths".format(len(result)))
    return result

def build(data, cpfs, engines, pool = None):
    """
    Takes the data and runs the engines. For best performance, specify
    pool to a multiprocessing.Pool object, which will allow this method to
    run in parallel.
    """
    mapper = pool.map if pool else map
    result = {}
    for engine in engines:
        logger.info("Computing tracks with {0}".format(str(engine)))
        result[engine] = mapper(build_do, ((x, cpfs, engine) for x in data))
    return result

def build_do(workorder):
    (video, label, path), cpfs, engine = workorder
    return engine(video, path, cpfs), path

def plotperformance(data, scorer):
    """
    Plots a performance curve for the data with the specified engines.
    """
    fig = pylab.figure()
    for engine, predictions in data.iteritems():
        logger.info("Plotting and scoring tracks for {0}".format(str(engine)))
        scores = {}
        lengths = {}
        for prediction, groundtruth in predictions:
            for cpf, path in prediction.iteritems():
                if cpf not in scores:
                    scores[cpf] = 0
                    lengths[cpf] = 0

                score = sum(scorer(x,y) for x, y in zip(path, groundtruth))
                scores[cpf] += score
                lengths[cpf] += len(path)

        # normalize scores
        for cpf in scores:
            scores[cpf] = scores[cpf] / float(lengths[cpf])

        x, y = zip(*sorted(scores.items()))
        pylab.plot(x, y, "{0}.-".format(engine.color()), label = str(engine))

    pylab.ylabel("Average error per frame ({0})".format(str(scorer)))
    pylab.xlabel("Average clicks per frame")
    pylab.legend()
    pylab.show()
