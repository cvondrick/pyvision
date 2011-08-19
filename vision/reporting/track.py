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
from vision import annotations
from vision import frameiterator, readpaths
from vision.alearn import marginals
from vision import visualize
from math import ceil, floor
import itertools
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, mpl
import logging
import os

logger = logging.getLogger("vision.reporting")

class Engine(object):
    """
    An engine to predict given a fixed number of clicks.
    """
    def __call__(self, video, groundtruths, cpfs, pool = None):
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
    def __call__(self, video, gtruths, cpfs, pool = None):
        """
        Computes the correct skip for a given cpf and builds the dictionary.
        Child classes should implement predict().
        """
        result = {}
        numframes = sum(x[-1].frame - x[0].frame for x in gtruths.values())
        logger.info("Total of {0} frames".format(numframes))
        for cpf in cpfs:
            clicks = int(cpf * numframes)
            usedclicks = 0

            logger.info("CPF {0} has {1} clicks".format(cpf, clicks))

            schedule = {}
            for id, gtruth in gtruths.items():
                gtruth.sort(key = lambda x: x.frame)
                pathclicks = clicks 
                pathclicks *= float(gtruth[-1].frame - gtruth[0].frame)
                pathclicks /= numframes
                pathclicks = int(floor(pathclicks))
                schedule[id] = max(pathclicks, 1)
                usedclicks += schedule[id]
            for id, _ in zip(itertools.cycle(gtruths.keys()),
                             range(clicks - usedclicks)):
                schedule[id] += 1

            for id, clicksinschedule in schedule.items():
                logger.info("ID {0} has {1} clicks for {2} frames".format(id,
                    clicksinschedule, len(gtruths[id])))

            for id, gtruth in gtruths.items():
                skip = int(ceil(float(gtruth[-1].frame - gtruth[0].frame) / schedule[id]))
                given = gtruth[::skip]
                given = given[:schedule[id]]

                if id not in result:
                    result[id] = {}
                logger.info("Processing {0} with {1} clicks".format(id, 
                    schedule[id]))
                result[id][cpf] = self.predict(video, given, gtruth[-1].frame,
                                               pool = pool)
        return result


    def predict(self, video, given, last, pool):
        """
        Given a video and a sparse path, predict the missing annotations.
        """
        raise NotImplementedError("predict() must be implemented")

class LinearInterpolationEngine(FixedRateEngine):
    """
    Uses linear interpolation to predict missing annotations as a fixed rate
    engine.
    """
    def predict(self, video, given, last, pool):
        path = interpolation.LinearFill(given)
        while path[-1].frame <= last:
            path.append(annotations.Box(path[-1].xtl,
                                        path[-1].ytl,
                                        path[-1].xbr,
                                        path[-1].ybr,
                                        path[-1].frame + 1))
        return path
    def color(self):
        return "b"

class DynamicProgrammingEngine(FixedRateEngine):
    """
    Uses a dynamic programming based tracker to predict the missing
    annotations.
    """
    def __init__(self, pairwisecost = 0.001, upperthreshold = 10, skip = 3, rgbbin = 8, hogbin = 8):
        self.pairwisecost = pairwisecost
        self.upperthreshold = upperthreshold
        self.skip = skip
        self.rgbbin = rgbbin
        self.hogbin = hogbin

    def predict(self, video, given, last, pool):
        return dp.fill(given, video, last = last, pool = pool,
                       pairwisecost = self.pairwisecost,
                       upperthreshold = self.upperthreshold,
                       skip = self.skip,
                       rgbbin = self.rgbbin,
                       hogbin = self.hogbin)

    def color(self):
        return "r"

class ActiveLearnDPEngine(Engine):
    """
    Uses an active learning approach to annotate the most informative frames.
    """
    def __init__(self, pairwisecost = 0.001, upperthreshold = 10, sigma = .1,
                 erroroverlap = 0.5, skip = 3, rgbbin = 8, hogbin = 8):
        self.pairwisecost = pairwisecost
        self.upperthreshold = upperthreshold
        self.sigma = sigma
        self.erroroverlap = erroroverlap
        self.skip = skip
        self.rgbbin = rgbbin
        self.hogbin = hogbin

    def __call__(self, video, gtruths, cpfs, pool = None):
        result = {}
        pathdict = {}
        for id, gtruth in gtruths.items():
            gtruth.sort(key = lambda x: x.frame)
            pathdict[id] = dict((x.frame, x) for x in gtruth)

        requests = {}
        for id, gtruth in gtruths.items():
            frame, score, predicted, _ = marginals.pick([gtruth[0]], video, 
                                         last = gtruth[-1].frame,
                                         pool = pool,
                                         pairwisecost = self.pairwisecost,
                                         upperthreshold = self.upperthreshold,
                                         sigma = self.sigma,
                                         erroroverlap = self.erroroverlap,
                                         skip = self.skip,
                                         rgbbin = self.rgbbin,
                                         hogbin = self.hogbin)
                                                     
            requests[id] = (score, frame, predicted, [gtruth[0]])
            result[id] = {}
        usedclicks = len(gtruths)

        logger.info("Used {0} clicks!".format(usedclicks))

        numframes = sum(x[-1].frame - x[0].frame for x in gtruths.values())
        reqclicks = [(int(numframes * x), x) for x in cpfs]
        reqclicks.sort()

        for clicks, cpf in reqclicks:
            for _ in range(clicks - usedclicks):
                id = max((y[0], x) for x, y in requests.items())[1]

                givens = list(requests[id][3])
                givens.append(pathdict[id][requests[id][1]])
                givens.sort(key = lambda x: x.frame)

                frame, score, predicted, _ = marginals.pick(givens, video,
                                        last = max(pathdict[id]),
                                        pool = pool,
                                        pairwisecost = self.pairwisecost,
                                        upperthreshold = self.upperthreshold,
                                        sigma = self.sigma,
                                        erroroverlap = self.erroroverlap,
                                        skip = self.skip,
                                        rgbbin = self.rgbbin,
                                        hogbin = self.hogbin)

                requests[id] = (score, frame, predicted, givens)
                usedclicks += 1

                logger.info("Used {0} clicks with {1} total in this cpf!"
                            .format(usedclicks, clicks))

            for id, (_, _, path, _) in requests.iteritems():
                result[id][cpf] = path
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

class Intersection(object):
    def __call__(self, x, y):
        if x.frame != y.frame:
            raise RuntimeError("Frames do not match")

        if x.intersects(y):
            return 0
        else:
            return 1

    def __str__(self):
        return "Intersection"

def filetovideo(base):
    def process(filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        return frameiterator(base + "/" + name)
    return process

def load(data, frames, onlylabels = None, breakup = True,
         limit = None, toframe = None):
    """
    Produces a list over tracks found in the files for data. frames 
    should be a callable that returns a frame iterator.
    """ 
    result = []
    numpaths = 0
    for file in data:
        video = frames(file)
        paths = []
        for label, path in readpaths(open(file)):
            if not onlylabels or label in onlylabels:
                if breakup:
                    path.sort(key = lambda x: x.frame)
                    currentpath = []
                    for box in path:
                        if box.lost:
                            if len(currentpath) > 1:
                                paths.append((label, currentpath))
                                currentpath = []
                                numpaths += 1
                        else:
                            currentpath.append(box)
                    if len(currentpath) > 1:
                        paths.append((label, currentpath))
                        numpaths += 1
                elif len(currentpath) > 1:
                    paths.append((label, path))
                    numpaths += 1
        result.append((video, paths))

    # cut after a certain frame
    if toframe:
        cutresult = []
        for video, paths in result:
            pathsresult = []
            for label, path in paths:
                filtered = [x for x in path if x.frame <= toframe]
                if filtered:
                    pathsresult.append((label, filtered))
            if pathsresult:
                cutresult.append((video, pathsresult))
        result = cutresult

    # limit the number of videos
    if limit:
        limitresult = []
        counter = 0
        for video, paths in result:
            pathsresult = []
            for label, path in paths:
                counter += 1
                if counter <= limit:
                    pathsresult.append((label, path))
            if pathsresult:
                limitresult.append((video, pathsresult))
        result = limitresult

    return result

def merge(datas):
    merged = {}
    strmapping = {}
    for data in datas:
        for engine, predictions in data.iteritems():
            if str(engine) not in strmapping:
                strmapping[str(engine)] = engine
                merged[engine] = []
            print strmapping
            merged[strmapping[str(engine)]].extend(predictions)
    return merged

def build(data, cpfs, engines, pool = None):
    """
    Takes the data and runs the engines.
    """
    result = {}
    for engine in engines:
        logger.info("Computing tracks with {0}".format(str(engine)))
        result[engine] = []
        for video, paths in data:
            for path in paths:
                path[1].sort(key = lambda x: x.frame)
            paths = [x[1] for x in paths]

            keys = range(len(paths))
            paths = dict(zip(keys, paths))

            predictions = engine(video, paths, cpfs, pool)

            result[engine].extend((predictions[x], paths[x], video)
                                  for x in keys)
    return result

def scoreperformance(data, scorer):
    """
    Plots a performance curve for the data with the specified engines.
    """
    results = {}
    for engine, predictions in data.iteritems():
        logger.info("Plotting and scoring tracks for {0}".format(str(engine)))
        scores = {}
        lengths = {}
        for prediction, groundtruth, video in predictions:
            for cpf, path in prediction.iteritems():
                if cpf not in scores:
                    scores[cpf] = 0
                    lengths[cpf] = 0

                try:
                    score = sum(scorer(x,y) for x, y in zip(path, groundtruth))
                except Exception as e:
                    logger.exception(e)
                else:
                    scores[cpf] += score
                    lengths[cpf] += len(path)

        # normalize scores
        for cpf in scores:
            scores[cpf] = scores[cpf] / float(lengths[cpf])

        results[engine] = zip(*sorted(scores.items()))
    return results

def plotperformance(data, scorer, only = []):
    fig = pylab.figure()
    for engine, (x, y) in scoreperformance(data, scorer).items():
        if only and str(engine) not in only:
            continue
        pylab.plot(x, y, "{0}.-".format(engine.color()), label = str(engine),
                   linewidth = 4)
    pylab.ylabel("Average error per frame ({0})".format(str(scorer)))
    pylab.xlabel("Average clicks per frame per object")
    pylab.legend()
    pylab.show()

def plotcorrect(data, scorer, threshold = 0):
    fig = pylab.figure()
    for engine, predictions in data.iteritems():
        counter = {}
        for prediction, groundtruth, video in predictions:
            for cpf, path in prediction.iteritems():
                if cpf not in counter:
                    counter[cpf] = 0

                try:
                    score = sum(scorer(x, y) for x, y in zip(path, groundtruth))
                except Exception as e:
                    logger.exception(e)
                else:
                    if score <= threshold:
                        counter[cpf] += 1

        x, y = zip(*sorted(counter.items()))
        pylab.plot(x, y, "{0}.-".format(engine.color()), label = str(engine),
                   linewidth = 4)
    pylab.ylabel("Number of completely correct tracks ({0}, threshold = {1})".format(str(scorer), threshold))
    pylab.xlabel("Average clicks per frame per object")
    pylab.legend()
    pylab.show()

def plotsurface(input, scorer, left, right,
    cpucostfact = 20, humancostfact = 2500 / 5):

    data = scoreperformance(input, scorer)

    # find left and right
    for potential in data.keys():
        if str(potential) == left:
            left = potential
        elif str(potential) == right:
            right = potential


    cpucost = numpy.arange(0, 1.05, 0.05)
    humcost = numpy.asarray(sorted(data[left][0]))
    error = numpy.zeros((humcost.shape[0], cpucost.shape[0]))

    cpucost, humcost = numpy.meshgrid(cpucost, humcost)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("CPU Cost")
    ax.set_ylabel("Human Cost")
    ax.set_zlabel("Error")
    ax.plot_surface(cpucost, humancost, error)
    ax.legend()

    plt.show()

def visualizepaths(data, dir):
    for engine, predictions in data.iteritems():
        for id, (prediction, groundtruth, video) in enumerate(predictions):
            for cpf, path in prediction.iteritems():
                logger.info("Visualizing engine {0} path {1} cpf {2}"
                            .format(str(engine), id, cpf))
                filepath = "{0}/{1}/{2}/{3}".format(dir, str(engine), id, cpf)
                try:
                    os.makedirs(filepath)
                except OSError:
                    pass
                vis = visualize.highlight_paths(video, [path, groundtruth])
                visualize.save(vis, lambda x: "{0}/{1}.jpg".format(filepath, x))
