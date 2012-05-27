from ..annotations import *
import logging

logger = logging.getLogger("vision.track.interpolation")

def Linear(source, target):
    """
    Performs basic linear interpolation in between a source and a target.
    """
    if target.frame <= source.frame:
        raise ValueError("Target frame must be greater than source frame "
        "(source = {0}, target = {1})".format(source.frame, target.frame))

    fdiff = float(target.frame - source.frame)
    xtlr  = (target.xtl - source.xtl) / fdiff
    ytlr  = (target.ytl - source.ytl) / fdiff
    xbrr  = (target.xbr - source.xbr) / fdiff
    ybrr  = (target.ybr - source.ybr) / fdiff

    results = []

    for i in range(source.frame, target.frame + 1):
        off = i - source.frame
        xtl = source.xtl + xtlr * off
        ytl = source.ytl + ytlr * off
        xbr = source.xbr + xbrr * off
        ybr = source.ybr + ybrr * off
        generated = int(i != source.frame and i != target.frame)
        lost = source.lost or target.lost
        results.append(Box(xtl, ytl, xbr, ybr,
                       frame = i, 
                       lost = lost,
                       occluded = source.occluded,
                       generated = generated,
                       attributes = list(source.attributes)))

    return results

def LinearFill(path, method = Linear):
    """
    Takes a sparse path and performs linear interpolation between the points.
    """
#    if logger.isEnabledFor(logging.DEBUG):
#        logger.debug("Linear fill for path:")
#        for item in path:
#            logger.debug(item)
        
    result = []
    for x, y in zip(path, path[1:]):
        result.extend(method(x, y)[:-1])
    result.append(path[-1])
    return result
