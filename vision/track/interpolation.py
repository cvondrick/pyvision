from ..boundingboxes import *

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
        results.append(LBox(xtl, ytl, xbr, ybr, i, 0))

    return results

def LinearFill(path, method = Linear):
    """
    Takes a sparse path and performs linear interpolation between the points.

    Set method to Lost if the path contains lossable boxes.
    """
    result = []
    for x, y in zip(path, path[1:]):
        result.extend(method(x, y)[:-1])
    result.append(path[-1])
    return result

def Lost(source, target):
    curry = LostCurry()
    return curry(source, target)

def LostCurry(halfway = True):
    """
    Performs linear interpolation between a source and target, but attempts
    to interpolate boxes when the source or target or both are lost.
    """

    def compute(source, target):
        if target.frame <= source.frame:
            raise ValueError("Target frame must be greater than source frame.")

        halfwaypoint = (target.frame - source.frame) // 2 + source.frame

        if target.lost and source.lost:
            path = [LBox(source.xtl, source.ytl, source.xbr, source.ybr,
                frame, True) for frame in range(source.frame, target.frame + 1)]
                
        elif target.lost and not source.lost:
            if halfway:
                path = [LBox(source.xtl, source.ytl, source.xbr, source.ybr,
                    frame, False) for frame in range(source.frame, halfwaypoint)]
                path.extend(LBox(target.xtl, target.ytl, target.xbr, target.ybr,
                    frame, True) for frame in range(halfway, target.frame + 1))
            else:
                path = []

        elif not target.lost and source.lost:
            if halfway:
                path = [LBox(source.xtl, source.ytl, source.xbr, source.ybr,
                    frame, True) for frame in range(source.frame, halfwaypoint)]
                path.extend(LBox(target.xtl, target.ytl, target.xbr, target.ybr,
                    frame,False) for frame in range(halfwaypoint, target.frame+1))
            else:
                path = []

        else:
            path = Linear(source, target)

        return path
    return compute
