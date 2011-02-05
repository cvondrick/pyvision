from .. import annotations

def calculateslidingspace(base, offset, frame):
    xstart = max(0, base.xtl - offset) 
    xstop  = min(frame[0] - 1, base.xbr + offset)
    ystart = max(0, base.ytl - offset)
    ystop  = min(frame[1] - 1, base.ybr + offset)
    return xstart, ystart, xstop, ystop

def buildslidingwindows(base, space, skip):
    """
    Generate sliding windows based off the image that are displaced and resized.
    """
    nextframe = base.frame
    w = base.get_width()
    h = base.get_height()
    xstart = space[0], ystart = space[1]
    xstop = space[2] - w
    ystop = space[3] - h
    boxes = []
    for i in range(xstart, xstop, skip):
        for j in range(ystart, ystop, skip):
            boxes.append(annotations.Box(i, j, i + w, j + h, nextframe))
    return boxes
