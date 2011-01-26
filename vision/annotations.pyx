import math
import numpy
import Image

cimport numpy
cimport cython

cdef extern from "math.h":
    double sqrt(double i)

cdef class Box(object):

    cdef public int xtl, ytl, xbr, ybr
    cdef public int lost, occluded
    cdef public int frame

    """
    A unlabeled bounding box not bound to a frame.
    """
    @cython.profile(False)
    def __init__(self, int xtl, int ytl, int xbr, int ybr, int frame = 0, int lost = 0, int occluded = 0):
        """Initializes the bounding box."""
        if xbr <= xtl:
            raise TypeError("xbr must be > xtl")
        elif ybr <= ytl:
            raise TypeError("ybr must be > ytl")
        elif xtl < 0:
            raise TypeError("xtl must be nonnegative")
        elif ytl < 0:
            raise TypeError("ytl must be nonnegative")

        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.frame = frame
        self.lost = lost
        self.occluded = occluded

    def center(self):
        """
        Calculates the center of the bounding box.
        """
        return self.xtl+self.get_width()/2, self.ytl+self.get_height()/2

    def area(self):
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def distance(self, oth):
        """
        Calculate the Euclidean distance between boxes.
        """
        scenter = abs(self.xbr  + self.xtl)  / 2, \
                  abs(self.ybr  + self.ytl)  / 2
        ocenter = abs(oth.xbr + oth.xtl) / 2, \
                  abs(oth.ybr + oth.ytl) / 2
        diff    = scenter[0] - ocenter[0], \
                  scenter[1] - ocenter[1]
        sum     = diff[0]**2 + diff[1]**2
        return math.sqrt(sum)

    def intersects(self, oth):
        """
        Determines if there is any overlap between two boxes.
        """
        xlap = max(self.xtl, oth.xtl) <= min(self.xbr, oth.xbr)
        ylap = max(self.ytl, oth.ytl) <= min(self.ybr, oth.ybr)
        return xlap and ylap

    def percentoverlap(self, oth):
        """
        Calculates the percent of boxes that overlap.
        """
        xdiff = min(self.xbr, oth.xbr) - max(self.xtl, oth.xtl) 
        ydiff = min(self.ybr, oth.ybr) - max(self.ytl, oth.ytl) 

        if xdiff <= 0 or ydiff <= 0:
            return 0

        uni = self.area() + oth.area() - xdiff * ydiff
        return float(xdiff * ydiff) / float(uni)

    def resize(self, xratio, yratio = None):
        """
        Resizes the box by the xratio and yratio. If no yratio is specified,
        defaults to the xratio.
        """
        if yratio is None:
            yratio = xratio

        return Box(self.xtl, self.ytl,
                   self.xtl + self.get_width() * xratio,
                   self.ytl + self.get_height() * yratio)

    def transform(self, xratio, yratio = None):
        """
        Transforms the space that the box exists in by an x and y ratio. If
        the y ratio is not specified, defaults to the xratio.
        """
        if yratio is None:
            yratio = xratio

        return Box(self.xtl * xratio, self.ytl * yratio,
                   self.xbr * xratio, self.ybr * yratio)

    def __str__(self):
        """
        Returns a string representation.
        """
        return "Box({0}, {1}, {2}, {3}, {4}, {5})".format(
            self.xtl, self.ytl, self.xbr, self.ybr,
            self.frame, self.lost)

    def __repr__(self):
        """
        Returns a string representation.
        """
        return str(self)

    def __richcmp__(self, other, int t):
        """
        A comparator  to see if boxes are equal or not.
        """
        if not isinstance(other, Box):
            return False

        equality = self.xtl is other.xtl and \
            self.ytl is other.ytl and \
            self.xbr is other.xbr and \
            self.ybr is other.ybr and \
            self.frame is other.frame and \
            self.lost is other.lost

        if t == 0:   return self.frame < other.frame
        elif t == 1: return self.frame <= other.frame
        elif t == 2: return equality
        elif t == 3: return not equality
        elif t == 4: return self.frame > other.frame
        elif t == 5: return self.frame >= other.frame
        else:        return False

    def __reduce__(self):
        """
        Provides support to serialize the box.
        """
        return (Box, (self.xtl, self.ytl, self.xbr, self.ybr, self.frame, self.lost, self.occluded))

    def __getitem__(self, a):
        """
        Allows accessing bounding box as if its a tuple
        """
        tuple = (self.xtl, self.ytl, self.xbr, self.ybr, self.frame, self.lost, self.occluded)
        return tuple[a]
