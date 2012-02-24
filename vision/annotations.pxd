cdef class Box(object):
    cdef public int xtl, ytl, xbr, ybr
    cdef public int lost
    cdef public int occluded
    cdef public int frame
    cdef public int generated
    cdef public object image
    cdef public object label
    cdef public double score
    cdef public object attributes
