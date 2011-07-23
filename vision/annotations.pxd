cdef class Box(object):
    cdef public int xtl, ytl, xbr, ybr
    cdef public int lost
    cdef public int occluded
    cdef public int frame
    cdef public int generated
    cdef public object attributes
