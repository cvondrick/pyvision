cdef class Box(object):
    cdef public int xtl, ytl, xbr, ybr
    cdef public int lost
    cdef public int frame
