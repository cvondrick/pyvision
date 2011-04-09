from vision cimport annotations

cdef class Node(object):
    cdef public annotations.Box box
    cdef public double cost
    cdef public double total_cost
    cdef public Node previous

cdef class NodeMatrix(object):
    cdef public int width, height, skip
    cdef public list matrix

    cdef bint contains(self, int x, int y)
    cdef Node get(self, int x, int y)
    cdef set(self, int x, int y, Node node)
