import numpy
cimport numpy

cdef double Infinity = 1e300

cdef extern from "math.h":
    float exp(float n)

# see Pedro Felzenszwalb et. al
cpdef quadratic_1d(numpy.ndarray[numpy.double_t, ndim=1] src,
                            numpy.ndarray[numpy.double_t, ndim=1] dst,
                            numpy.ndarray[numpy.int_t, ndim=1] ptr,
                            int step, int n, double a, double b, int o):
    
    cdef numpy.ndarray[numpy.int_t, ndim=1] v = numpy.zeros(n, dtype=numpy.int)
    cdef numpy.ndarray[numpy.double_t, ndim=1] z = numpy.zeros(n+1,
                                                   dtype = numpy.double)
    cdef int k = 0
    v[0] = 0
    z[0] = -Infinity
    z[1] = Infinity

    cdef int q
    cdef double s

    for q in range(1, n):
        s = ((src[q*step+o]-src[v[k]*step+o])-b*(q-v[k])+a*(q**2-v[k]**2))
        s = s / (2*a*(q-v[k]))

        while s <= z[k]:
            k = k - 1
            s = ((src[q*step+o]-src[v[k]*step+o])-b*(q-v[k])+a*(q**2-v[k]**2))
            s = s / (2*a*(q-v[k]))

        k = k + 1
        v[k] = q
        z[k] = s
        z[k+1] = Infinity

    k = 0
    for q in range(0, n):
        while z[k+1] < q:
            k = k + 1
        dst[q*step+o] = a*(q-v[k])**2 + b*(q-v[k]) + src[v[k]*step+o]
        ptr[q*step+o] = v[k]

# see Pedro Felzenszwalb et. al
def quadratic(numpy.ndarray[numpy.double_t, ndim=2] scores,
                       double cost):
    cdef int w, h, x, y, p
    w = scores.shape[0]
    h = scores.shape[1]

    cdef numpy.ndarray[numpy.double_t, ndim=1] vals = scores.flatten()

    cdef numpy.ndarray[numpy.double_t, ndim=1] M = numpy.zeros(w*h, 
                                                   dtype = numpy.double)
    cdef numpy.ndarray[numpy.int_t, ndim=2] Ix = numpy.zeros((w, h), 
                                                 dtype = numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=2] Iy = numpy.zeros((w, h), 
                                                 dtype = numpy.int)

    cdef numpy.ndarray[numpy.double_t, ndim=1] tmpM = numpy.zeros(w*h, 
                                                      dtype = numpy.double)
    cdef numpy.ndarray[numpy.int_t, ndim=1] tmpIx = numpy.zeros(w*h, 
                                                    dtype = numpy.int)
    cdef numpy.ndarray[numpy.int_t, ndim=1] tmpIy = numpy.zeros(w*h, 
                                                    dtype = numpy.int)

    for x in range(w):
        quadratic_1d(vals, tmpM, tmpIy, 1, h, cost, 0, x * h)

    for y in range(h):
        quadratic_1d(tmpM, M, tmpIx, h, w, cost, 0, y)

    for x in range(w):
        for y in range(h):
            p = x * h + y
            Ix[x, y] = tmpIx[p]
            Iy[x, y] = tmpIy[tmpIx[p]*h+y]

    return M.reshape((w,h)), Ix, Iy

def manhattan(inscores, incost):
    cdef int w, h, i, j, ri, rj
    w, h = inscores.shape

    cdef double cost = incost
    cdef numpy.ndarray[numpy.double_t, ndim=2] scores = inscores

    cdef numpy.ndarray[numpy.double_t, ndim=2] forward, backward
    cdef numpy.ndarray[numpy.int_t, ndim=2] forwardxp, forwardyp
    cdef numpy.ndarray[numpy.int_t, ndim=2] backwardxp, backwardyp

    forward = numpy.zeros((w,h), dtype = numpy.double)
    forwardxp = numpy.zeros((w,h), dtype = numpy.int)
    forwardyp = numpy.zeros((w,h), dtype = numpy.int)
    backward = numpy.zeros((w,h), dtype = numpy.double)
    backwardxp = numpy.zeros((w,h), dtype = numpy.int)
    backwardyp = numpy.zeros((w,h), dtype = numpy.int)

    # forward pass
    for i in range(w):
        for j in range(h):
            forward[i, j] = scores[i, j]
            forwardxp[i, j] = i
            forwardyp[i, j] = j

            if j-1 >= 0 and forward[i, j-1] + cost < forward[i, j]:
                forward[i, j] = forward[i, j-1] + cost
                forwardxp[i, j] = forwardxp[i, j-1]
                forwardyp[i, j] = forwardyp[i, j-1]
            if i-1 >= 0 and forward[i-1, j] + cost < forward[i, j]:
                forward[i, j] = forward[i-1, j] + cost
                forwardxp[i, j] = forwardxp[i-1, j]
                forwardyp[i, j] = forwardyp[i-1, j]

    # backwards pass
    for ri in range(w):
        i = w - ri - 1
        for rj in range(h):
            j = h - rj - 1
            backward[i, j] = forward[i, j]
            backwardxp[i, j] = forwardxp[i, j]
            backwardyp[i, j] = forwardyp[i, j]

            if j+1 < h and backward[i, j+1] + cost < backward[i, j]:
                backward[i, j] = backward[i, j+1] + cost
                backwardxp[i, j] = backwardxp[i, j+1]
                backwardyp[i, j] = backwardyp[i, j+1]
            if i+1 < w and backward[i+1, j] + cost < backward[i, j]:
                backward[i, j] = backward[i+1, j] + cost
                backwardxp[i, j] = backwardxp[i+1, j]
                backwardyp[i, j] = backwardyp[i+1, j]

    return backward, backwardxp, backwardyp

def hinge(inscores, int radius = 30):
    cdef int w, h, i, j, pstart, pstop, x, y 
    w, h = inscores.shape

    cdef numpy.ndarray[numpy.double_t, ndim=2] scores = inscores
    cdef numpy.ndarray[numpy.double_t, ndim=2] vertical, horizontal
    cdef numpy.ndarray[numpy.int_t, ndim=2] xp, yp

    vertical = numpy.zeros((w,h), dtype = numpy.double)
    horizontal = numpy.zeros((w,h), dtype = numpy.double)
    xp = numpy.zeros((w,h), dtype = numpy.int)
    yp = numpy.zeros((w,h), dtype = numpy.int)

    for i in range(w):
        for j in range(h):
            pstart = j - radius
            pstop  = j + radius 
            if pstart < 0:
                pstart = 0
            if pstop > h:
                pstop = h
            vertical[i, j] = scores[i, j]
            yp[i, j] = j
            for y in range(pstart, pstop): 
                if scores[i, y] < vertical[i, j]:
                    vertical[i, j] = scores[i, y]
                    yp[i, j] = y

    for i in range(w):
        for j in range(h):
            pstart = i - radius
            pstop  = i + radius
            if pstart < 0:
                pstart = 0
            if pstop > w:
                pstop = w
            horizontal[i, j] = vertical[i, j]
            xp[i, j] = i
            for x in range(pstart, pstop):
                if vertical[x, j] < horizontal[i, j]:
                    horizontal[i, j] = vertical[x, j]
                    xp[i, j] = x

    return horizontal, xp, yp
