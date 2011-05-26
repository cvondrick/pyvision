import numpy as np
import features
import logging

cimport numpy as np
cimport cython

log = logging.getLogger("vision.convolution")

cpdef hogrgb(image, filtersize, hogfilter, rgbfilter,
             int hogbin = 8, int rgbbin = 8):
    """
    Efficiently convolves a filter around an image for HOG and RGB filters.

    Only computes features in the valid region of the image and returns a score
    matrix. The filtersize should be the size of the templated to score with,
    and must match with hogfilter.

    - image should be an Python Image Library image.
    - filtersize should be a 2-tuple of (width,height) sizes for the template
      filter.
    - hogfilter should be a (width/hogbin-2, height/hogbin-2, 13) numpy array
      of the learned HOG weights.
    - rgbfilter should be a rgbin^3 length matrix of the learned RGB histogram
      weights.
    """

    # initialize some useful stuff
    cdef int width = image.size[0], height = image.size[1], i = 0, j = 0
    cdef int filterwidth = filtersize[0], filterheight = filtersize[1]
    cdef np.ndarray[np.double_t, ndim=3] data
    data = np.asarray(image, dtype=np.double)

    log.debug("Convolving {width} x {height} image with "
              "{filter[0]} x {filter[1]} filter".format(width=width,
                                                        height=height,
                                                        filter=filtersize))
    log.debug("HOG filter is {0} with bin size {1}".format(hogfilter.shape,
                                                           hogbin))
    log.debug("RGB filter is {0} with bin size {1}".format(rgbfilter.shape,
                                                           rgbbin))

    # efficiently precompute hog features
    cdef np.ndarray[ndim=3, dtype=np.double_t] hogfeat
    hogfeat = features.hog(image, hogbin)
    cdef np.ndarray[ndim=3, dtype=np.double_t] hogfiltert = hogfilter

    # efficiently precompute integral rgb score image
    cdef np.ndarray[np.double_t, ndim=2] sumrgb = np.zeros((width, height))
    cdef np.ndarray[ndim=1, dtype=np.double_t] rgbfiltert = rgbfilter
    cdef double localrgbscore
    for i from 0 <= i < width:
        for j from 0 <= j < height:
            # we cannot manipulate the algebra below because we are taking
            # advantage of rounding
            localrgbbin  = (<int>data[j,i,0]) / (256/rgbbin)
            localrgbbin += (<int>data[j,i,1]) / (256/rgbbin) * rgbbin
            localrgbbin += (<int>data[j,i,2]) / (256/rgbbin) * rgbbin * rgbbin
            localrgbscore = rgbfiltert[localrgbbin]
            # lookup recursive scores
            if i > 0:
                localrgbscore += sumrgb[i-1, j]
            if j > 0:
                localrgbscore += sumrgb[i, j-1] 
                if i > 0: # do not count twice
                    localrgbscore -= sumrgb[i-1, j-1]
            sumrgb[i, j] = localrgbscore

    # convolve
    cdef int hfwidth = hogfilter.shape[0], hfheight = hogfilter.shape[1]
    cdef double hogscore, rgbscore, hogfeatvalue, hogfiltervalue
    cdef np.ndarray[ndim=2, dtype=np.double_t] output
    output = np.zeros((width - filterwidth, height - filterheight))

    for i from 0 <= i < width - filterwidth:
        for j from 0 <= j < height - filterheight:
            # compute hog score
            hogscore = 0. 
            for hfi from 0 <= hfi < hfwidth:
                for hfj from 0 <= hfj < hfheight:
                    for hfk from 0 <= hfk < 13:
                        hogfeatvalue = hogfeat[j/8+hfj, i/8+hfi, hfk] 
                        hogfiltervalue = hogfiltert[hfj, hfi, hfk]
                        hogscore += hogfeatvalue * hogfiltervalue

            # compute rgb score using summed area tables
            rgbscore  = sumrgb[i+filterwidth, j+filterheight] + sumrgb[i, j]
            rgbscore -= sumrgb[i, j+filterheight] + sumrgb[i+filterwidth, j]
            rgbscore  = rgbscore / (filterwidth * filterheight)

            # store final score
            output[i,j] = hogscore + rgbscore
    return output

cpdef hogrgbmean(image, filtersize, hogfilter, rgbfilter, int hogbin = 8):
    """
    Efficiently convolves a filter around an image for HOG and RGB filters.

    Only computes features in the valid region of the image and returns a score
    matrix. The filtersize should be the size of the templated to score with,
    and must match with hogfilter.

    - image should be an Python Image Library image.
    - filtersize should be a 2-tuple of (width,height) sizes for the template
      filter.
    - hogfilter should be a (width/hogbin-2, height/hogbin-2, 13) numpy array
      of the learned HOG weights.
    """

    # initialize some useful stuff
    cdef int width = image.size[0], height = image.size[1], i = 0, j = 0
    cdef int filterwidth = filtersize[0], filterheight = filtersize[1]
    cdef np.ndarray[np.double_t, ndim=3] data
    data = np.asarray(image, dtype=np.double)

    log.debug("Convolving {width} x {height} image with "
              "{filter[0]} x {filter[1]} filter".format(width=width,
                                                        height=height,
                                                        filter=filtersize))
    log.debug("HOG filter is {0} with bin size {1}".format(hogfilter.shape,
                                                           hogbin))
    log.debug("RGB filter is {0}".format(rgbfilter.shape))

    # efficiently precompute hog features
    cdef np.ndarray[ndim=3, dtype=np.double_t] hogfeat
    hogfeat = features.hog(image, hogbin)
    hogfeat = features.hogpad(hogfeat)
    cdef np.ndarray[ndim=3, dtype=np.double_t] hogfiltert = hogfilter

    # efficiently precompute integral rgb score image
    cdef np.ndarray[np.double_t, ndim=2] sumrgb = np.zeros((width, height))
    cdef np.ndarray[ndim=1, dtype=np.double_t] rgbfiltert = rgbfilter
    cdef double localrgbscore
    for i from 0 <= i < width:
        for j from 0 <= j < height:
            r = data[j, i, 0] / <double>(255)
            g = data[j, i, 1] / <double>(255)
            b = data[j, i, 2] / <double>(255)
            localrgbscore = 0
            localrgbscore += r * rgbfiltert[0]
            localrgbscore += g * rgbfiltert[1]
            localrgbscore += b * rgbfiltert[2]
            localrgbscore += r * r * rgbfiltert[3]
            localrgbscore += r * g * rgbfiltert[4]
            localrgbscore += r * b * rgbfiltert[5]
            localrgbscore += g * g * rgbfiltert[6]
            localrgbscore += g * b * rgbfiltert[7]
            localrgbscore += b * b * rgbfiltert[8]

            # lookup recursive scores
            if i > 0:
                localrgbscore += sumrgb[i-1, j]
            if j > 0:
                localrgbscore += sumrgb[i, j-1] 
                if i > 0: # do not count twice
                    localrgbscore -= sumrgb[i-1, j-1]
            sumrgb[i, j] = localrgbscore

    # convolve
    cdef int hfwidth = hogfilter.shape[0], hfheight = hogfilter.shape[1]
    cdef double hogscore, rgbscore, hogfeatvalue, hogfiltervalue
    cdef np.ndarray[ndim=2, dtype=np.double_t] output
    output = np.zeros((width - filterwidth,
                       height - filterheight))

    for i from 0 <= i < width - filterwidth:
        for j from 0 <= j < height - filterheight:
            # compute hog score
            hogscore = 0. 
            for hfi from 0 <= hfi < hfwidth:
                for hfj from 0 <= hfj < hfheight:
                    for hfk from 0 <= hfk < 13:
                        hogfeatvalue = hogfeat[j/hogbin+hfj, 
                                               i/hogbin+hfi, hfk] 
                        hogfiltervalue = hogfiltert[hfj, hfi, hfk]
                        hogscore += hogfeatvalue * hogfiltervalue

            # compute rgb score using summed area tables
            rgbscore  = sumrgb[i, j]
            rgbscore += sumrgb[i+filterwidth, j+filterheight]
            rgbscore -= sumrgb[i, j+filterheight]
            rgbscore -= sumrgb[i+filterwidth, j]

            rgbscore = rgbscore / (filterwidth * filterheight)

            # store final score
            output[i,j] = hogscore + rgbscore
    return output
