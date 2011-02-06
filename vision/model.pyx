import svm
import annotations
import features
import random
import logging
import numpy

logger = logging.getLogger("vision.model")

cimport numpy
from vision cimport annotations

class PathModel(object):
    """
    A model that learns a linear SVM weight vector based off a path.

    The model extracts positve examples from the given path and negative
    examples from sliding windows that do not overlap with a given example.
    We extract both HOG features and RGB features. 

    For fast scoring, use the corresponding convolution.pyx routine.
    """

    def __init__(self, images, givens, dim = (40,40), hogbin = 8,
                 rgbbin = 8, bgskip = 4, bgsize = 5e4):
        """
        Constructs a path based model from the given path.
        """
        self.dim = dim
        self.hogbin = hogbin
        self.rgbbin = rgbbin

        logger.info("Extracting features from path")
        positives, negatives = self.extractpath(images, givens, dim,
                                                hogbin, rgbbin, bgskip,
                                                bgsize)

        # svm.sanity(positives, negatives) # uncomment when debugging

        logger.info("Learning weights for path with {0} foregrounds and "
                    "{1} backgrounds".format(len(positives), len(negatives)))
        model = svm.train(negatives, positives)
        self.weights, self.bias = model.weights, model.bias

        logger.info("Weights learned with bias {0}".format(self.bias))

    def extractpath(self, images, givens, dim, 
                    int hogbin, int rgbbin, int bgskip, int bgsize):
        """
        Extracts features from a path.
        """
        cdef int dimw = dim[0], dimh = dim[1]
        cdef int imw, imh
        cdef int i, j, xtl, ytl, xbr, ybr
        cdef double wr, hr
        cdef annotations.Box given

        cdef numpy.ndarray[ndim=3, dtype=numpy.double_t] hogim

        positives = []
        negatives = []
        for given in givens:
            logger.debug("Extracting features from "
                         "frame {0}".format(given.frame))
            wr = float(dim[0]) / given.width
            hr = float(dim[1]) / given.height
            im = images[given.frame]
            im = im.resize((int(im.size[0]*wr), int(im.size[1]*hr)))
            imw, imh = im.size
            mapped = given.transform(wr, hr)

            # positives
            xtl, ytl, xbr, ybr = mapped[0:4]
            patch = im.crop((xtl-hogbin, ytl-hogbin, xbr+hogbin, ybr+hogbin))
            hogpatch = features.hog(patch, hogbin)[1:-1,1:-1].flatten()
            rgbpatch = features.rgbhist(patch, rgbbin).flatten()
            positives.append(numpy.append(hogpatch, rgbpatch))

            logger.debug("Extracting negatives")

            # negatives
            hogim = features.hog(im, hogbin)
            for i from 0 <= i < imw-dimw by bgskip:
                for j from 0 <= j < imh-dimh by bgskip:
                    if not annotations.Box(i/wr,
                                           j/hr, 
                                           (i + dimw)/wr,
                                           (j + dimh)/hr).intersects(given):

                        hogpatch = hogim[j/hogbin:(j+dimh)/hogbin-2,
                                         i/hogbin:(i+dimw)/hogbin-2, :]
                        hogpatch = hogpatch.flatten()

                        rgbpatchregion = (i, j, i+dimw, j+dimh)
                        rgbpatch = features.rgbhist(im.crop(rgbpatchregion))
                        rgbpatch = rgbpatch.flatten()

                        negatives.append(numpy.append(hogpatch, rgbpatch))
        if len(negatives) > bgsize:
            negatives = random.sample(negatives, int(bgsize))
        return positives, negatives

    def hogweights(self):
        """
        Returns the SVM weights just for the HOG features.
        """
        x = self.dim[0]/self.hogbin-2
        y = self.dim[1]/self.hogbin-2
        return self.weights[0:x*y*13].reshape((y, x, 13))

    def rgbweights(self):
        """
        Returns the SVM weights just for the RGB features.
        """
        x = self.dim[0]/self.hogbin-2
        y = self.dim[1]/self.hogbin-2
        return self.weights[x*y*13:]
