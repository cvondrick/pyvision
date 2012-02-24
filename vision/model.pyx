import svm
import annotations
import features
import random
import logging
import numpy
import convolution
from math import ceil

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
                 rgbbin = 8, bgskip = 2, bgsize = 5e4, c = 0.000001,
                 realprior = None, realpriorweight = 10):
        """
        Constructs a path based model from the given path.
        """
        self.dim = dim
        self.hogbin = hogbin
        self.rgbbin = rgbbin
        self.c = c

        logger.info("Extracting features from path")
        positives, negatives = self.extractpath(images, givens, dim,
                                                hogbin, rgbbin, bgskip,
                                                bgsize)

        # svm.sanity(positives, negatives) # uncomment when debugging

        logger.info("Learning weights for path with {0} foregrounds and "
                    "{1} backgrounds".format(len(positives), len(negatives)))
        svm.sanity(negatives, positives)
        model = svm.train(negatives, positives, c = c)
        self.weights, self.bias = model.weights, model.bias

        logger.info("Weights learned with bias {0}".format(self.bias))

        self.realprior = realprior
        self.realpriorweight = realpriorweight
        if self.realprior and not self.realprior.built:
            self.realprior.build(givens, forcescore = 1)

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
        negativesperframe = int(bgsize / len(givens))
        for given in givens:
            logger.debug("Extracting features from "
                         "frame {0}".format(given.frame))
            wr = float(dim[0]) / given.width
            hr = float(dim[1]) / given.height
            im = images[given.frame]
            im = im.resize((int(im.size[0]*wr), int(im.size[1]*hr)), 2)
            imw, imh = im.size
            mapped = given.transform(wr, hr)
            mapped.xbr = mapped.xtl + dim[0]
            mapped.ybr = mapped.ytl + dim[1]

            # positives
            xtl, ytl, xbr, ybr = mapped[0:4]
            patch = im.crop((xtl-hogbin*2, ytl-hogbin*2,
                             xbr+hogbin*2, ybr+hogbin*2))
            hogpatch = features.hog(patch, hogbin)[1:-1,1:-1,:]
            hogpatch = hogpatch.flatten()
            patch = im.crop((xtl, ytl, xbr, ybr))
            rgbpatch = features.rgbmean(patch)
            positives.append(numpy.append(hogpatch, rgbpatch))

            # we shift the positives patch by a couple of pixels to have a 
            # larger training set
#            xtl, ytl, xbr, ybr = mapped[0:4]
#            for horzoffset in range(-hogbin+1, hogbin):
#                for vertoffset in range(-hogbin+1, hogbin):
#                    patch = im.crop((xtl-hogbin+horzoffset,
#                                     ytl-hogbin+vertoffset,
#                                     xbr+hogbin+horzoffset,
#                                     ybr+hogbin+vertoffset))
#                    hogpatch = features.hog(patch, hogbin)[1:-1,1:-1].flatten()
#                    rgbpatch = features.rgbhist(patch, self.rgbbin).flatten()
#                    positives.append(numpy.append(hogpatch, rgbpatch))

            logger.debug("Extracting negatives")

            # negatives
            hogim = features.hog(im, hogbin)
            hogim = features.hogpad(hogim)
            framenegatives = []
            for i from 0 <= i < imw-dimw by bgskip:
                for j from 0 <= j < imh-dimh by bgskip:
                    if annotations.Box(i/wr, j/hr, (i + dimw)/wr,
                        (j + dimh)/hr).percentoverlap(given) < 0.5:
                        framenegatives.append((i, j))

            logger.debug("Sampling negatives")
            if len(framenegatives) > negativesperframe:
                framenegatives = random.sample(framenegatives,
                                               negativesperframe)

            for i, j in framenegatives: 
                hogj = j/hogbin 
                hogjs = (j+dimh)/hogbin 
                hogi = i/hogbin 
                hogis = (i+dimw)/hogbin 

                hogpatch = hogim[hogj:hogjs, hogi:hogis, :]
                hogpatch = hogpatch.flatten()

                rgbpatchregion = (i, j, i+dimw, j+dimh)
                rgbpatch = features.rgbmean(im.crop(rgbpatchregion))
                rgbpatch = rgbpatch.flatten()

                negatives.append(numpy.append(hogpatch, rgbpatch))
        return positives, negatives

    def hogweights(self):
        """
        Returns the SVM weights just for the HOG features.
        """
        x = self.dim[0]/self.hogbin
        y = self.dim[1]/self.hogbin
        return self.weights[0:x*y*13].reshape((y, x, 13))

    def rgbweights(self):
        """
        Returns the SVM weights just for the RGB features.
        """
        x = self.dim[0]/self.hogbin
        y = self.dim[1]/self.hogbin
        return self.weights[x*y*13:]

    def scoreframe(self, image, size, frame = None):
        # resize image to so box has 'dim' in the resized space
        cdef int i, j, width, height, dim0, dim1
        cdef double rpw = self.realpriorweight
        cdef double probsum = 0
        cdef numpy.ndarray[ndim=2, dtype=numpy.double_t] proj

        width, height = image.size
        dim0, dim1 = self.dim
        wr = self.dim[0] / <double>(size[0])
        hr = self.dim[1] / <double>(size[1])
        rimage = image.resize((int(ceil(width * wr)), int(ceil(height * hr))), 2)

        cost = convolution.hogrgbmean(rimage, self.dim,
                                self.hogweights(),
                                self.rgbweights(),
                                hogbin = self.hogbin)

        if self.realprior and frame and self.realprior.hasprojection(frame):
            proj = self.realprior.scorelocations(frame)
            proj = convolution.sumprob(proj, self.dim)
            for i from 0 <= i < <int>(width * wr - dim0):
                for j from 0 <= j < <int>(height * hr - dim1):
                    probsum = proj[<int>(i / wr), <int>(j / hr)]
                    probsum += proj[<int>((i + dim0) / wr), <int>((j + dim1) / hr)]
                    probsum -= proj[<int>(i / wr), <int>((j + dim1) / hr)]
                    probsum -= proj[<int>((i + dim0) / wr), <int>(j / hr)]
                    cost[i, j] += cost[i, j] - rpw * probsum
        return cost
