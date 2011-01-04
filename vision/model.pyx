import svm
import annotations
import features
import random
import logging
import numpy

logger = logging.getLogger("model")

cimport numpy

class PathModel(object):
    """
    A model that learns a linear SVM weight vector based off a path.

    The model extracts positve examples from the given path and negative examples
    from sliding windows that do not overlap with a given example. We extract
    both HOG features and RGB features. 

    For fast scoring, use the corresponding convolution.pyx routine.
    """

    def __init__(self, images, givens, dim = (40,40), hogbin = 8, rgbbin = 8, bgskip = 4, bgsize = 5e4):
        """
        Constructs a path based model from the given path.
        """
        self.dim = dim
        self.hogbin = hogbin
        self.rgbbin = rgbbin

        logger.info("Extracting features from path")
        positives, negatives = self.__extractpath(images, givens, dim, hogbin, rgbbin, bgskip, bgsize)

        # svm.sanity(positives, negatives) # uncomment when debugging

        logger.info("Learning weights for path with {0} foregrounds and {1} backgrounds".format(len(positives), len(negatives)))
        model = svm.train(negatives, positives)
        self.weights, self.bias = model.weights, model.bias

        logger.info("Weights learned with bias {0}".format(self.bias))

    def __extractpath(self, images, givens, dim, hogbin, rgbbin, bgskip, bgsize):
        """
        Extracts features from a path.
        """
        positives = []
        negatives = []
        for given in givens:
            logger.debug("Extracting features from frame {0}".format(given.frame))
            wr = float(dim[0]) / given.get_width()
            hr = float(dim[1]) / given.get_height()
            im = images[given.frame]
            im = im.resize((int(im.size[0]*wr), int(im.size[1]*hr)))
            mapped = given.transform(wr, hr)

            # positives
            xtl, ytl, xbr, ybr = mapped.position()
            patch = im.crop((xtl-hogbin, ytl-hogbin, xbr+hogbin, ybr+hogbin))
            hogpatch = features.hog(patch, hogbin)[1:-1,1:-1].flatten()
            rgbpatch = features.rgbhist(patch, rgbbin).flatten()
            positives.append(numpy.append(hogpatch, rgbpatch))

            # negatives
            hogim = features.hog(im, hogbin)
            for i in range(0, im.size[0]-dim[0], bgskip):
                for j in range(0, im.size[1]-dim[1], bgskip):
                    if not annotations.Box(i/wr, j/hr, (i + dim[0])/wr, (j + dim[1])/hr).intersects(given):
                        hogpatch = hogim[j/hogbin:(j+dim[1])/hogbin-2, i/hogbin:(i+dim[0])/hogbin-2, :].flatten()
                        rgbpatch = features.rgbhist(im.crop((i, j, i+dim[0], j+dim[1]))).flatten()
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
