from vision.alearn.marginals import pairwise_quadratic, pairwise_quadratic_1d
import numpy
import pylab

a = numpy.arange(100, dtype = numpy.double).reshape((10,10))

p, x, y = pairwise_quadratic(a, 1)
print p
print x
print y

pylab.imshow(p)
pylab.show()
