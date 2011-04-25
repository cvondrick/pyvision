from vision.alearn.marginals import pairwise_quadratic, pairwise_quadratic_1d
import numpy

a = numpy.ones((5,10)) * 100000
a[0, 0] = 0

p, x, y = pairwise_quadratic(a, 1)
print p
print x
print y
