from vision.model import PathModel
from vision import convolution
from vision import *
from vision.visualize import highlight_box
from vision import features
from vision.toymaker import *
from PIL import Image
import numpy
import pylab
from scipy.io import savemat as savematlab

import logging
logging.basicConfig(level = logging.DEBUG)

b = Box(498, 336, 498 + 131, 336 + 68, 200)
#b = Box(128, 310, 128 + 91, 310 + 55, 200)
b = Box(94, 256, 94 + 61, 257 + 40, 200)
stop = 300

frames = frameiterator("/scratch/virat/frames/VIRAT_S_040302_01_001240_001586")

#frames = Geppetto((500,500))
#frames.add(Rectangle((0,0), size = (250, 500), color = "black"))
#path = Rectangle((200,200), color = "gray")
#frames.add(path)
##frames.add(Rectangle((400,300), color = "blue"))
##frames.add(Rectangle((200,300), color = "blue"))
#b = path[0]
#stop = 0

model  = PathModel(frames, [b], hogbin = 4, c = 1)

wr = model.dim[0] / float(b.width)
hr = model.dim[1] / float(b.height)

image = frames[stop]
image = image.resize((int(wr * image.size[0]), int(hr * image.size[1])), 2)
costs = convolution.hogrgbmean(image, model.dim,
                            model.hogweights(), model.rgbweights(),
                            hogbin = model.hogbin)

x, y = numpy.unravel_index(numpy.argmin(costs), costs.shape)
x = int((x) / wr)
y = int((y) / hr)
result = Box(x, y, x + b.width, y + b.height, stop)

f = features.rgbmean(frames[b.frame].crop(b[0:4]))
print f, numpy.dot(f.transpose(), model.rgbweights())
f = features.rgbmean(frames[result.frame].crop(result[0:4]))
print f, numpy.dot(f.transpose(), model.rgbweights())

savematlab(open("weight.mat", "w"), {"w": model.hogweights()}, oned_as="row")

#numpy.set_printoptions(threshold='nan')
print model.hogweights().shape
print model.rgbweights()

print "Given", b
print "Predicted", result

pylab.figure(1)
pylab.subplot(221)
pylab.title("training")
pylab.imshow(numpy.asarray(highlight_box(frames[b.frame], b)))

pylab.subplot(222)
pylab.title("best")
pylab.imshow(numpy.asarray(highlight_box(frames[stop], result)))

pylab.subplot(223)
pylab.title("costs (min = {0}, max = {1})".format(costs.min(), costs.max()))
pylab.set_cmap("gray")
pylab.imshow(costs.transpose())

pylab.subplot(224)
pylab.title("best (resized)")
pylab.imshow(numpy.asarray(highlight_box(image, result.transform(wr, hr))))

pylab.show()
