from vision.model import PathModel
from vision import convolution
from vision import *
from vision.visualize import highlight_box
from vision import features
from vision.toymaker import *
import Image
import numpy
import pylab
from scipy.io import savemat as savematlab

import logging
logging.basicConfig(level = logging.DEBUG)

b = Box(329, 286, 329 + 100, 286 + 48, 0)
stop = 10

frames = frameiterator("/scratch/virat/frames/VIRAT_S_050201_00_000012_000116")

#frames = Geppetto()
#path = Rectangle((100,100), color = "gray")
#path.linear((200,200), 10)
#frames.add(path)
#frames.add(Rectangle((300,400), color = "gray"))
##frames.add(Rectangle((400,300), color = "blue"))
##frames.add(Rectangle((200,300), color = "blue"))
#frames.add(Rectangle((300,300)))
#b = path[0]
#
model  = PathModel(frames, [b], hogbin = 8, c = 100)

wr = model.dim[0] / float(b.width)
hr = model.dim[1] / float(b.height)

image = frames[stop]
image = image.resize((int(wr * image.size[0]), int(hr * image.size[1])), 2)
costs = convolution.hogrgbmean(image, model.dim,
                            model.hogweights(), model.rgbweights(),
                            hogbin = model.hogbin)

x, y = numpy.unravel_index(numpy.argmin(costs), costs.shape)
x = int(x / wr)
y = int(y / hr)
result = Box(x, y, x + b.width, y + b.height, stop)

#numpy.set_printoptions(threshold='nan')
#print model.hogweights().shape
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
