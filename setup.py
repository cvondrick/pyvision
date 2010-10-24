from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("boundingboxes", ["vision/boundingboxes.pyx", "vision/boundingboxes.pxd"]),
    Extension("features", ["vision/features.pyx"]),
    Extension("svm", ["vision/svm.pyx"]),
    Extension("model", ["vision/model.pyx"]),
    Extension("convolution", ["vision/convolution.pyx"]),
    Extension("track.standard", ["vision/track/standard.pyx"]),
    Extension("track.alearn", ["vision/track/alearn.pyx"]),
    Extension("ffmpeg", ["vision/ffmpeg.pyx"]),
    ]

setup(
    name = "pyvision",
    author = "Carl Vondrick",
    author_email = "cvondric@ics.uci.edu",
    version = "0.1.0",
    packages = ['vision', 'vision.track'],
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    ext_package = 'vision'
)
