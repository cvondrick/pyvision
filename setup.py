from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

print "building ffmpeg/_extract.o"
os.system("g++ -D__STDC_CONSTANT_MACROS -c -O3 -fPIC vision/ffmpeg/_extract.c -o vision/ffmpeg/_extract.o")

ext_modules = [
    Extension("boundingboxes", ["vision/boundingboxes.pyx", "vision/boundingboxes.pxd"]),
    Extension("features", ["vision/features.pyx"]),
    Extension("svm", ["vision/svm.pyx"]),
    Extension("model", ["vision/model.pyx"]),
    Extension("convolution", ["vision/convolution.pyx"]),
    Extension("track.standard", ["vision/track/standard.pyx"]),
    Extension("track.alearn", ["vision/track/alearn.pyx"]),
    Extension("ffmpeg.extract", sources =
        ["vision/ffmpeg/extract.pyx", 
        "vision/ffmpeg/extract.pxd"],
        include_dirs = [os.getcwd() + '/vision/ffmpeg/'],
        library_dirs = [os.getcwd() + '/vision/ffmpeg/'],
        libraries = ['avformat', 'avcodec', 'avutil', 'swscale'],
        extra_objects = [os.getcwd() + '/vision/ffmpeg/_extract.o'],
        language = 'c++')
    ]

setup(
    name = "pyvision",
    author = "Carl Vondrick",
    author_email = "cvondric@ics.uci.edu",
    version = "0.1.0",
    packages = ['vision', 'vision.track', 'vision.ffmpeg'],
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    ext_package = 'vision'
)
