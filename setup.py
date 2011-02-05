# immediately below is stupid hackery for setuptools to work with Cython
import distutils.extension
from distutils.extension import Extension as _Extension
from setuptools import setup
Extension = distutils.extension.Extension = _Extension
from Cython.Distutils import build_ext
# end stupid hackery

import os

print "building ffmpeg/_extract.o"
os.system("g++ -Wno-deprecated-declarations -D__STDC_CONSTANT_MACROS -c -O3 -fPIC vision/ffmpeg/_extract.c -o vision/ffmpeg/_extract.o")

print "building liblinear"
os.system("make -C vision/liblinear")

root = os.getcwd() + "/vision/"

ext_modules = [
    Extension("annotations", ["vision/annotations.pyx", "vision/annotations.pxd"]),
    Extension("features", ["vision/features.pyx"]),
    Extension("model", ["vision/model.pyx"]),
    Extension("convolution", ["vision/convolution.pyx"]),
    Extension("track.standard", ["vision/track/standard.pyx"]),
    Extension("track.alearn", ["vision/track/alearn.pyx"]),
    Extension("svm", ["vision/svm.pyx"],
        extra_objects = [root + "liblinear/linear.o",
                         root + "liblinear/tron.o",
                         root + "liblinear/blas/blas.a"],
        language = "c++"),
    Extension("ffmpeg.extract",
        sources = ["vision/ffmpeg/extract.pyx"],
        include_dirs = [root + 'ffmpeg/'],
        library_dirs = [root + 'ffmpeg/'],
        libraries = ['avformat', 'avcodec', 'avutil', 'swscale'],
        extra_objects = [root + 'ffmpeg/_extract.o'],
        language = 'c++')
    ]

for e in ext_modules:
    e.pyrex_directives = {
        "boundscheck": False,
        "cdivision": True,
        "infer_types": True}
    e.include_dirs.append(".")
    e.extra_compile_args = ['-w']

setup(
    name = "pyvision",
    author = "Carl Vondrick",
    author_email = "cvondric@ics.uci.edu",
    description = "A concise computer vision toolkit",
    license = "MIT",
    version = "0.0.3",
    classifiers = ['Development Status :: 1 - Planning',
                   'Intended Audience :: Developers'],
    packages = ['vision', 'vision.track', 'vision.ffmpeg'],
    package_dir = {'vision': 'vision'},
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    ext_package = 'vision'
)
