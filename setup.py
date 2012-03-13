# immediately below is stupid hackery for setuptools to work with Cython
import distutils.extension
import distutils.command.build_ext
from distutils.extension import Extension as _Extension
from setuptools import setup 
distutils.extension.Extension = _Extension
distutils.command.build_ext.Extension = _Extension
Extension = _Extension
from Cython.Distutils import build_ext 
# end stupid hackery

# these lines will cause html annotation files to be generated
from Cython.Compiler.Main import default_options as pyrex_default_options
pyrex_default_options['annotate'] = True

import os
import numpy
import sys

print "building liblinear"
os.system("make -C vision/liblinear")

root = os.getcwd() + "/vision/"

ext_modules = [
    Extension("vision.annotations", ["vision/annotations.pyx",
                                     "vision/annotations.pxd"]),
    Extension("vision.features", ["vision/features.pyx"]),
    Extension("vision.model", ["vision/model.pyx"]),
    Extension("vision.convolution", ["vision/convolution.pyx"]),
    Extension("vision.track.standard", ["vision/track/standard.pyx"]),
    Extension("vision.alearn.linear", ["vision/alearn/linear.pyx"]),
    Extension("vision.alearn.marginals", ["vision/alearn/marginals.pyx"]),
    Extension("vision.track.dp", ["vision/track/dp.pyx",
                                  "vision/track/dp.pxd"]),
    Extension("vision.track.realprior", ["vision/track/realprior.pyx"]),
    Extension("vision.track.pairwise", ["vision/track/pairwise.pyx"]),
    Extension("vision.svm", ["vision/svm.pyx"],
        extra_objects = [root + "liblinear/linear.o",
                         root + "liblinear/tron.o",
                         root + "liblinear/blas/blas.a"],
        language = "c++")]

for e in ext_modules:
    e.pyrex_directives = {
        "boundscheck": False,
        "cdivision": True,
        "infer_types": True,
        "embedsignature": True}
#    e.include_dirs.append(".")
    e.extra_compile_args = ["-w"]
    e.include_dirs.append(numpy.get_include())


setup(
    name = "pyvision",
    author = "Carl Vondrick",
    author_email = "cvondric@ics.uci.edu",
    description = "A concise computer vision toolkit",
    license = "MIT",
    version = "0.3.1",
    classifiers = ["Development Status :: 1 - Planning",
                   "Intended Audience :: Developers"],
    packages = ["vision",
                "vision.track",
                "vision.alearn",
                "vision.reporting",
                "vision.reconstruction"],
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    #ext_package = "vision"
)
