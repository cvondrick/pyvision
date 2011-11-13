# Automatically reload all of the modules in the current directory
# or its subdirectories every time you enter a command in IPython. 
# Usage: from the IPython toplevel, do
# In [1]: import autoreload
# In [2]: autoreload.enable()
# Author: Andrew Owens
import os, sys, traceback
import IPython.core.ipapi

def relative_module(m):
    return hasattr(m, '__file__') \
           and ((not m.__file__.startswith('/')) \
           or m.__file__.startswith(os.getcwd()))

def reload_all():
    # Reloading __main__ is supposed to throw an error
    # For some reason in ipython I did not get an error and lost the
    # ability to send reload_all() to my ipython shell after making
    # changes. 
    excludes = set(['__main__', 'autoreload'])
    for name, m in sys.modules.iteritems():
        if m and relative_module(m) and (name not in excludes):
            reload(m)

def ipython_autoreload_hook(self):
    try:
        reload_all()
    except:
        traceback.print_exc()
        print 'Reload error. Modules not reloaded'

def enable():
    print 'autoreload enabled'
    ip = IPython.core.ipapi.get()
    ip.set_hook('pre_run_code_hook', ipython_autoreload_hook)
