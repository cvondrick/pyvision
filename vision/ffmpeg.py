import os
import shutil
import random
from PIL import Image

def which(program):
    """Function to check for presence of executable/installed program
       Used for checking presense of ffmpeg/avconv"""
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

class extract(object):
    def __init__(self, path, fps = None, size = None):
        self.key = int(random.random() * 1000000000)
        self.key = "pyvision-ffmpeg-{0}".format(self.key)
        self.output = "/tmp/{0}".format(self.key)
        self.path = path
        try:
            os.makedirs(self.output) 
        except:
            pass

        if which("ffmpeg") is not None:
            cmd = "ffmpeg -i {0} -b 10000k".format(path)
        else:
            cmd = "avconv -i {0} -b 10000k".format(path)
        if fps:
            cmd = "{0} -r {1}".format(cmd, int(fps))
        if size:
            w, h = size
            cmd = "{0} -s {1}x{2}".format(cmd, int(w), int(h))
        os.system("{0} {1}/%d.jpg".format(cmd, self.output))

    def __del__(self):
        if self.output:
            shutil.rmtree(self.output)

    def __getitem__(self, k):
        return Image.open(self.getframepath(k))

    def getframepath(self, k):
        return "{0}/{1}.jpg".format(self.output, k+1)

    def __len__(self):
        f = 1
        while True:
            if not os.path.exists(self.getframepath(f)):
                return f
            f += 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
