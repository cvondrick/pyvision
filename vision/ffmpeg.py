import os
import shutil
import random
import Image

class extract(object):
    def __init__(self, path):
        self.key = int(random.random() * 1000000000)
        self.key = "pyvision-ffmpeg-{0}".format(self.key)
        self.output = "/tmp/{0}".format(self.key)
        self.path = path
        try:
            os.makedirs(self.output) 
        except:
            pass
        os.system("ffmpeg -i {0} -b 10000k {1}/%d.jpg".format(path, self.output))

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
