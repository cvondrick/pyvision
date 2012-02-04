import os
import numpy

def read_patches(data):
    """
    Reads a PMVS patch file, typically called something like option-0000.patch. 
    Returns an array of patches.
    """
    if data.readline().strip() != "PATCHES":
        raise RuntimeError("expected PATCHES header")

    patches = []
    for patch in range(int(data.readline().strip())):
        line = data.readline().strip()
        while not line:
            line = data.readline().strip()
        if line != "PATCHS":
            raise RuntimeError("expected PATCH for patch {0}".format(patch))

        line = data.readline().strip()
        realcoords = [float(x) for x in line.split()]

        line = data.readline().strip()
        normal = [float(x) for x in line.split()]

        line = data.readline().strip()
        score, _, _ = [float(x) for x in line.split()]

        data.readline()
        line = data.readline()
        visibles = [int(x) for x in line.split()]

        data.readline()
        line = data.readline().strip()
        disagrees = [int(x) for x in line.split()]

        patches.append(Patch(realcoords, normal, score, visibles, disagrees))

    return patches

class Patch(object):
    def __init__(self, realcoords, normal, score, visibles, disagrees):
        self.realcoords = numpy.array(realcoords)
        self.normal = numpy.array(normal)
        self.score = score
        self.visibles = visibles
        self.disagrees = disagrees

    def __repr__(self):
        return "Patch%s" % str((self.realcoords, self.normal, self.score,
                                self.visibles, self.disagrees))

    def project(self, projection):
        a = numpy.dot(projection, self.realcoords)
        return a / a[2]

    def projectall(self, projections, disagrees = True):
        if disagrees:
            use = self.visibles + self.disagrees
        else:
            use = self.visibles
        mapping = {}
        for point in self.visibles + self.disagrees:
            mapping[point] = self.project(projections[point])
        return mapping

def read_projections(root):
    """
    Reads in all the projection information stored inside txt files.
    """
    projections = {}
    for file in os.listdir(root):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(root, file)) as data:
            if data.readline().strip() != "CONTOUR":
                raise RuntimeError("expceted CONTOUR header")
            read = lambda x: [float(y) for y in x.readline().strip().split()]
            m = numpy.array([read(data), read(data), read(data)])
            name, _ = file.split(".")
            projections[float(name)] = m
    return projections

if __name__ == "__main__":
    p = read_patches(open("option-0000.patch"))
    proj = read_projections("/csail/vision-videolabelme/databases/video_adapt/home_ac_a/frames/0/bundler/pmvs/txt")

    print p[0].project(proj[0])
