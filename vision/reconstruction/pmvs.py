import os
import numpy
import itertools
import logging
from scipy.spatial.kdtree import KDTree

from memo import memo

logger = logging.getLogger("vision.reconstruction.pmvs")

def read(root, start = 0):
    """
    Returns a reconstruction object that has all the important information in it
    that is necessary to reconstruct the scene.
    """
    logger.debug("Read {0}".format(root))
    patches = read_patches(open(find_patch_file(root)), start) 
    projections = read_projections("{0}/txt/".format(root), start)

    return patches, projections

def find_patch_file(root):
    """
    Finds the patch file
    """
    root = os.path.join(root, "models")
    for file in os.listdir(root):
        if file.endswith(".patch"):
            return os.path.join(root, file)
    raise RuntimeError("No patch file found in {0}".format(root))

def read_patches(data, start = 0):
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
        visibles = [int(x) - start for x in line.split()]

        data.readline()
        line = data.readline().strip()
        disagrees = [int(x) - start for x in line.split()]

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
        return projection.project(self.realcoords)

    def projectall(self, projections, disagrees = True):
        if disagrees:
            use = self.visibles + self.disagrees
        else:
            use = self.visibles
        mapping = {}
        for point in self.visibles + self.disagrees:
            mapping[point] = self.project(projections[point])
        return mapping

def get_patch_bounds(patches):
    xmin = ymin = zmin = float("+infinity")
    xmax = ymax = zmax = float("-infinity")

    for patch in patches:
        x, y, z, _ = patch.realcoords

        xmin = min(xmin, x)
        xmax = max(xmax, x)

        ymin = min(ymin, y)
        ymax = max(ymax, y)

        zmin = min(zmin, z)
        zmax = max(zmax, z)

    return ((xmin, xmax), (ymin, ymax), (zmin, zmax))

def read_projections(root, start = 0):
    """
    Reads in all the projection information stored inside txt files.
    """
    frames = open(os.path.join(root, "..", "list.rd.txt"))
    frames = iter(int(x[2:-5]) for x in frames)
    projections = {}
    for file in os.listdir(root):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(root, file)) as data:
            if data.readline().strip() != "CONTOUR":
                raise RuntimeError("expceted CONTOUR header")
            read = lambda x: [float(y) for y in x.readline().strip().split()]
            m = numpy.array([read(data), read(data), read(data)])
            frame = frames.next()
            frame -= start
            projections[frame] = Projection(m, frame)
    return projections

class Projection(object):
    def __init__(self, matrix, frame):
        self.matrix = matrix
        self.frame = frame

    def project(self, real):
        a = numpy.dot(self.matrix, real) 
        return (a / a[2])[0:2]

class RealWorldMap(object):
    def __init__(self, patches, projections):
        self.patches = patches
        self.projections = projections

        self.build()

    def build(self):
        logger.debug("Building maps for conversion between real and image space")

        self.imagetree = {}
        self.imagemap = {}
        self.realtree = KDTree([x.realcoords for x in self.patches])
        self.realmapping = {}

        for num, patch in enumerate(self.patches):
            if num % 1000 == 0:
                logger.debug("Built maps for {0} of {1} patches".format(num, len(self.patches)))

            resp = {}
            for _, projection in self.projections.items():
                inimage = patch.project(projection)
                if projection.id not in self.imagetree:
                    self.imagetree[projection.id] = []
                    self.imagemap[projection.id] = {}
                self.imagetree[projection.id].append(inimage)
                self.imagemap[projection.id][tuple(inimage)] = patch.realcoords
                resp[projection.id] = inimage
                
            self.realmapping[tuple(patch.realcoords)] = resp

        logger.debug("Done building maps for {0} patches".format(len(self.patches)))

        logger.debug("Building image KD tree")
        for key, imagecoords in self.imagetree.items():
            self.imagetree[key] = KDTree(imagecoords)

    def realtoimages(self, coords):
        _, nearestindex = self.realtree.query(coords)
        nearest = self.realtree.data[nearestindex]
        return self.realmapping[tuple(nearest)]

    def realregiontoimages(self, coords):
        _, nearestindices = self.realtree.query(coords)
        resp = {}
        for nearestindex in nearestindices:
            nearest = self.realtree.data[nearestindex]
            points = self.realmapping[tuple(nearest)]
            for k, v in points.iteritems():
                if k not in resp:
                    resp[k] = []
                resp[k].append(v)
        return resp

    def imagetoreal(self, projection, coords):
        try:
            projection = projection.id
        except:
            pass

        _, nearestindex = self.imagetree[projection].query(coords)
        nearest = self.imagetree[projection].data[nearestindex]
        return self.imagemap[projection][tuple(nearest)]

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)

    patches, projections = read("/csail/vision-videolabelme/databases/"
                                "video_adapt/demos/bottle_table/bundler/pmvs")

    print get_patch_bounds(patches)

    mapping = RealWorldMap(patches, projections)

    patch = patches[0]
    print "REAL ="
    print patch.realcoords
    print ""
    a, b = mapping.realtoimages(patch.realcoords).items()[0]
    print "PROJECTION ="
    print b
    print ""
    print "REAL (again) ="
    r = mapping.imagetoreal(a, b)
    print r
