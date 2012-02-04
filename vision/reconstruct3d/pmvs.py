def read_patches(data):
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
        self.realcoords = realcoords
        self.normal = normal
        self.score = score
        self.visibles = visibles
        self.disagrees = disagrees

    def __repr__(self):
        return "Patch%s" % str((self.realcoords, self.normal, self.score,
                                self.visibles, self.disagrees))

if __name__ == "__main__":
    p = read_patches(open("option-0000.patch"))
