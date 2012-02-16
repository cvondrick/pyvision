import cStringIO

def red(value, lower, upper):
    red = int((value - lower) / (upper - lower) * 255)
    return red, 0, 0

def filtertrue(value, lower, upper):
    return True

def filterlower(value, lower, upper):
    return value > lower

def filterupper(value, lower, upper):
    return value < upper

def write(outfile, data, colormap = red, condition = filtertrue, bounds = None):
    f = cStringIO.StringIO()
    lower = data.min()
    upper = data.max()
    count = 0

    xs, ys, zs = data.shape

    if not bounds:
        bounds = (0, xs), (0, ys), (0, zs)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    for x in range(0, xs):
        xi = float(x) / xs * (xmax - xmin) + xmin
        for y in range(0, ys):
            yi = float(y) / ys * (ymax - ymin) + ymin
            for z in range(0, zs):
                zi = float(z) / zs * (zmax - zmin) + zmin
                value = data[x, y, z]
                if not condition(value, lower, upper):
                    continue
                count += 1
                r, g, b = colormap(value, lower, upper)
                f.write("{0} {1} {2} {3} {4} {5}\n".format(xi, yi, zi,
                                                           r, g, b))
    outfile.write("ply\n")
    outfile.write("format ascii 1.0\n")
    outfile.write("element vertex {0}\n".format(count))
    outfile.write("property float x\n")
    outfile.write("property float y\n")
    outfile.write("property float z\n")
    outfile.write("property uchar diffuse_red\n")
    outfile.write("property uchar diffuse_green\n")
    outfile.write("property uchar diffuse_blue\n")
    outfile.write("end_header\n")
    outfile.write(f.getvalue())
    f.close()
