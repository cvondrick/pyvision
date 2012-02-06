def read(inputdata):
    """
    Reads in a bundler output file. Returns both the cameras and the points.
    """
    reading_cameras = False
    reading_points = False

    cameras = []
    points = []

    camera_state = 0
    camera_rotation = []
    camera_current = None

    point_state = 0
    point_current = None

    numcameras = None
    numpoints = None

    for line in inputdata:
        if line[0] == "#":
            continue
        if not reading_cameras and not reading_points:
            numcameras, numpoints = (int(x) for x in line.split())
            reading_cameras = True

        elif reading_cameras:
            data = [float(x) for x in line.split()]
            if camera_state == 0:
                focal, radial0, radial1 = data
                camera_current = Camera(len(cameras))
                camera_current.focal = focal
                camera_current.radialdist = (radial0, radial1)
                camera_state = 1
            elif camera_state == 1:
                camera_rotation.append(data)
                if len(camera_rotation) == 3:
                    camera_current.rotation = camera_rotation
                    camera_rotation = []
                    camera_state = 2
            elif camera_state == 2:
                camera_current.translation =  data
                camera_state = 0
                cameras.append(camera_current)
            if len(cameras) == numcameras:
                reading_cameras = False
                reading_points = True

        elif reading_points:
            if point_state == 0:
                point_current = Point()
                point_current.position = [float(x) for x in line.split()]
                point_state = 1
            elif point_state == 1:
                point_current.color = [int(x) for x in line.split()]
                point_state = 2
            elif point_state == 2:
                data = line.split()
                views = []
                viewcameras = [int(x) for x in data[1::4]]
                keys = [int(x) for x in data[2::4]]
                xs = [float(x) for x in data[3::4]]
                ys = [float(x) for x in data[4::4]]
                for camera, key, x, y in zip(viewcameras, keys, xs, ys):
                    point_current.views.append(PointView(cameras[camera], key, x, y))
                point_state = 0
                points.append(point_current)

    assert point_state == 0
    assert camera_state == 0
    assert reading_cameras == False
    assert reading_points == True

    return cameras, points

class Camera(object):
    def __init__(self, id, focal = None, radialdist = None,
                 rotation = None, translation = None):
        self.id = id
        self.focal = focal
        self.radialdist = radialdist
        self.rotation = rotation
        self.translation = translation

    def __repr__(self):
        return "Camera%s" % str((self.id, self.focal, self.radialdist,
                                 self.rotation, self.translation))

class Point(object):
    def __init__(self, position = None, color = None, views = None):
        if views is None:
            views = []
        self.position = position
        self.color = color
        self.views = views

    def __repr__(self):
        return "Point%s" % str((self.position, self.color, self.views))

class PointView(object):
    def __init__(self, camera = None, key = None, x = None, y = None):
        self.camera = camera
        self.key = key
        self.x = x
        self.y = y

    def __repr__(self):
        return "PointView%s" % str((self.camera, self.key, self.x, self.y))
