import ImageDraw
import itertools
import random

defaultwidth = 2
colors = ["#FF00FF",
          "#FF0000",
          "#FF8000",
          "#FFD100",
          "#008000",
          "#0080FF",
          "#0000FF",
          "#000080",
          "#800080"]

def highlight_box(image, box, color = colors[0], width = defaultwidth):
    """
    Highlights the bounding box on the given image.
    """
    draw = ImageDraw.Draw(image)
    for i in range(width):
        draw.rectangle((box[0] + i, box[1] + i, box[2] - i, box[3] - i),
                       outline=color)
    return image

def highlight_boxes(image, boxes, colors = colors, width = defaultwidth):
    """
    Highlights an iterable of boxes.
    """
    for box, color in zip(boxes, itertools.cycle(colors)):
        highlight_box(image, box, color, width)
    return image

def highlight_path(images, path, color = colors[0], width = defaultwidth):
    """
    Highlights a path across many images. The images must be indexable
    by the frame. Produces a generator.
    """
    for box in path:
        try:
            lost = box.lost
        except:
            lost = False
        if not lost:
            image = images[box.frame]
            highlight_box(image, box, color, width)
            yield image, box.frame

def highlight_paths(images, paths, colors = colors, width = defaultwidth):
    """
    Highlights multiple paths across many images. The images must be indexable
    by the frame. Produces a generator.
    """
    boxmap = {}
    paths = zip(paths, itertools.cycle(colors))

    for path, color in paths:
        for box in path:
            if box.frame not in boxmap:
                boxmap[box.frame] = [(box, color)]
            else:
                boxmap[box.frame].append((box, color))

    for frame, boxes in sorted(boxmap.items()):
        im = images[frame]
        for box, color in boxes:
            try:
                lost = box.lost
            except:
                lost = False
            if not lost:
                highlight_box(im, box, color, width)
        yield im, frame

def save(images, output):
    """
    Saves images produced by the path iterators.
    """
    for image, frame in images:
        image.save(output(frame))
