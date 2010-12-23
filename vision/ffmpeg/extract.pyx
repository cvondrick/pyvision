import numpy
import Image

cimport numpy

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)

cdef extern from "_extract.h":
    cdef struct video_stream:
        void *format_context
        void *codec_context
        void *frame_reg
        void *frame_rgb
        void *buffer
        int video_stream
        int width
        int height

    int extract_video(char *filename, video_stream *output)
    int read_frame(video_stream *stream, unsigned char **output)
    void cleanup_extraction(video_stream *stream)

cdef class extract(object):
    """
    Given a filename, extracts the video into frames. Use as an iterator:

    >>> for id, frame in enumerate(ffmpeg.extract("filename.mpg")):
    ...     frame.save("frame{0}.jpg".format(id))
    """

    cdef video_stream *vs
    cdef int ready

    def __init__(self, filename):
        """
        Builds the frame iterator. Filename must point to a valid video file. 
        Almost every common video codec is supported.
        """
        self.ready = 0
        self.vs = <video_stream*> malloc(sizeof(video_stream))
        cdef int code = extract_video(filename, self.vs)
        if code == 1:
            raise IOError("File not found or not readable")
        elif code == 2:
            raise FFmpegError("Unable to find video stream in file")
        elif code != 0:
            raise FFmpegError("libvideoextract returned error {0}".format(code))
        self.ready = 1

    def __dealloc__(self):
        """
        Perform memory cleanup from libvideoextract.
        """
        if self.ready:
            cleanup_extraction(self.vs)
        free(self.vs)

    def __iter__(self):
        """
        Returns the iterator.
        """
        return self

    def __next__(self):
        """
        Returns a Python Image Library image of the next frame. If no frame 
        exists, then throws a StopIteration exception.
        """
        cdef unsigned char *buffer 
        cdef int code = read_frame(self.vs, &buffer)
        if code == -1:
            raise StopIteration()
        elif code != 0:
            raise FFmpegError("libvideoextract returned error {0}".format(code))
        cdef numpy.ndarray[numpy.uint8_t, ndim=3] matrix
        matrix = numpy.empty((self.vs.height, self.vs.width, 3),
                             dtype=numpy.uint8)
        for i in range(self.vs.width):
            for j in range(self.vs.height):
                matrix[j,i,0] = buffer[j * 3 * self.vs.width + i * 3]
                matrix[j,i,1] = buffer[j * 3 * self.vs.width + i * 3 + 1]
                matrix[j,i,2] = buffer[j * 3 * self.vs.width + i * 3 + 2]
        return Image.fromarray(matrix)

class FFmpegError(RuntimeError):
    """
    An error caused by the FFmpeg library.
    """
    def __init__(self, message):
        RuntimeError.__init__(self, message)
