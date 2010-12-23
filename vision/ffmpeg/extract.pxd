#cdef extern from "stdlib.h":
#    ctypedef unsigned long size_t
#    void free(void *ptr)
#    void *malloc(size_t size)
#
#cdef extern from "_extract.h":
#    cdef struct video_stream:
#        void *format_context
#        void *codec_context
#        void *frame_reg
#        void *frame_rgb
#        void *buffer
#        int video_stream
#        int width
#        int height
#
#    int extract_video(char *filename, video_stream *output)
#    int read_frame(video_stream *stream, unsigned char **output)
#    void cleanup_extraction(video_stream *stream)
#
#cdef class extract(object):
#    cdef video_stream *vs
#    cdef int ready
