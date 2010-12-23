struct video_stream
{
    void *format_context; // AVFormatContext
    void *codec_context; // AVCodecContext
    void *frame_reg; // AVFrame
    void *frame_rgb; // AVFrame
    void *buffer; // uint8_t
    int video_stream;
    int width;
    int height;
};

int extract_video(char *filename, struct video_stream *output);
int read_frame(struct video_stream *stream, unsigned char **output);
void cleanup_extraction(struct video_stream *stream);
