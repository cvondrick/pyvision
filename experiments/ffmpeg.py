from vision import ffmpeg

f = ffmpeg.extract("/csail/vision-videolabelme/databases/video_adapt/demos/bottle_table.mov",
fps = None)
i = iter(f)

print len(f)

for t in f:
    print t.size
