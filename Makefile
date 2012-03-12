pyvision :
	python setup.py build

clean :
	rm -rfv build
	rm -fv vision/*.c
	rm -fv vision/*.cpp
	rm -fv vision/*.html
	rm -fv vision/*.so
	rm -fv vision/track/*.c
	rm -fv vision/track/*.html
	rm -fv vision/track/*.so
	rm -fv vision/alearn/*.c
	rm -fv vision/alearn/*.html
	rm -fv vision/alearn/*.so
	rm -fv vision/ffmpeg/extract.cpp
	rm -fv vision/ffmpeg/extract.so
	rm -fv vision/ffmpeg/extract.html
	$(MAKE) -C vision/liblinear clean
