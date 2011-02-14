pyvision :
	python setup.py build

clean :
	rm -rfv build
	rm -fv vision/*.c
	rm -fv vision/*.cpp
	rm -fv vision/*.html
	rm -fv vision/track/*.c
	rm -fv vision/track/*.html
	rm -fv vision/ffmpeg/extract.cpp
	rm -fv vision/ffmpeg/extract.html
	$(MAKE) -C vision/liblinear clean
