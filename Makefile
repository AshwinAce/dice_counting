dice: dice.o
	g++ -std=c++11 -o dice dice.o -l opencv_core -l opencv_imgcodecs -l opencv_imgproc -l opencv_highgui
dice.o: dice.cpp 
	g++ -std=c++11 -c dice.cpp 
.PHONY: clean
clean:
	rm -f dice *.o
