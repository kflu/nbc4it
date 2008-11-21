CC = g++

Main: *.cpp
	$(CC) -Wall -c -ggdb *.cpp
	$(CC) *.o

clean:
	rm -f *.o
	rm -f *~
	rm -fr doc/
	rm -f a.out

doc: *.cpp
	doxygen Doxyfile

backup:
	cd ..
	tar -cvvzf ../prj2src.tar.gz ../src
