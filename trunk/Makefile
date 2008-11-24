CC = g++
EXEC = nb4it

Main: *.cpp *.h
	$(CC) -Wall -c -ggdb *.cpp
	$(CC) -o $(EXEC) *.o

clean:
	rm -f *.o
	rm -f *~
	rm -fr doc/
	rm -f $(EXEC)

doc: *.cpp
	doxygen Doxyfile

backup:
	cd ..
	tar -cvvzf ../prj2src.tar.gz ../src
