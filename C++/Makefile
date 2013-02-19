CXX		= clang++
CFLAGS	= -I /opt/local/include -Wall -O0
DEBUG	= -g
OBJS	= main.o svc.o svm.o
VPATH	= src/

classifier: $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $(OBJS) $(DEBUG)

.cc.o:
	$(CXX) $(CFLAGS) -c $< $(DEBUG)

.PHONY: clean
clean:
	rm *.o classifier


