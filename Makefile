CC = mpicc
CFLAGS = -O3 -march=armv8-a+sve -mtune=native -Wall

all: sve_bw_test

sve_bw_test: sve_bw_test.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f sve_bw_test

run: sve_bw_test
	mpirun -np 4 ./sve_bw_test

.PHONY: all clean run