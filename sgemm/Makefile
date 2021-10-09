# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = gcc
OPT = -O3 -Ofast -march=native -mcpu=tsv110 -mtune=tsv110
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT)
LDFLAGS = -Wall 
# mkl is needed for blas implementation
LDLIBS = -lopenblas -lpthread -lm

targets = benchmark-test sgemm-blocked.S benchmark-naive benchmark-blocked benchmark-blas \
	benchmark-blocked-initial benchmark-blocked-loop benchmark-blocked-pack benchmark-blocked-pack-c benchmark-blocked-const \
	benchmark-blocked-two-level benchmark-blocked-small \
	benchmark-blocked-a benchmark-blocked-a-pack-c benchmark-blocked-a-pack-b \
	benchmark-blocked-intrinsics benchmark-blocked-intrinsics-8x8 benchmark-blocked-intrinsics-8x8-load \
	benchmark-blocked-intrinsics-8x8-transpose benchmark-blocked-intrinsics-8x8-tuning benchmark-blocked-intrinsics-8x8-align
objects = benchmark-test.o benchmark.o sgemm-naive.o sgemm-blocked.o sgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-naive : benchmark.o sgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-% : benchmark.o sgemm-blocked-%.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o sgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

%.S : %.o
	objdump -S $^ > $@

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
