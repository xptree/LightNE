ifeq (, $(shell which jemalloc-config))
JEMALLOC = else
JEMALLOCLD = $(shell jemalloc-config --libdir)
JEMALLOC = -L$(JEMALLOCLD) -ljemalloc
endif

# Always compile with LONG (note that timings on small graphs may be a bit
# faster w/o this flag).
INTT = -DLONG

ifdef EDGELONG
INTE = -DEDGELONG
endif

# If you install intel mkl directly, then you need to set something like:
INCLUDE_DIRS = -I../ -I/opt/intel/mkl/include
LINK_DIRS = -L"/opt/intel/mkl/lib/intel64"

OPT = -O3 -DNDEBUG #-g
#OPT = -O0 -g

CFLAGS = $(INCLUDE_DIRS) -I../ligra -I./mklredsvd -mcx16 -ldl -std=c++17 -march=native -Wall $(OPT) $(INTT) $(INTE) -Wl,--no-as-needed -DMKL_ILP64 -DAMORTIZEDPD $(CONCEPTS) -DUSEMALLOC $(LINK_DIRS) -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -liomp5 -lpthread -lm


OMPFLAGS = -DOPENMP -fopenmp
CILKFLAGS = -DCILK -fcilkplus
HGFLAGS = -DHOMEGROWN -pthread

ifdef CLANG
CC = clang++
PFLAGS = $(CILKFLAGS)
else ifdef CILK
CC = g++
PFLAGS = $(CILKFLAGS)
else ifdef OPENMP
CC = g++
PFLAGS = $(OMPFLAGS)
else ifdef HOMEGROWN
CC = g++
PFLAGS = $(HGFLAGS)
else ifdef SERIAL
CC = g++
PFLAGS =
else # default is homegrown
CC = g++
PFLAGS = $(HGFLAGS)
endif

ALL= LightNE

all: $(ALL)

% : %.cc
	$(CC) $(CFLAGS) $(PFLAGS) mklredsvd/mklhelper.h mklredsvd/util.cc -o $@ $<

.PHONY : clean

clean :
	rm -f *.o $(ALL)
