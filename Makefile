#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set et ft=make ts=8 sw=8 fenc=utf-8 :vi ──────────────────────┘

CC = cosmocc
CXX = cosmoc++
COPTS = -g -O3 -Wall -mcosmo -pthread -fopenmp #-fsanitize=address -fsanitize=undefined
TARGET_ARCH = -Xaarch64-march=armv8.6-a
LDFLAGS = -pthread -fopenmp
LDLIBS = -lm

CC = gcc
CXX = g++
COPTS = -g -O3 -Wall -pthread -fopenmp #-fsanitize=address -fsanitize=undefined
LDFLAGS = -fopenmp
TARGET_ARCH = -march=native
#TARGET_ARCH = -march=skylake
CPPFLAGS = -Wframe-larger-than=65536 -Walloca-larger-than=65536
LDLIBS = -lm

# mkl (gnu threads)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
export MKLROOT = /opt/intel/oneapi/mkl/2024.1
MKL_COPTS = -fopenmp -pthread -DMKL_ILP64 -I$(MKLROOT)/include
MKL_LDLIBS = -Wl,--start-group $(MKLROOT)/lib/libmkl_intel_ilp64.a $(MKLROOT)/lib/libmkl_gnu_thread.a $(MKLROOT)/lib/libmkl_core.a -Wl,--end-group
COPTS = -g -Wall -O3 $(MKL_COPTS) $(COSMO_COPTS) #-fsanitize=address -fsanitize=undefined
LDLIBS = $(MKL_LDLIBS) -lm

# # single-threaded mkl
# # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# export MKLROOT = /opt/intel/oneapi/mkl/2024.1
# MKL_COPTS = -DMKL_ILP64 -I$(MKLROOT)/include
# MKL_LDLIBS = -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group
# COPTS = -g -Wall -O3 -ffast-math -Wno-unknown-pragmas $(MKL_COPTS) $(COSMO_COPTS) #-fsanitize=address -fsanitize=undefined
# LDLIBS = $(MKL_LDLIBS) -lm

.PHONY: o/$(MODE)/
.PRECIOUS: o/$(MODE)/%
o/$(MODE)/: o/$(MODE)/beats-mkl-2048.runs

.PHONY: clean
clean:; rm -rf o

o/$(MODE)/%.o: %.cc linalg.h Makefile
	@mkdir -p $(@D)
	$(CXX) $(COPTS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -o $@ $<

o/$(MODE)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) $(COPTS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -o $@ $<

o/$(MODE)/%.o: %.S Makefile
	@mkdir -p $(@D)
	$(CC) $(COPTS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -o $@ $<

o/$(MODE)/%: o/$(MODE)/%.o o/$(MODE)/gold.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

o/$(MODE)/%: o/$(MODE)/%.o o/$(MODE)/gold.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

o/$(MODE)/%.runs: o/$(MODE)/%
	$<

o/$(MODE)/%.runs: o/$(MODE)/%
	$<
