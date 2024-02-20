#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set et ft=make ts=8 sw=8 fenc=utf-8 :vi ──────────────────────┘

COPTS = -g -O3 -ffast-math -Wall -march=native -pthread -fopenmp #-fsanitize=address -fsanitize=undefined
LDFLAGS = -pthread -fopenmp
LDLIBS = -lm

# # single threaded
# COPTS = -g -O3 -ffast-math -Wall -march=native #-fsanitize=address -fsanitize=undefined
# CPPFLAGS = -Wno-unknown-pragmas

# # cosmo flags (used for portable builds)
# COPTS += -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -mno-red-zone -fno-optimize-sibling-calls -fno-schedule-insns2 -fpatchable-function-entry=18,16 -fno-inline-functions-called-once

# # clang
# CC = clang
# CXX = clang++
# COPTS = -g -O3 -ffast-math -Wall -march=native -pthread -fopenmp #-fsanitize=address -fsanitize=undefined
# CXXFLAGS = -std=gnu++23 -Wno-vla-cxx-extension
# CFLAGS = -std=gnu23
# LDLIBS = -lm

# # mkl (gnu threads)
# # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# export MKLROOT = /opt/intel/oneapi/mkl/2024.0
# MKL_COPTS = -fopenmp -pthread -DMKL_ILP64 -I$(MKLROOT)/include
# MKL_LDLIBS = -Wl,--start-group $(MKLROOT)/lib/libmkl_intel_ilp64.a $(MKLROOT)/lib/libmkl_gnu_thread.a $(MKLROOT)/lib/libmkl_core.a -Wl,--end-group
# COPTS = -g -Wall -O3 -march=native -ffast-math $(MKL_COPTS) #-fsanitize=address -fsanitize=undefined
# LDLIBS = $(MKL_LDLIBS) -lm

# # mkl (single threaded)
# # https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# export MKLROOT = /opt/intel/oneapi/mkl/2024.0
# MKL_COPTS = -DMKL_ILP64 -I$(MKLROOT)/include
# MKL_LDLIBS = -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group
# COPTS = -g -Wall -O3 -march=native -ffast-math $(MKL_COPTS) #-fsanitize=address -fsanitize=undefined
# CPPFLAGS = -Wno-unknown-pragmas
# LDLIBS = $(MKL_LDLIBS) -lm

# # atlas (single threaded)
# ATLAS_COPTS = -I/usr/include/x86_64-linux-gnu
# ATLAS_LDLIBS = -L/usr/lib/x86_64-linux-gnu/atlas -lblas
# COPTS = -g -Wall -O3 -march=native -ffast-math $(ATLAS_COPTS) #-fsanitize=address -fsanitize=undefined
# CPPFLAGS = -Wno-unknown-pragmas
# LDLIBS = $(ATLAS_LDLIBS) -lm

# # blis (single threaded)
# COPTS = -g -Wall -O3 -ffast-math -march=native #-fsanitize=address -fsanitize=undefined
# CPPFLAGS = -Wno-unknown-pragmas
# LDLIBS = -lblis -lm

# # blis (multi threaded)
# COPTS = -g -fopenmp -Wall -O3 -ffast-math -march=native #-fsanitize=address -fsanitize=undefined
# LDLIBS = -lblis -lm

.PHONY: o/$(MODE)/
.PRECIOUS: o/$(MODE)/%
o/$(MODE)/: o/$(MODE)/mope.runs

.PHONY: clean
clean:; rm -rf o

o/$(MODE)/gold.o: private COPTS = -g -O3 -march=native -fopenmp

o/$(MODE)/%.o: %.cc linalg.h Makefile
	@mkdir -p $(@D)
	$(CXX) $(COPTS) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

o/$(MODE)/%.o: %.c Makefile
	@mkdir -p $(@D)
	$(CC) $(COPTS) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

o/$(MODE)/%.o: %.S Makefile
	@mkdir -p $(@D)
	$(CC) $(COPTS) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

o/$(MODE)/%: o/$(MODE)/%.o o/$(MODE)/gold.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

o/$(MODE)/%.com: o/$(MODE)/%.o o/$(MODE)/gold.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

o/$(MODE)/%.runs: o/$(MODE)/%
	$<

o/$(MODE)/%.com.runs: o/$(MODE)/%.com
	$<
