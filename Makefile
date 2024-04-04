#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set et ft=make ts=8 sw=8 fenc=utf-8 :vi ──────────────────────┘

CC = cosmocc
CXX = cosmoc++
COPTS = -g -O3 -Wall -mcosmo -pthread -fopenmp #-fsanitize=address -fsanitize=undefined
TARGET_ARCH = -Xaarch64-march=armv8.6-a
LDFLAGS = -pthread -fopenmp
LDLIBS = -lm

.PHONY: o/$(MODE)/
.PRECIOUS: o/$(MODE)/%
o/$(MODE)/: o/$(MODE)/mope.runs

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
