# omega-blas build. The compute unit om/blas.om is compiled by the pinned OM-2 omc and bound by
# omega-bridge tools/native (ABI v2); libomegablas.so links that binding with the interim C layer.
#   make BRIDGE=<omega-bridge checkout> OMC=<seeded omc> OMC2=<OM-2 omc>
#   make ... GB10=1 CUDAPP=<CUDApp checkout>   the GB10 device path (README.md), into build-gb10/:
#        links CUDApp's libomega_gpu.a, built position-independent into $(CUDAPP)/build/omegablas-pic
# Pins: see PINS (parent omega-mv sha, omc sha256s, omega-bridge sha, CUDApp sha).
BRIDGE ?= ../omega-bridge
OMC ?= $(error set OMC to the seeded omc (sha256 in PINS))
OMC2 ?= $(error set OMC2 to the OM-2 omc (sha256 in PINS))
CC ?= cc
CFLAGS ?= -O2
GB10 ?= 0
N := build/native
ifeq ($(GB10),1)
CUDAPP ?= $(error set CUDAPP to a CUDApp checkout (sha in PINS))
B := build-gb10
GPU_LIB := $(CUDAPP)/build/omegablas-pic/libomega_gpu.a
GPU_CFLAGS := -DOMB_GB10 -I$(CUDAPP)/src \
  -DOMB_CUDAPP_REV='"$(shell git -C $(CUDAPP) rev-parse HEAD)$(shell git -C $(CUDAPP) diff --quiet HEAD || echo -dirty)"'
GPU_LIBS := $(GPU_LIB) -lm
else
B := build
endif

all: $(B)/libomegablas.so

$(N)/omega_native.c: om/blas.om
	OMEGA_NATIVE_OMC=$(OMC) OMEGA_NATIVE_OMC2=$(OMC2) $(BRIDGE)/tools/native $(N) om/blas.om

$(B)/libomegablas.so: $(N)/omega_native.c src/omegablas.c include/omegablas.h $(GPU_LIB)
	mkdir -p $(B)
	$(CC) -std=gnu11 $(CFLAGS) -Wall -Wextra -Werror -fPIC -shared -Wl,-soname,libomegablas.so \
	  -Iinclude -I$(N) -include $(N)/omega_native.h $(GPU_CFLAGS) -o $@ src/omegablas.c \
	  $(N)/omega_native.c $(GPU_LIBS) -lpthread

ifeq ($(GB10),1)
# CUDApp's own build with -fPIC added to the compiler; its CFLAGS, including the gemm_ex
# migration-pending switch of mk/gemm-ex.mk, stay as CUDApp sets them.
$(GPU_LIB): FORCE
	$(MAKE) -C $(CUDAPP) OUT_DIR=build/omegablas-pic CC="$(CC) -fPIC" build/omegablas-pic/libomega_gpu.a
endif

clean:
	rm -rf build build-gb10

.PHONY: all clean FORCE
