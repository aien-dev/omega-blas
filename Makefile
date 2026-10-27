# omega-blas build. The compute unit om/blas.om is compiled by the pinned OM-2 omc and bound by
# omega-bridge tools/native (ABI v2); libomegablas.so links that binding with the interim C layer.
#   make BRIDGE=<omega-bridge checkout> OMC=<seeded omc> OMC2=<OM-2 omc>
# Pins: see PINS (parent omega-mv sha, omc sha256s, omega-bridge sha).
BRIDGE ?= ../omega-bridge
OMC ?= $(error set OMC to the seeded omc (sha256 in PINS))
OMC2 ?= $(error set OMC2 to the OM-2 omc (sha256 in PINS))
CC ?= cc
CFLAGS ?= -O2
B := build
N := $(B)/native

all: $(B)/libomegablas.so

$(N)/omega_native.c: om/blas.om
	OMEGA_NATIVE_OMC=$(OMC) OMEGA_NATIVE_OMC2=$(OMC2) $(BRIDGE)/tools/native $(N) om/blas.om

$(B)/libomegablas.so: $(N)/omega_native.c src/omegablas.c include/omegablas.h
	$(CC) -std=gnu11 $(CFLAGS) -Wall -Wextra -Werror -fPIC -shared -Wl,-soname,libomegablas.so \
	  -Iinclude -I$(N) -include $(N)/omega_native.h -o $@ src/omegablas.c $(N)/omega_native.c -lpthread

clean:
	rm -rf $(B)

.PHONY: all clean
