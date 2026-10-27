#!/bin/sh
# tests/gb10/run.sh: build and run tests/gb10/batched.c (strided, pointer-array and single f32
# products, the epilogue and non-f32 calls that must stay on the host) against one library build.
#   OMB_BUILD=<dir>        library directory (default build-gb10; build for the host-only library)
#   OMEGABLAS_DEVICE=gb10  run on the device path (unset: host); the route predictions follow it
#   GB10_HOOKS=<CUDApp>    link tests/gb10/hooks.c (OMB_TEST_BACKEND=model, OMB_TEST_MUTANT=<n>)
# Exit status is the driver's: 0 only if every case is inside its derived limit, leaves every
# element outside the products alone, and moves the route counters as predicted.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CC=${CC:-cc}
OMB_BUILD=$(cd "$ROOT" && cd "${OMB_BUILD:-build-gb10}" && pwd)
W="$OMB_BUILD/gb10test"
rm -rf "$W"; mkdir -p "$W"
HOOKS=
if [ -n "${GB10_HOOKS:-}" ]; then
  $CC -O1 -I"$GB10_HOOKS/src" -I"$GB10_HOOKS/tests" -c -o "$W/hooks.o" "$ROOT/tests/gb10/hooks.c"
  HOOKS="$W/hooks.o"
fi
$CC -O1 -Wall -I"$ROOT/include" -o "$W/batched" "$ROOT/tests/gb10/batched.c" $HOOKS \
  -L"$OMB_BUILD" -lomegablas -Wl,-rpath,"$OMB_BUILD" -lm
echo "gb10 batched: library $OMB_BUILD/libomegablas.so, device ${OMEGABLAS_DEVICE:-host}"
"$W/batched"
