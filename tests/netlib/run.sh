#!/bin/sh
# tests/netlib/run.sh: the netlib reference BLAS test drivers (sblat2, sblat3) and CBLAS test
# drivers (c_sblat2, c_sblat3) from LAPACK 3.12.1, linked unmodified against libomegablas.so.
# Only the implemented routines are enabled in the input decks (SGEMV, SGEMM; error exits on);
# every other routine the drivers reference is an abort() stub, so a stray call fails loudly.
# Usage: LAPACK=<lapack-3.12.1 tree> FC=<gfortran> sh tests/netlib/run.sh
#   OMB_BUILD=<dir>        library directory (default build; build-gb10 for make GB10=1)
#   OMEGABLAS_DEVICE=gb10  run on the GB10 device path (README.md); each driver must then report
#                          host 0 and gb10 > 0 calls (OMEGABLAS_ROUTE_REPORT line in its log)
#   GB10_HOOKS=<CUDApp>    link tests/gb10/hooks.c into the drivers (OMB_TEST_BACKEND=model runs the
#                          device glue on the chip-exact HMMA model; OMB_TEST_MUTANT=1 the negative control)
#   THRESH=<x>             replace the decks' 16.0: a measurement run that prints every routine's
#                          maximum test ratio (SUSPECT lines are then expected); fails only on FATAL,
#                          FAILED or a missing END OF TESTS. Not a gate.
#   DRIVERS="<names>"      run only these of sblat2 sblat3 c_sblat2 c_sblat3 (to fit a chip hold);
#                          the others print NOT_RUN and the exit status is 1
# Exit 0 only if all four drivers report every enabled routine PASSED and none FAILED.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAPACK=${LAPACK:?set LAPACK to an unpacked lapack-3.12.1 tree (tests/netlib/fetch.sh)}
FC=${FC:-gfortran}
CC=${CC:-cc}
OMB_BUILD=$(cd "$ROOT" && cd "${OMB_BUILD:-build}" && pwd)
DEV=${OMEGABLAS_DEVICE:-host}
THRESH=${THRESH:-}
W=$OMB_BUILD/netlib
rm -rf "$W"; mkdir -p "$W"
LIB="-L$OMB_BUILD -lomegablas -Wl,-rpath,$OMB_BUILD"
[ "$DEV" = gb10 ] && export OMEGABLAS_ROUTE_REPORT=1
echo "netlib: library $OMB_BUILD/libomegablas.so, device $DEV${THRESH:+, THRESH $THRESH (measurement)}"

stubs() {
  for r in "$@"; do printf 'void %s(void) { fputs("omega-blas test stub: %s called\\n", stderr); abort(); }\n' "$r" "$r"; done
}
{
  echo '#include <stdio.h>'; echo '#include <stdlib.h>'
  stubs sgbmv_ sger_ ssbmv_ sspmv_ sspr_ sspr2_ ssymv_ ssyr_ ssyr2_ stbmv_ stbsv_ stpmv_ stpsv_ \
        strmv_ strsv_ sgemmtr_ ssymm_ ssyr2k_ ssyrk_ strmm_ strsm_
  stubs cblas_sgbmv cblas_sger cblas_ssbmv cblas_sspmv cblas_sspr cblas_sspr2 cblas_ssymv \
        cblas_ssyr cblas_ssyr2 cblas_stbmv cblas_stbsv cblas_stpmv cblas_stpsv cblas_strmv \
        cblas_strsv cblas_sgemmtr cblas_ssymm cblas_ssyr2k cblas_ssyrk cblas_strmm cblas_strsm
} > "$W/stubs.c"
$CC -O1 -c -o "$W/stubs.o" "$W/stubs.c"
HOOKS=
if [ -n "${GB10_HOOKS:-}" ]; then
  $CC -O1 -I"$GB10_HOOKS/src" -I"$GB10_HOOKS/tests" -c -o "$W/hooks.o" "$ROOT/tests/gb10/hooks.c"
  HOOKS="$W/hooks.o"
fi

# input decks: keep the SGEMV / SGEMM lines enabled, disable every other routine
# (fixed-column reads: flip the T in place, keep the spacing)
deck() {
  sed -E "/^(cblas_)?[sS][A-Za-z0-9]+ +[TF] /{ /^$2 /!s/^([A-Za-z0-9_]+ +)T/\\1F/; }" "$1" |
    if [ -n "$THRESH" ]; then sed -E "s/^16\\.0( +THRESHOLD VALUE)/$THRESH\\1/"; else cat; fi
}
deck "$LAPACK/BLAS/TESTING/sblat2.in" SGEMV > "$W/sblat2.in"
deck "$LAPACK/BLAS/TESTING/sblat3.in" SGEMM > "$W/sblat3.in"
deck "$LAPACK/CBLAS/testing/sin2" cblas_sgemv > "$W/sin2"
deck "$LAPACK/CBLAS/testing/sin3" cblas_sgemm > "$W/sin3"
grep -q "^${THRESH:-16.0} *THRESHOLD VALUE" "$W/sblat2.in" "$W/sin3" || { echo "deck threshold not set"; exit 1; }

FFLAGS="-O0 -fno-optimize-sibling-calls"
$FC $FFLAGS -o "$W/sblat2" "$LAPACK/BLAS/TESTING/sblat2.f" "$W/stubs.o" $HOOKS $LIB
$FC $FFLAGS -o "$W/sblat3" "$LAPACK/BLAS/TESTING/sblat3.f" "$W/stubs.o" $HOOKS $LIB
printf '#ifndef F77_GLOBAL\n#define F77_GLOBAL(lcname,UCNAME) lcname##_\n#endif\n' > "$W/cblas_mangling.h"
CI="-I$W -I$LAPACK/CBLAS/include -DADD_"
T=$LAPACK/CBLAS/testing
for f in c_sblas2 c_s2chke c_sblas3 c_s3chke auxiliary c_xerbla; do $CC -O1 $CI -c -o "$W/$f.o" "$T/$f.c"; done
$FC $FFLAGS -o "$W/c_sblat2" "$T/c_sblat2.f" "$W/c_sblas2.o" "$W/c_s2chke.o" "$W/auxiliary.o" "$W/c_xerbla.o" "$W/stubs.o" $HOOKS $LIB
$FC $FFLAGS -o "$W/c_sblat3" "$T/c_sblat3.f" "$W/c_sblas3.o" "$W/c_s3chke.o" "$W/auxiliary.o" "$W/c_xerbla.o" "$W/stubs.o" $HOOKS $LIB

rc=0
check() { # name log nblocks routine runlog: require END OF TESTS, no failure/suspect lines, the
  #        error-exit PASSED line and nblocks computational PASSED lines (2 for CBLAS: column- and
  #        row-major); on the GB10, the route line of runlog with host 0 and gb10 > 0
  n=$1; log=$2; want=$3; r=$4; run=$5
  if [ "$DEV" = gb10 ]; then
    rl=$(grep "^omega-blas routes:" "$run" || true)
    echo "$n: $rl"
    case "$rl" in
      "omega-blas routes: device gb10 host 0 gb10 0 "*|"") echo "$n: FAIL (no calls reached the GB10)"; rc=1; return ;;
      "omega-blas routes: device gb10 host 0 gb10 "*) ;;
      *) echo "$n: FAIL (calls ran on the host)"; rc=1; return ;;
    esac
  fi
  if [ -n "$THRESH" ]; then
    grep "MAXIMUM TEST RATIO\|PASSED" "$log" | sed "s/^/$n: /"
    if grep -q "FATAL\|FAILED" "$log" || [ "$(grep -c "END OF TESTS" "$log" || true)" != 1 ]; then echo "$n: FAIL"; rc=1; return; fi
    echo "$n: measured (THRESH $THRESH)"; return
  fi
  if grep -q "FAIL\|SUSPECT\|\*\*\*\*\*\*" "$log"; then echo "$n: FAIL"; grep -n "FAIL\|SUSPECT\|\*\*\*\*\*\*" "$log" | head -20; rc=1; return; fi
  c=$(grep -c "^ *$r  *PASSED THE .*COMPUTATIONAL TESTS" "$log" || true)
  e=$(grep -c "^ *$r  *PASSED THE TESTS OF ERROR-EXITS" "$log" || true)
  t=$(grep -c "END OF TESTS" "$log" || true)
  if [ "$c" != "$want" ] || [ "$e" != 1 ] || [ "$t" != 1 ]; then
    echo "$n: FAIL (computational PASSED $c/$want, error-exits PASSED $e/1, END OF TESTS $t/1)"; rc=1; return
  fi
  echo "$n: PASS"; grep "^ *$r  *PASSED" "$log"
}
cd "$W"
# the F77 drivers write their summary to sblat?.out (named in the deck); the CBLAS ones to stdout
run() { # driver deck summary nblocks routine
  case " ${DRIVERS:-sblat2 sblat3 c_sblat2 c_sblat3} " in *" $1 "*) ;; *) echo "$1: NOT_RUN (DRIVERS)"; rc=1; return ;; esac
  t0=$(date +%s.%N)
  ./$1 < $2 > $1.log 2>&1 || { echo "$1 exit $?"; rc=1; }
  echo "$1: wall $(echo "$(date +%s.%N) - $t0" | bc) s"
  check $1 $3 $4 $5 $1.log
}
run sblat2 sblat2.in sblat2.out 1 SGEMV
run sblat3 sblat3.in sblat3.out 1 SGEMM
run c_sblat2 sin2 c_sblat2.log 2 cblas_sgemv
run c_sblat3 sin3 c_sblat3.log 2 cblas_sgemm
exit $rc
