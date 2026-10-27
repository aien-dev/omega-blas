#!/bin/sh
# tests/netlib/run.sh: the netlib reference BLAS test drivers (sblat2, sblat3) and CBLAS test
# drivers (c_sblat2, c_sblat3) from LAPACK 3.12.1, linked against build/libomegablas.so.
# Only the implemented routines are enabled in the input decks (SGEMV, SGEMM; error exits on);
# every other routine the drivers reference is an abort() stub, so a stray call fails loudly.
# Usage: LAPACK=<lapack-3.12.1 tree> FC=<gfortran> sh tests/netlib/run.sh
# Exit 0 only if all four drivers report every enabled routine PASSED and none FAILED.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAPACK=${LAPACK:?set LAPACK to an unpacked lapack-3.12.1 tree (https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz)}
FC=${FC:-gfortran}
CC=${CC:-cc}
W=$ROOT/build/netlib
rm -rf "$W"; mkdir -p "$W"
LIB="-L$ROOT/build -lomegablas -Wl,-rpath,$ROOT/build"

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

# input decks: keep the SGEMV / SGEMM lines enabled, disable every other routine
# (fixed-column reads: flip the T in place, keep the spacing)
deck() { sed -E "/^(cblas_)?[sS][A-Za-z0-9]+ +[TF] /{ /^$2 /!s/^([A-Za-z0-9_]+ +)T/\\1F/; }" "$1"; }
deck "$LAPACK/BLAS/TESTING/sblat2.in" SGEMV > "$W/sblat2.in"
deck "$LAPACK/BLAS/TESTING/sblat3.in" SGEMM > "$W/sblat3.in"
deck "$LAPACK/CBLAS/testing/sin2" cblas_sgemv > "$W/sin2"
deck "$LAPACK/CBLAS/testing/sin3" cblas_sgemm > "$W/sin3"

FFLAGS="-O0 -fno-optimize-sibling-calls"
$FC $FFLAGS -o "$W/sblat2" "$LAPACK/BLAS/TESTING/sblat2.f" "$W/stubs.o" $LIB
$FC $FFLAGS -o "$W/sblat3" "$LAPACK/BLAS/TESTING/sblat3.f" "$W/stubs.o" $LIB
printf '#ifndef F77_GLOBAL\n#define F77_GLOBAL(lcname,UCNAME) lcname##_\n#endif\n' > "$W/cblas_mangling.h"
CI="-I$W -I$LAPACK/CBLAS/include -DADD_"
T=$LAPACK/CBLAS/testing
for f in c_sblas2 c_s2chke c_sblas3 c_s3chke auxiliary c_xerbla; do $CC -O1 $CI -c -o "$W/$f.o" "$T/$f.c"; done
$FC $FFLAGS -o "$W/c_sblat2" "$T/c_sblat2.f" "$W/c_sblas2.o" "$W/c_s2chke.o" "$W/auxiliary.o" "$W/c_xerbla.o" "$W/stubs.o" $LIB
$FC $FFLAGS -o "$W/c_sblat3" "$T/c_sblat3.f" "$W/c_sblas3.o" "$W/c_s3chke.o" "$W/auxiliary.o" "$W/c_xerbla.o" "$W/stubs.o" $LIB

rc=0
check() { # name log nblocks routine: require END OF TESTS, no failure/suspect lines, the error-exit
  #        PASSED line and nblocks computational PASSED lines (2 for CBLAS: column- and row-major)
  n=$1; log=$2; want=$3; r=$4
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
./sblat2 < sblat2.in > sblat2.log 2>&1 || { echo "sblat2 exit $?"; rc=1; }
check sblat2 sblat2.out 1 SGEMV
./sblat3 < sblat3.in > sblat3.log 2>&1 || { echo "sblat3 exit $?"; rc=1; }
check sblat3 sblat3.out 1 SGEMM
./c_sblat2 < sin2 > c_sblat2.log 2>&1 || { echo "c_sblat2 exit $?"; rc=1; }
check c_sblat2 c_sblat2.log 2 cblas_sgemv
./c_sblat3 < sin3 > c_sblat3.log 2>&1 || { echo "c_sblat3 exit $?"; rc=1; }
check c_sblat3 c_sblat3.log 2 cblas_sgemm
exit $rc
