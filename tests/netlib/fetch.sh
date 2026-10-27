#!/bin/sh
# tests/netlib/fetch.sh <dir>: download and verify netlib LAPACK v3.12.1 (only its BLAS/CBLAS test
# drivers are used, by tests/netlib/run.sh), unpack it under <dir>, print the LAPACK= path.
set -eu
D=${1:?usage: fetch.sh <dir>}
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz
SHA=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
mkdir -p "$D"
T=$D/lapack-3.12.1.tar.gz
[ -f "$T" ] || curl -fsSL -o "$T" "$URL"
echo "$SHA  $T" | sha256sum -c - >&2
[ -d "$D/lapack-3.12.1" ] || tar -xzf "$T" -C "$D"
echo "LAPACK=$(cd "$D/lapack-3.12.1" && pwd)"
