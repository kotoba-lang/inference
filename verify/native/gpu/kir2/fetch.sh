#!/bin/sh
# verify/native/gpu/kir2/fetch.sh AMU-CLONE WORK STAGE
#
# Fetches Amu's KIR v2 GPU modules at artifacts.edn's :amu/commit, checks
# them with inference's native gate (kir2_artifacts.kotoba), and stages them
# in STAGE under the names the decode guests load. JUN-375; amu ADR 0374
# option D. Nothing here runs a GPU.
#
# AMU-CLONE  a clone of kotoba-lang/amu that has :amu/commit (read with git archive)
# WORK       a new directory for Amu's tree, its build and the gate's runs
# STAGE      an empty or new directory; it receives the staged modules
#
# 1. refs (this host): Amu's oracle input, glslang's modules of the same
#    kernels. Amu's build compares against them. It needs glslangValidator.
# 2. build (arm64 Linux container): Amu selfhost writes the KIR v2 modules,
#    using the pinned root seed/bootstrap/aarch64-linux-static-root.elf.
# 3. oracle (this host): Amu's spirv-val and spirv-dis over every module.
# 4. gate (the same container): the same root compiles kir2_artifacts.kotoba
#    to a static image. The image checks and stages every artifact. Three
#    negative controls must fail by name:
#    * one byte of a module changed;
#    * one hex digit of a module's pin changed;
#    * one hex digit of the receipt's pin changed.
#
# glslang and spirv-val run only inside Amu's phases, as Amu's oracles. The
# staged modules are Amu's KIR v2 modules. CONTAINER (default podman) and
# IMAGE select the container runtime and its image.
set -eu

ROOT=/amu/seed/bootstrap/aarch64-linux-static-root.elf
TABLE=/inference/verify/native/gpu/kir2/artifacts.edn

# ---- step 4, inside the container ----
if [ "${1:-}" = "--gate" ]; then
  gate() {
    env -i PATH=/usr/bin:/bin KEXE_ALLOW=3,35,37,38,39 KEXE_CAP_RESOURCES_35=/inference:/amu:/out:/gate:/stage \
      KEXE_FUEL=4000000000 KEXE_STRUCTURED_REPORT=1 /gate/kir2_artifacts.elf "$@"
  }
  # one FAIL line and result 1, the line starting with $2
  red_as() {
    [ "$(grep -c '^FAIL ' "$1")" = 1 ] && grep -q "^FAIL $2" "$1" && grep -q '^{:status :ok :result 1 ' "$1"
  }
  env -i PATH=/usr/bin:/bin KEXE_CAP_RESOURCES_35=/gate "$ROOT" compile /gate/kir2_artifacts.kotoba \
    --target aarch64-linux --output /gate/kir2_artifacts.elf --format static
  chmod 700 /gate/kir2_artifacts.elf

  gate "$TABLE" /amu /out/spv /stage > /gate/run.out 2>&1 || true
  cat /gate/run.out
  if grep -q '^FAIL ' /gate/run.out || ! grep -q '^FAILED 0$' /gate/run.out || ! grep -q '^{:status :ok :result 0 ' /gate/run.out; then
    echo "RED: the gate failed"; exit 1
  fi

  spv=$(sed -n 's/^.*{:spv "\([^"]*\)" :amu "\([^"]*\)".*$/\1/p' "$TABLE" | head -n 1)
  amu=$(sed -n 's/^.*{:spv "\([^"]*\)" :amu "\([^"]*\)".*$/\2/p' "$TABLE" | head -n 1)

  # control 1: byte 100 of the first artifact's module, plus one
  mkdir /gate/ctl-spv
  cp /out/spv/kir2-*.spv /gate/ctl-spv/
  b=$(dd if="/gate/ctl-spv/$amu" bs=1 skip=100 count=1 2>/dev/null | od -An -tu1 | tr -d ' ')
  printf "\\$(printf '%03o' $(( (b + 1) % 256 )))" | dd of="/gate/ctl-spv/$amu" bs=1 seek=100 conv=notrunc 2>/dev/null
  gate "$TABLE" /amu /gate/ctl-spv - > /gate/ctl-module.out 2>&1 || true
  red_as /gate/ctl-module.out "sha256 $spv:" || { cat /gate/ctl-module.out; echo "RED: control 1 (a changed module) did not fail by name"; exit 1; }

  # the table with the first digit after the first occurrence of $1 changed
  flip() {
    awk -v k="$1" 'd == 0 && index($0, k) { i = index($0, k) + length(k); c = substr($0, i, 1); $0 = substr($0, 1, i - 1) (c == "0" ? "1" : "0") substr($0, i + 1); d = 1 } { print }' "$TABLE"
  }

  # control 2: the first hex digit of the first artifact's pin, changed
  flip ':sha256 "' > /gate/ctl-artifacts.edn
  gate /gate/ctl-artifacts.edn /amu /out/spv - > /gate/ctl-pin.out 2>&1 || true
  red_as /gate/ctl-pin.out "receipt $spv:" || { cat /gate/ctl-pin.out; echo "RED: control 2 (a changed pin) did not fail by name"; exit 1; }

  # control 3: the first hex digit of the receipt's pin, changed
  flip ':amu/receipt-sha256 "' > /gate/ctl-receipt.edn
  gate /gate/ctl-receipt.edn /amu /out/spv - > /gate/ctl-receipt.out 2>&1 || true
  red_as /gate/ctl-receipt.out "receipt sha256 " || { cat /gate/ctl-receipt.out; echo "RED: control 3 (a changed receipt pin) did not fail by name"; exit 1; }

  echo "{:fetch :kotoba-inference.kir2-fetch/v1 :root {:sha256 \"$(sha256sum "$ROOT" | cut -d' ' -f1)\"}" \
    ":gate {:source {:sha256 \"$(sha256sum /gate/kir2_artifacts.kotoba | cut -d' ' -f1)\"} :image {:sha256 \"$(sha256sum /gate/kir2_artifacts.elf | cut -d' ' -f1)\"}}" \
    ":run :green :controls {:module-byte :red-by-name :pin-digit :red-by-name :receipt-digit :red-by-name}}"
  exit 0
fi

# ---- steps 1-4, on the developer host ----
[ $# -eq 3 ] || { echo "usage: $0 AMU-CLONE WORK STAGE" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
INF=$(cd "$HERE/../../../.." && pwd)
CONTAINER=${CONTAINER:-podman}
IMAGE=${IMAGE:-docker.io/library/python:3.12-alpine}
COMMIT=$(sed -n 's/^{:amu\/commit "\([0-9a-f]\{40\}\)"$/\1/p' "$HERE/artifacts.edn")
[ -n "$COMMIT" ] || { echo "no :amu/commit in artifacts.edn" >&2; exit 1; }
mkdir "$2"
mkdir -p "$3"
[ -z "$(ls -A "$3")" ] || { echo "STAGE $3 is not empty" >&2; exit 1; }
WORK=$(cd "$2" && pwd)
STAGE=$(cd "$3" && pwd)
mkdir "$WORK/amu" "$WORK/out" "$WORK/gate"
git -C "$1" archive "$COMMIT" | tar -x -C "$WORK/amu"

python3 -I "$WORK/amu/scripts/seed/check-gpu-kernel-spirv.py" refs --suite kir2 --output "$WORK/out"
"$CONTAINER" run --rm --platform linux/arm64 -v "$WORK/amu:/amu:ro" -v "$WORK/out:/out" "$IMAGE" \
  python3 -I /amu/scripts/seed/check-gpu-kernel-spirv.py build --suite kir2 --compiler "$ROOT" --output /out
python3 -I "$WORK/amu/scripts/seed/check-gpu-kernel-spirv.py" oracle --suite kir2 --output "$WORK/out"

cp "$HERE/kir2_artifacts.kotoba" "$WORK/gate/"
"$CONTAINER" run --rm --platform linux/arm64 -v "$WORK/amu:/amu:ro" -v "$INF:/inference:ro" -v "$WORK/out:/out:ro" \
  -v "$WORK/gate:/gate" -v "$STAGE:/stage" "$IMAGE" sh /inference/verify/native/gpu/kir2/fetch.sh --gate \
  > "$WORK/gate/fetch.out" 2>&1 && status=0 || status=$?
cat "$WORK/gate/fetch.out"
exit "$status"
