# drive a resident guest over its stdin/stdout pipe: prompt ids, then feed each argmax back (TEST HARNESS beside the
# oracle, not tooling; the serving shell is serve_http.cljk). python3 resident_drive.py <loader> <resident.bin> <offset> <ids> <ntok> [requests] [isa]
import subprocess, sys, os, time, struct
loader, binf, off, plen_ids, ntok, reqs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1
isa = sys.argv[7] if len(sys.argv) > 7 else "x86_64"
env = dict(os.environ, KEXE_RESULT_TYPE="string", KEXE_STRUCTURED_REPORT="1", KEXE_FUEL="1000000000", KEXE_WALL_SECONDS="86400", KEXE_STRING_POOL="16777216", KEXE_PAIRS="4194304")
t0 = time.time()
p = subprocess.Popen([loader, binf, off, "0", isa, "42,33,41,37,39"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)
def step(tok, reset):
    p.stdin.write(f"{tok},{1 if reset else 0}".encode()); p.stdin.flush()
    line = p.stdout.readline().decode().strip()
    ns, idhex = line.split("|"); return int(ns), struct.unpack("<I", bytes.fromhex(idhex))[0]
first = None
for r in range(reqs):
    ids = [int(t) for t in plen_ids.split(",")]; out = []; nss = []; tr = time.time()
    for i, t in enumerate(ids):
        ns, am = step(t, i == 0); nss.append(ns)
    out.append(am)
    for _ in range(ntok - 1):
        ns, am = step(am, False); nss.append(ns); out.append(am)
    print(f"request {r}: generated {','.join(map(str, out))}  steps {len(nss)}  ms/token {sum(nss)/len(nss)/1e6:.2f}  wall {time.time()-tr:.3f} s" + ("" if first is None else ("  same" if out == first else "  DIFFERENT")))
    if first is None: first = out; print(f"first request wall since process start {time.time()-t0:.2f} s")
p.stdin.close(); rest = p.stdout.read().decode(); p.wait(); print(rest.strip()[-300:])
