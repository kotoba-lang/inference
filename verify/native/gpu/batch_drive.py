# drive a `resident-replay … batch:<B>` guest: B sequences advance together, one message "t0,…,tB-1,f0,…,fB-1" per step
# (TEST HARNESS beside the oracle). Each row has its own prompt and token budget; a finished or empty row is fed
# token 0 with flag 1 (a rewind that costs nothing). Rows are compared with the single-sequence oracles.
# python3 batch_drive.py <loader> <batch.bin> <offset> <isa> <B> <prompt-ids>:<ntok>[:<expected-ids>] ...
import subprocess, sys, os, struct, time
loader, binf, off, isa, B = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = []
for spec in sys.argv[6:]:
    parts = spec.split(":"); ids = [int(t) for t in parts[0].split(",")]; ntok = int(parts[1])
    exp = [int(t) for t in parts[2].split(",")] if len(parts) > 2 and parts[2] else None
    rows.append({"ids": ids, "ntok": ntok, "exp": exp, "i": 0, "out": [], "next": ids[0], "reset": True})
assert len(rows) <= B
env = dict(os.environ, KEXE_RESULT_TYPE="string", KEXE_STRUCTURED_REPORT="1", KEXE_FUEL="1000000000", KEXE_WALL_SECONDS="86400", KEXE_STRING_POOL="1073741824", KEXE_PAIRS="67108864")
p = subprocess.Popen([loader, binf, off, "0", isa, "42,33,41,37,39"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0)
def active(r): return r["i"] < len(r["ids"]) or len(r["out"]) < r["ntok"]
nss = []; t0 = None; steps = 0
while any(active(r) for r in rows):
    toks = []; flags = []
    for r in rows:
        if active(r): toks.append(r["next"]); flags.append(1 if r["reset"] else 0)
        else: toks.append(0); flags.append(1)
    while len(toks) < B: toks.append(0); flags.append(1)
    p.stdin.write((",".join(map(str, toks + flags))).encode()); p.stdin.flush()
    if t0 is None: t0 = time.time()
    line = p.stdout.readline().decode().strip(); ns, idhex = line.split("|"); nss.append(int(ns)); steps += 1
    words = [struct.unpack("<I", bytes.fromhex(idhex[k:k + 8]))[0] for k in range(0, len(idhex), 8)]
    for ri, r in enumerate(rows):
        if not active(r): continue
        pick = words[ri * 64 + 61]
        r["reset"] = False; r["i"] += 1
        if r["i"] < len(r["ids"]): r["next"] = r["ids"][r["i"]]          # still the prompt
        else: r["out"].append(pick); r["next"] = pick
wall = time.time() - t0
p.stdin.close(); p.wait()
ok = 0
for ri, r in enumerate(rows):
    good = (r["exp"] is None) or (r["out"] == r["exp"]); ok += good
    print(f"row {ri}: generated {','.join(map(str, r['out']))}" + ("" if r["exp"] is None else ("  OK" if good else f"  BAD expected {','.join(map(str, r['exp']))}")))
print(f"BATCH\t{ok}/{len(rows)} rows equal to their oracle  steps {steps}  gpu ms total {sum(nss)/1e6:.1f}  ms/step {sum(nss)/1e6/steps:.2f}  wall {wall:.3f} s")
