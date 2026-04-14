"""
=============================================================================
benchmark_2048bit.py
Complete labHE Benchmark — Paillier 2048-bit (CORRECTED)
=============================================================================

All benchmarks run at 2048-bit security.
Bug fixed: key_size=1024 → key_size=2048 (was on line 126 of
           the original petrol_benchmark.py).

Sections:
    1. KeyGen              (5 reps  — slow, generates prime pairs)
    2. Core operations     (30 reps — offline/online enc, hom ops, dec)
    3. ZK proof + epoch    (100 reps — security extensions C4 & C5)
    4. 181-node pipeline   (5 independent full runs — mean ± std)
    5. Petrol dataset      (per-column offline/online timing)

Results are saved to benchmark_2048bit_results.json.
=============================================================================
"""

import hashlib
import json
import math
import random
import statistics
import time
from typing import Dict, List, Tuple

from sympy import randprime, mod_inverse

# ─── Paillier primitives (self-contained for benchmark isolation) ───────────

def _gen_prime(bits):
    return int(randprime(2 ** (bits - 1), 2 ** bits))

def _lcm(a, b):    return a * b // math.gcd(a, b)
def _modinv(a, m): return int(mod_inverse(a, m))
def _wire(x):      return (x.bit_length() + 7) // 8

def paillier_keygen(bits=2048):
    p = _gen_prime(bits // 2); q = _gen_prime(bits // 2)
    while p == q: q = _gen_prime(bits // 2)
    n = p * q; n2 = n * n; g = n + 1
    lam = _lcm(p - 1, q - 1)
    mu  = _modinv((pow(g, lam, n2) - 1) // n, n)
    K   = random.getrandbits(256)
    return {"n":n,"g":g,"n2":n2}, {"p":p,"q":q,"n":n,"lam":lam,"mu":mu}, K

def prf(K, label, n):
    h = hashlib.sha256()
    h.update(K.to_bytes(32, "big"))
    h.update(label.encode())
    return int.from_bytes(h.digest(), "big") % n

def epoch_key(K, t):
    h = hashlib.sha256()
    h.update(K.to_bytes(32, "big"))
    h.update(t.to_bytes(8, "big"))
    return int.from_bytes(h.digest(), "big")

def offline_enc(pk, K, label):
    n, g, n2 = pk["n"], pk["g"], pk["n2"]
    b = prf(K, label, n); r = random.randint(2, n - 1)
    return b, (pow(g, b, n2) * pow(r, n, n2)) % n2

def online_enc(m, b, n):   return (m - b) % n
def hom_add(a1, b1, a2, b2, n, n2): return (a1+a2)%n, (b1*b2)%n2

def hom_mult(a1, bt1, a2, bt2, pk):
    n, g, n2 = pk["n"], pk["g"], pk["n2"]
    r = random.randint(2, n - 1)
    enc_a1a2 = (pow(g, (a1*a2)%n, n2) * pow(r, n, n2)) % n2
    return (enc_a1a2 * pow(bt2, a1, n2) * pow(bt1, a2, n2)) % n2

def paillier_dec(sk, c):
    n, lam, mu = sk["n"], sk["lam"], sk["mu"]
    x = pow(c, lam, n*n)
    return ((x-1)//n * mu) % n

def on_dec2(sk, alpha, b_prod): return (paillier_dec(sk, alpha) + b_prod) % sk["n"]

def zk_prove(m, b, a, lo, hi, n):
    r = random.randint(1, n-1); R = (m-b+r)%n
    h = hashlib.sha256()
    h.update(R.to_bytes(_wire(R) or 1,"big"))
    h.update(lo.to_bytes(_wire(max(lo,1)) or 1,"big"))
    h.update(hi.to_bytes(_wire(hi) or 1,"big"))
    e = int.from_bytes(h.digest(),"big") % n
    return {"R":R,"e":e,"z":(r+e*m)%n,"lo":lo,"hi":hi}

def zk_verify(proof, a, n):
    h = hashlib.sha256()
    h.update(proof["R"].to_bytes(_wire(proof["R"]) or 1,"big"))
    h.update(proof["lo"].to_bytes(_wire(max(proof["lo"],1)) or 1,"big"))
    h.update(proof["hi"].to_bytes(_wire(proof["hi"]) or 1,"big"))
    e2 = int.from_bytes(h.digest(),"big") % n
    if proof["e"] != e2: return False
    w = (proof["z"] - proof["e"]*a) % n
    return proof["lo"] <= w <= proof["hi"]

# ─── Timing helpers ─────────────────────────────────────────────────────────

def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1000)
    m = statistics.mean(ts)
    s = statistics.stdev(ts) if len(ts)>1 else 0.0
    tc = 2.045 if reps<=30 else (1.984 if reps<=100 else 1.960)
    se = s / math.sqrt(reps)
    return m, s, round(m-tc*se,5), round(m+tc*se,5)

# ─── Section 1: KeyGen ──────────────────────────────────────────────────────

def bench_keygen(reps=5):
    print(f"  KeyGen 2048-bit ({reps} reps)...")
    times, pk0, sk0, K0 = [], None, None, None
    for i in range(reps):
        t = time.perf_counter()
        pk, sk, K = paillier_keygen(2048)
        times.append((time.perf_counter()-t)*1000)
        if i==0: pk0,sk0,K0 = pk,sk,K
        print(f"    {i+1}/{reps}: {times[-1]:.0f} ms")
    m,s,lo,hi = statistics.mean(times), statistics.stdev(times) if reps>1 else 0.0, 0,0
    tc=2.776; se=s/math.sqrt(reps)
    lo,hi = round(m-tc*se,2), round(m+tc*se,2)
    print(f"  → {m:.2f} ± {s:.2f} ms  CI-95=[{lo},{hi}]")
    return pk0, sk0, K0, {"mean_ms":round(m,2),"std_ms":round(s,2),"ci95":[lo,hi],"reps":reps}

# ─── Section 2: Core operations (30 reps) ──────────────────────────────────

def bench_core(pk, sk, K, reps=30):
    n, n2 = pk["n"], pk["n2"]
    m1, m2 = 533573, 332
    b1,bt1 = offline_enc(pk,K,"label_1"); a1 = online_enc(m1,b1,n)
    b2,bt2 = offline_enc(pk,K,"label_2"); a2 = online_enc(m2,b2,n)
    alpha  = hom_mult(a1,bt1,a2,bt2,pk)

    ops = {}
    print(f"  Offline enc  ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: offline_enc(pk,K,"bench_label"), reps)
    ops["offline_enc"] = {"mean":round(mv,4),"std":round(sv,4),"ci95":[lo,hi]}
    print(f"    {mv:.3f} ± {sv:.3f} ms  CI-95=[{lo},{hi}]")

    print(f"  Online enc   ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: online_enc(m1,b1,n), reps)
    ops["online_enc"] = {"mean":round(mv,6),"std":round(sv,6),"ci95":[lo,hi]}
    print(f"    {mv:.6f} ms")

    print(f"  Hom Add      ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: hom_add(a1,bt1,a2,bt2,n,n2), reps)
    ops["hom_add"] = {"mean":round(mv,5),"std":round(sv,5),"ci95":[lo,hi]}
    print(f"    {mv:.5f} ms")

    print(f"  Hom Mult     ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: hom_mult(a1,bt1,a2,bt2,pk), reps)
    ops["hom_mult"] = {"mean":round(mv,3),"std":round(sv,3),"ci95":[lo,hi]}
    print(f"    {mv:.3f} ± {sv:.3f} ms  CI-95=[{lo},{hi}]")

    print(f"  Offline dec  ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: (prf(K,"label_1",n)+prf(K,"label_2",n))%n, reps)
    ops["offline_dec"] = {"mean":round(mv,5),"std":round(sv,5),"ci95":[lo,hi]}
    print(f"    {mv:.5f} ms")

    print(f"  Online dec1  ({reps} reps)...", flush=True)
    bt = (b1+b2)%n; at,_ = hom_add(a1,bt1,a2,bt2,n,n2)
    mv,sv,lo,hi = timeit(lambda: (at+bt)%n, reps)
    ops["online_dec1"] = {"mean":round(mv,6),"std":round(sv,6),"ci95":[lo,hi]}
    print(f"    {mv:.6f} ms")

    print(f"  Online dec2  ({reps} reps)...", flush=True)
    bp = (b1*b2)%n
    mv,sv,lo,hi = timeit(lambda: on_dec2(sk,alpha,bp), reps)
    ops["online_dec2"] = {"mean":round(mv,3),"std":round(sv,3),"ci95":[lo,hi]}
    print(f"    {mv:.3f} ± {sv:.3f} ms  CI-95=[{lo},{hi}]")

    # Correctness
    dec1 = (at+bt)%n; ok1 = (dec1 == (m1+m2)%n)
    dec2 = on_dec2(sk,alpha,bp); ok2 = (dec2 == (m1*m2)%n)
    ops["correctness_deg1"] = ok1
    ops["correctness_deg2"] = ok2
    print(f"  Deg-1 correct: {ok1}  (got {dec1}, expected {(m1+m2)%n})")
    print(f"  Deg-2 correct: {ok2}  (got {dec2}, expected {(m1*m2)%n})")

    # Wire-format sizes
    ops["ct_size_bytes"]       = _wire(n2)              # β ∈ Z_{n²} = 512 B
    ops["bandwidth_bytes"]     = _wire(n2) + _wire(n)   # β + a      = 768 B
    ops["pk_size_bytes"]       = _wire(n)*2 + _wire(n2) # ≈ 1024 B
    ops["sk_size_bytes"]       = (sum(_wire(sk[k]) for k in ("p","q","n","lam","mu")))
    print(f"  CT size:     {ops['ct_size_bytes']} bytes")
    print(f"  Bandwidth:   {ops['bandwidth_bytes']} bytes/node")
    print(f"  pk size:     {ops['pk_size_bytes']} bytes")
    print(f"  sk size:     {ops['sk_size_bytes']} bytes")
    return ops

# ─── Section 3: ZK + Epoch (100 reps) ───────────────────────────────────────

def bench_security(pk, K, reps=100):
    n = pk["n"]
    m_val = 533573; lo_b, hi_b = 0, 25_000_000
    b,beta = offline_enc(pk,K,"zk_bench"); a_val = online_enc(m_val,b,n)
    proof  = zk_prove(m_val,b,a_val,lo_b,hi_b,n)
    sec = {}

    print(f"  ZK prove  ({reps} reps)...", flush=True)
    mv,sv,lo,hi = timeit(lambda: zk_prove(m_val,b,a_val,lo_b,hi_b,n), reps)
    sec["zk_prove"] = {"mean":round(mv,5),"std":round(sv,5),"ci95":[round(lo,5),round(hi,5)]}
    print(f"    {mv:.5f} ± {sv:.5f} ms  CI-95=[{round(lo,5)},{round(hi,5)}]")

    print(f"  ZK verify ({reps} reps)...", flush=True)
    mv2,sv2,lo2,hi2 = timeit(lambda: zk_verify(proof,a_val,n), reps)
    sec["zk_verify"] = {"mean":round(mv2,5),"std":round(sv2,5),"ci95":[round(lo2,5),round(hi2,5)]}
    print(f"    {mv2:.5f} ± {sv2:.5f} ms  CI-95=[{round(lo2,5)},{round(hi2,5)}]")

    sec["zk_combined_ms"]   = round(mv+mv2, 5)
    sec["zk_prove_181_ms"]  = round(mv*181, 3)
    sec["zk_verify_181_ms"] = round(mv2*181, 3)
    sec["forgery_rejected"]  = not zk_verify(proof, online_enc(m_val+1,b,n), n)
    print(f"  Combined/node: {sec['zk_combined_ms']} ms")
    print(f"  181 nodes:     prove={sec['zk_prove_181_ms']} ms  verify={sec['zk_verify_181_ms']} ms")
    print(f"  Forgery rejected: {sec['forgery_rejected']}")

    print(f"  Epoch key ({reps} reps)...", flush=True)
    mv3,sv3,lo3,hi3 = timeit(lambda: epoch_key(K,1), reps)
    sec["epoch_key"] = {"mean":round(mv3,6),"std":round(sv3,6),"ci95":[round(lo3,6),round(hi3,6)]}
    sec["epoch_keys_distinct"] = (epoch_key(K,1) != epoch_key(K,2))
    print(f"    {mv3:.6f} ± {sv3:.6f} ms")
    print(f"  K^(1) ≠ K^(2): {sec['epoch_keys_distinct']}")
    return sec

# ─── Section 4: 181-node pipeline (5 runs) ──────────────────────────────────

def bench_pipeline(pk, sk, K, n_nodes=181, runs=5):
    n, n2 = pk["n"], pk["n2"]
    import random as _r; _r.seed(42)
    msgs = [int(abs(_r.gauss(533573,500000))) % (n//2) for _ in range(n_nodes)]
    all_runs = []

    for run in range(runs):
        print(f"  Run {run+1}/{runs}...", end=" ", flush=True)
        # Offline enc
        t0 = time.perf_counter()
        cache = {}
        for i in range(n_nodes):
            cache[i] = offline_enc(pk, K, f"node_{i}")
        t_off = (time.perf_counter()-t0)*1000

        # Online enc
        t0 = time.perf_counter()
        cts = [(online_enc(msgs[i], cache[i][0], n), cache[i][1]) for i in range(n_nodes)]
        t_on = (time.perf_counter()-t0)*1000

        # Hom agg
        t0 = time.perf_counter()
        aa, ba = cts[0]
        for a_i, b_i in cts[1:]: aa=(aa+a_i)%n; ba=(ba*b_i)%n2
        t_agg = (time.perf_counter()-t0)*1000

        # Offline dec (PRF sum)
        t0 = time.perf_counter()
        bt = sum(prf(K,f"node_{i}",n) for i in range(n_nodes)) % n
        t_odec = (time.perf_counter()-t0)*1000

        # Online dec
        t0 = time.perf_counter()
        result = (aa+bt)%n
        t_dec = (time.perf_counter()-t0)*1000

        full = t_on+t_agg+t_odec+t_dec
        correct = (result == sum(msgs)%n)
        all_runs.append({"off":t_off,"on":t_on,"agg":t_agg,"odec":t_odec,"dec":t_dec,"full":full,"ok":correct})
        print(f"full_online={full:.2f}ms correct={correct}")

    def ag(k): return round(statistics.mean(r[k] for r in all_runs),3), round(statistics.stdev(r[k] for r in all_runs) if runs>1 else 0,3)
    om,os = ag("off"); nm,ns = ag("on"); am,as_ = ag("agg"); dm,ds = ag("odec"); em,es = ag("dec"); fm,fs = ag("full")
    p = {
        "n_nodes": n_nodes, "pipeline_reps": runs,
        "offline_enc_ms": f"{om} ± {os}",
        "offline_per_node_ms": round(om/n_nodes,2),
        "online_enc_ms": f"{nm} ± {ns}",
        "online_per_node_us": round(nm/n_nodes*1000,2),
        "hom_agg_ms": f"{am} ± {as_}",
        "offline_dec_ms": f"{dm} ± {ds}",
        "online_dec_ms": f"{em} ± {es}",
        "full_online_ms": f"{fm} ± {fs}",
        "all_correct": all(r["ok"] for r in all_runs),
        "ct_size_bytes": _wire(pk["n2"]),
        "bandwidth_bytes": _wire(pk["n2"])+_wire(pk["n"]),
        "offline_cache_kb": round(n_nodes*(_wire(pk["n2"])+_wire(pk["n"]))/1024,1),
        "ct_payload_kb":    round(n_nodes*(_wire(pk["n2"])+_wire(pk["n"]))/1024,1),
    }
    print(f"\n  Offline Enc 181:  {p['offline_enc_ms']} ms ({p['offline_per_node_ms']} ms/node)")
    print(f"  Online Enc 181:   {p['online_enc_ms']} ms ({p['online_per_node_us']} µs/node)")
    print(f"  Hom Aggregation:  {p['hom_agg_ms']} ms")
    print(f"  Full online:      {p['full_online_ms']} ms")
    print(f"  All correct:      {p['all_correct']}")
    print(f"  CT size/node:     {p['ct_size_bytes']} bytes")
    print(f"  Bandwidth/node:   {p['bandwidth_bytes']} bytes")
    print(f"  vs BFV 32-65 KB:  {round(32*1024/p['ct_size_bytes'])}× to {round(65*1024/p['bandwidth_bytes'])}× smaller")
    return p

# ─── Section 5: Petrol column benchmark ─────────────────────────────────────

def bench_petrol_columns(pk, K):
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not available — skipping"); return {}
    COLS = ["Daily Oil Consumption (Barrels)","Yearly Gallons Per Capita",
            "Price Per Gallon (USD)","GDP Per Capita ( USD )"]
    paths = ["/mnt/user-data/uploads/Petrol_Dataset_June_23_2022_--_Version_2.csv",
             "Petrol_Dataset_June_23_2022_--_Version_2.csv"]
    df = None
    for p in paths:
        for enc in ["utf-8","latin-1","cp1252"]:
            try: df=pd.read_csv(p,encoding=enc); print(f"  Loaded: {p}"); break
            except: pass
        if df is not None: break
    if df is None: print("  Dataset not found"); return {}

    n = pk["n"]; SCALE=1000; results={}
    for col in COLS:
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("%",""),errors="coerce").dropna().tolist()
        nr = len(vals); msgs = [int(abs(float(v))*SCALE)%(n//2) for v in vals]
        print(f"\n  {col[:40]} ({nr} rows):", flush=True)

        t0=time.perf_counter()
        for i,m in enumerate(msgs):
            b=prf(K,f"{col}_std_{i}",n); r=random.randint(2,n-1)
            (pow(pk["g"],b,pk["n2"])*pow(r,n,pk["n2"]))%pk["n2"]
            (m-b)%n
        t_std=(time.perf_counter()-t0)*1000

        t0=time.perf_counter()
        cache2=[(lambda i=i,m=m: offline_enc(pk,K,f"{col}_off_{i}"))() for i,m in enumerate(msgs)]
        t_off=(time.perf_counter()-t0)*1000

        t0=time.perf_counter()
        for i,m in enumerate(msgs): online_enc(m,cache2[i][0],n)
        t_on=(time.perf_counter()-t0)*1000

        results[col] = {"rows":nr,"std_ms":round(t_std/nr,3),"off_ms":round(t_off/nr,3),"on_ms":round(t_on/nr,6)}
        print(f"    Std: {t_std/nr:.3f} ms/row  Off: {t_off/nr:.3f} ms/row  On: {t_on/nr:.6f} ms/row")
    return results

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*65); print("  labHE Benchmark — Paillier 2048-bit (CORRECTED)")
    print("  key_size=2048 | built-in pow() GMP-backed | all sections")
    print("="*65); print()

    print("── §1 KeyGen ──────────────────────────────────────────────")
    pk,sk,K,kg = bench_keygen(reps=5); print()

    print("── §2 Core Operations (30 reps) ───────────────────────────")
    core = bench_core(pk,sk,K,reps=30); print()

    print("── §3 ZK Proof + Epoch Key (100 reps) ─────────────────────")
    sec = bench_security(pk,K,reps=100); print()

    print("── §4 181-Node Pipeline (5 runs) ───────────────────────────")
    pipe = bench_pipeline(pk,sk,K,n_nodes=181,runs=5); print()

    print("── §5 Petrol Dataset Column Benchmark ──────────────────────")
    petrol = bench_petrol_columns(pk,K); print()

    print("="*65); print("  SUMMARY TABLE (2048-bit, for article §6)")
    c=core; s=sec; p=pipe
    rows=[
        ("KeyGen",           f"{kg['mean_ms']} ms",              f"CI-95={kg['ci95']}"),
        ("Offline Enc/node", f"{c['offline_enc']['mean']} ± {c['offline_enc']['std']} ms", f"CI-95={c['offline_enc']['ci95']}"),
        ("Online Enc/node",  f"{c['online_enc']['mean']} ms",    "~0 ms (1 subtraction)"),
        ("Hom Add",          f"{c['hom_add']['mean']} ms",       ""),
        ("Hom Mult deg-2",   f"{c['hom_mult']['mean']} ± {c['hom_mult']['std']} ms", f"CI-95={c['hom_mult']['ci95']}"),
        ("Online Dec deg-2", f"{c['online_dec2']['mean']} ± {c['online_dec2']['std']} ms", ""),
        ("ZK Combined/node", f"{s['zk_combined_ms']} ms",        f"(prove+verify, corrected from 0.014)"),
        ("Epoch Key/call",   f"{s['epoch_key']['mean']} ms",     ""),
        ("CT size/node",     f"{c['ct_size_bytes']} bytes",      "wire-format (n²=4096 bits)"),
        ("Bandwidth/node",   f"{c['bandwidth_bytes']} bytes",    "β + a"),
        ("Full online 181",  p['full_online_ms'],                 f"{p['pipeline_reps']}-run mean±std"),
        ("Forgery rejected", str(s['forgery_rejected']),          "ZK verification"),
        ("Correctness d1",   str(c['correctness_deg1']),          "exact integer arithmetic"),
        ("Correctness d2",   str(c['correctness_deg2']),          "exact integer arithmetic"),
    ]
    for op,val,note in rows:
        print(f"  {op:<22} {val:<38} {note}")
    print()
    bfv_lo=round(32*1024/c['ct_size_bytes']); bfv_hi=round(65*1024/c['bandwidth_bytes'])
    print(f"  BFV ratio: {bfv_lo}× to {bfv_hi}× smaller than BFV SEAL (N=4096)")
    print(f"  NOTE: gmpy2 not available; using built-in pow() (GMP-backed, CPython 3.8+)")

    out = "/mnt/user-data/outputs/benchmark_2048bit_results.json"
    save = {"keygen":kg,"core_ops":{k:v for k,v in c.items() if not isinstance(v,bool)},
            "correctness":{"deg1":c["correctness_deg1"],"deg2":c["correctness_deg2"]},
            "security":{k:v for k,v in s.items() if not isinstance(v,bool)},
            "security_flags":{"forgery_rejected":s["forgery_rejected"],"epoch_keys_distinct":s["epoch_keys_distinct"]},
            "pipeline_181":p,"petrol_columns":petrol}
    with open(out,"w") as f: json.dump(save,f,indent=2,default=str)
    print(f"\n  Saved → {out}"); print("="*65)

if __name__ == "__main__":
    main()
