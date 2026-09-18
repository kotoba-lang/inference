# CPU dequant oracle for the :gpu/compute K-quant dot guests (verify/native/gpu). Python because it is a
# TEST ORACLE beside numpy, not operations tooling (CLAUDE.md kbb-first is for tooling); a .cljk twin is debt.
# GGUF tensor directory + CPU dequant reference for Q5_K / Q6_K / IQ4_XS rows.
# Prints "name type offset(bytes, absolute in file) dims" and writes a reference dot.
import struct, sys, numpy as np, json
def read_gguf_dir(path):
    f=open(path,'rb'); r=f.read
    def u32(): return struct.unpack('<I',r(4))[0]
    def u64(): return struct.unpack('<Q',r(8))[0]
    def s():
        n=u64(); return r(n).decode()
    def val(t):
        if t in (0,1): return struct.unpack('<B' if t==0 else '<b',r(1))[0]
        if t in (2,3): return struct.unpack('<H' if t==2 else '<h',r(2))[0]
        if t in (4,5): return struct.unpack('<I' if t==4 else '<i',r(4))[0]
        if t==6: return struct.unpack('<f',r(4))[0]
        if t==7: return struct.unpack('<B',r(1))[0]
        if t==8: return s()
        if t==9:
            et=u32(); n=u64(); return [val(et) for _ in range(n)]
        if t in (10,11): return struct.unpack('<Q' if t==10 else '<q',r(8))[0]
        if t==12: return struct.unpack('<d',r(8))[0]
        raise ValueError(t)
    assert r(4)==b'GGUF'; ver=u32(); nt=u64(); nkv=u64()
    kv={}
    for _ in range(nkv):
        k=s(); t=u32(); kv[k]=val(t)
    align=kv.get('general.alignment',32)
    tensors=[]
    for _ in range(nt):
        name=s(); nd=u32(); dims=[u64() for _ in range(nd)]; typ=u32(); off=u64()
        tensors.append((name,typ,off,dims))
    pos=f.tell(); data_start=(pos+align-1)//align*align
    return kv,tensors,data_start
KV=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],dtype=np.float32)
def f16(b): return np.frombuffer(b,dtype=np.float16)[0].astype(np.float32)
def deq_iq4xs(blk):
    d=f16(blk[0:2]); sh=struct.unpack('<H',blk[2:4])[0]; sl=blk[4:8]; qs=np.frombuffer(blk[8:136],dtype=np.uint8)
    out=np.zeros(256,np.float32)
    for ib in range(8):
        ls=((sl[ib//2]>>(4*(ib%2)))&0xf)|(((sh>>(2*ib))&3)<<4); dl=np.float32(d*(ls-32))
        q=qs[16*ib:16*ib+16]; out[32*ib:32*ib+16]=dl*KV[q&0xf]; out[32*ib+16:32*ib+32]=dl*KV[q>>4]
    return out
def q4k_scales(sc):
    scales=np.zeros(8,np.int32); mins=np.zeros(8,np.int32)
    for j in range(8):
        if j<4: scales[j]=sc[j]&63; mins[j]=sc[j+4]&63
        else: scales[j]=(sc[j+4]&0xf)|((sc[j-4]>>6)<<4); mins[j]=(sc[j+4]>>4)|((sc[j]>>6)<<4)
    return scales,mins
def deq_q5k(blk):
    d=f16(blk[0:2]); dmin=f16(blk[2:4]); scales,mins=q4k_scales(np.frombuffer(blk[4:16],dtype=np.uint8))
    qh=np.frombuffer(blk[16:48],dtype=np.uint8); qs=np.frombuffer(blk[48:176],dtype=np.uint8)
    out=np.zeros(256,np.float32)
    for g in range(4):  # 64-value groups
        lo=qs[32*g:32*g+32]&0xf; hi=qs[32*g:32*g+32]>>4
        b_lo=((qh>>(2*g))&1)<<4; b_hi=((qh>>(2*g+1))&1)<<4
        j0=2*g; j1=2*g+1
        out[64*g:64*g+32]=d*scales[j0]*(lo|b_lo).astype(np.float32)-dmin*mins[j0]
        out[64*g+32:64*g+64]=d*scales[j1]*(hi|b_hi).astype(np.float32)-dmin*mins[j1]
    return out
def deq_q6k(blk):
    ql=np.frombuffer(blk[0:128],dtype=np.uint8); qh=np.frombuffer(blk[128:192],dtype=np.uint8)
    sc=np.frombuffer(blk[192:208],dtype=np.int8); d=f16(blk[208:210])
    out=np.zeros(256,np.float32)
    for g in range(2):  # 128-value halves
        qlg=ql[64*g:64*g+64]; qhg=qh[32*g:32*g+32]; scg=sc[8*g:8*g+8]
        for l in range(32):
            q1=((qlg[l]&0xf)|(((qhg[l]>>0)&3)<<4))-32
            q2=((qlg[l+32]&0xf)|(((qhg[l]>>2)&3)<<4))-32
            q3=((qlg[l]>>4)|(((qhg[l]>>4)&3)<<4))-32
            q4=((qlg[l+32]>>4)|(((qhg[l]>>6)&3)<<4))-32
            out[128*g+l]=d*scg[l//16]*q1; out[128*g+32+l]=d*scg[2+l//16]*q2
            out[128*g+64+l]=d*scg[4+l//16]*q3; out[128*g+96+l]=d*scg[6+l//16]*q4
    return out
BLOCK={13:(176,deq_q5k),14:(210,deq_q6k),23:(136,deq_iq4xs)}
if __name__=='__main__':
    path=sys.argv[1]; names=sys.argv[2:]
    kv,tensors,ds=read_gguf_dir(path)
    print("data_start",ds)
    f=open(path,'rb')
    for name,typ,off,dims in tensors:
        if name in names:
            cols,rows=dims[0],dims[1]; bb,deq=BLOCK[typ]; nb=cols//256; rowbytes=nb*bb
            print(name,typ,ds+off,dims,"bytes",rows*rowbytes)
            rng=np.random.default_rng(11); x=rng.standard_normal(cols).astype(np.float32); x.tofile(name.replace('.','_')+'.x.f32')
            f.seek(ds+off); raw=f.read(rows*rowbytes)
            nref=min(rows,64)  # reference for the first 64 rows (CPU dequant is slow)
            ref=np.zeros(nref,np.float64)
            for r in range(nref):
                row=np.concatenate([deq(raw[r*rowbytes+b*bb:r*rowbytes+(b+1)*bb]) for b in range(nb)])
                ref[r]=np.dot(row.astype(np.float64),x.astype(np.float64))
            np.save(name.replace('.','_')+'.ref.npy',ref)
