import re,sys,numpy as np
name=sys.argv[1]; out=open(sys.argv[2]).read()
m=re.search(r':result-utf8-hex "([0-9a-f]+)"', out)
if not m: print(name, "TRAP:", out[:300].replace("\n"," ")); sys.exit(1)
txt=bytes.fromhex(m.group(1)).decode(); one,ten,hexo=txt.split("|")
got=np.frombuffer(bytes.fromhex(hexo),dtype=np.float32)
xname={'attn_qkv':'blk_0_attn_qkv_weight','attn_gate':'blk_0_attn_gate_weight','ffn_gate_exp0':'blk_0_ffn_gate_exps_weight','output':'output_weight'}[name]
ref=np.load(xname+'.ref.npy')
rel=np.max(np.abs(got-ref)/np.maximum(np.abs(ref),1e-2))
nbytes={'attn_qkv':11534336,'attn_gate':4456448,'ffn_gate_exp0':557056,'output':417177600}[name]
print(f"{name}: one dispatch {int(one)/1e6:.3f} ms, 10-in-one-buffer {int(ten)/1e6:.2f} ms -> {10*nbytes/int(ten):.1f} GB/s ; max rel err vs CPU dequant {rel:.2e} ; got {got[:2]} ref {ref[:2]}")
