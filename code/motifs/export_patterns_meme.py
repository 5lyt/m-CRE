from pathlib import Path
import argparse,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument("--h5",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.h5,"r") as f,a.output.open("w") as w:
 w.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
 for group,prefix in [("pos_patterns","pos"),("neg_patterns","neg")]:
  for name in sorted(f[group],key=lambda s:int(s.split("_")[-1])):
   d=np.asarray(f[group][name]["sequence"],dtype=float);d=np.maximum(d,0);d=(d+1e-6)/(d.sum(axis=1,keepdims=True)+4e-6);tag=prefix+".p"+name.split("_")[-1];n=int(np.asarray(f[group][name]["seqlets/n_seqlets"]).ravel()[0])
   w.write(f"MOTIF {tag}\nletter-probability matrix: alength= 4 w= {len(d)} nsites= {n} E= 0\n")
   np.savetxt(w,d,fmt="%.8f");w.write("\n")
