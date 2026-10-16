from pathlib import Path
import argparse,json,numpy as np,pandas as pd,h5py
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--array-dir",type=Path);p.add_argument("--output-dir",type=Path,default=Path("verification_new"));a=p.parse_args();r=a.root;a.output_dir.mkdir(parents=True,exist_ok=True)
features=["UTR5LEN","CDSLEN","INTRONLEN","UTR3LEN","UTR5GC","CDSGC","UTR3GC","ORFEXONDENSITY"]
rows=[]
for sp in ["Atha","Sbic","Slyc","Zmay"]:
 d=pd.read_csv(r/"data/processed"/(sp+"_mRNA_features_common.csv"));assert d.gene_id.is_unique and list(d.columns)==["gene_id"]+features and np.isfinite(d[features]).all().all()
 rows += [{"species":sp,"feature":c,"n_genes":len(d),"mean":d[c].mean(),"population_sd":d[c].std(ddof=0)} for c in features]
pd.DataFrame(rows).to_csv(a.output_dir/"feature_summary.csv",index=False)
m=pd.read_csv(r/"data/candidates/gene_meta_1362_used.csv");z=pd.read_csv(r/"data/processed/Zmay_mRNA_features_common.csv");j=m.merge(z,on="gene_id",suffixes=("_motif","_full"),validate="one_to_one");assert len(j)==1357
for c in features:np.testing.assert_array_equal(j[c+"_motif"],j[c+"_full"])
o=s=None
if a.array_dir:
 o=np.load(a.array_dir/"ohe_1362.npz",allow_pickle=False)["sequences"];s=np.load(a.array_dir/"attr_1362.npz",allow_pickle=False)["attributions"];assert o.shape==s.shape==(1357,3020,4)
rows=[];membership=[]
with h5py.File(r/"data/motifs/tfmodisco_results_1362.h5","r") as f:
 assert len(f["pos_patterns"])==13 and len(f["neg_patterns"])==15
 for group in f:
  for pattern in f[group]:
   q=f[group][pattern]["seqlets"];ix=q["example_idx"][()].ravel().astype(int);st=q["start"][()].ravel().astype(int);en=q["end"][()].ravel().astype(int);rv=q["is_revcomp"][()].ravel();assert np.all((ix>=0)&(ix<len(m)))
   tag=group.replace("_patterns","")+".p"+pattern.split("_")[-1];matches=0
   for k,(i,b,e,rev) in enumerate(zip(ix,st,en,rv)):
    membership.append({"pattern":tag,"input_row":i,"gene_id":m.gene_id.iloc[i],"chromosome":int(m.chromosome.iloc[i]),"start":b,"end":e,"is_revcomp":bool(rev)})
    if o is not None:
     osub=o[i,b:e];ssub=s[i,b:e]
     if rev:osub=osub[::-1,::-1];ssub=ssub[::-1,::-1]
     np.testing.assert_array_equal(osub,q["sequence"][k]);np.testing.assert_array_equal(ssub,q["hypothetical_contribs"][k]);matches+=1
   rows.append({"pattern":tag,"n_seqlets":len(ix),"n_unique_genes":len(np.unique(ix)),"exact_matches_checked":matches if o is not None else None})
assert sum(x["n_seqlets"] for x in rows)==2576
pd.DataFrame(rows).to_csv(a.output_dir/"pattern_counts.csv",index=False);pd.DataFrame(membership).to_csv(a.output_dir/"seqlet_membership.csv",index=False)
print(json.dumps({"four_species_feature_tables":"verified","maize_candidate_rows_exactly_matched":len(j),"positive_patterns":13,"negative_patterns":15,"retained_seqlets":2576,"represented_genes":pd.DataFrame(membership).gene_id.nunique(),"seqlet_array_check":a.array_dir is not None}))
