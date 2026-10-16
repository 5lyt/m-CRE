from pathlib import Path
import json, argparse, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--array-dir",type=Path);p.add_argument("--output",type=Path,default=Path("verification_report.json"));args=p.parse_args();r=args.root
m=pd.read_csv(r/"data/candidates/gene_meta_1362_used.csv");c=pd.read_csv(r/"data/candidates/Zmay_WGCNA_supported_high_expression_candidate_genes_1362_info.csv")
assert len(c)==1362 and c.gene_id.is_unique and len(m)==1357 and m.gene_id.is_unique
assert m.true_label.eq(1).all() and np.array_equal(m.chromosome.astype(str),m.model_chr.astype(str))
assert m.pred_label.eq((m.pred_prob_high>=.5).astype(int)).all()
f=pd.read_csv(r/"data/model_metrics/Zmay_leaf/output_DeepCRE_mRNA_BiLSTM_chr_oof_predictions.csv")
a=m.merge(f,on="gene_id",validate="one_to_one",suffixes=("_candidate","_oof"))
report={"candidate_count":len(c),"actual_input_count":len(m),"excluded_ids":sorted(set(c.gene_id)-set(m.gene_id)),"candidate_replay_high":int(m.pred_label.sum()),"original_oof_count":len(f),"original_oof_pooled_auroc":roc_auc_score(f.label,f.pred_prob),"original_oof_candidate_overlap":len(a),"overlap_original_high":int(a.pred_prob.ge(.5).sum())}
fold=pd.read_csv(r/"results/revised/Supplementary_Data_1_fold_level_model_performance.csv");out=[]
for w,b in [("M4","M1"),("M4","M2"),("M3","M1"),("M2","M1"),("M4","M3")]:
 rows=[]
 for task,g in fold.groupby("task"):
  x=g[g.model_key==w][["fold","auc"]].merge(g[g.model_key==b][["fold","auc"]],on="fold",validate="one_to_one")
  stat,prob=ttest_rel(x.auc_x,x.auc_y);rows.append({"task":task,"contrast":w+"-"+b,"mean_delta":float((x.auc_x-x.auc_y).mean()),"p":float(prob)})
 order=np.argsort([q["p"] for q in rows]);adj=np.maximum.accumulate(np.array([rows[i]["p"] for i in order])*np.arange(len(rows),0,-1))
 for i,q in zip(order,adj):rows[i]["p_holm_8"]=float(min(q,1))
 out.extend(rows)
report["fold_contrasts"]=out
if args.array_dir:
 o=np.load(args.array_dir/"ohe_1362.npz",allow_pickle=False)["sequences"];s=np.load(args.array_dir/"attr_1362.npz",allow_pickle=False)["attributions"]
 assert o.shape==s.shape==(1357,3020,4) and np.isfinite(s).all()
 assert np.isin(o,[0,1]).all() and np.isin(o.sum(-1),[0,1]).all()
 for l,u in [(1000,1003),(1500,1520),(2017,2020)]:assert not o[:,l:u].any()
 report["arrays"]={"shape":list(s.shape),"negative_fraction":float((s<0).mean()),"signed":bool((s<0).any() and (s>0).any())}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="fold_contrasts"},indent=2))
