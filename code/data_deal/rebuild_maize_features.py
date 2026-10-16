"""Explicit raw-output reconstruction using the supplied historical maize extractor.
The supplied common feature CSVs remain the reference inputs; this does not recreate
unrecorded transformations that produced them.
"""
from pathlib import Path
import argparse,importlib.util,json,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument("--gff",required=True);p.add_argument("--fasta",required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
source=Path(__file__).resolve().parents[2]/"provenance/original_code/data_deal/Zmay_get_halflife.py"
spec=importlib.util.spec_from_file_location("legacy_extractor",source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
df=m.extract_mrna_features(a.gff,a.fasta)
if df.empty:raise ValueError("No mRNAs extracted; check maize IDs and GFF3 mRNA features")
g=df.groupby("Gene_ID",sort=True);out=g.agg({"UTR5LEN":"max","CDSLEN":"max","INTRONLEN":"sum","UTR3LEN":"max","ORFEXONDENSITY":"mean"})
for gc in ["UTR5GC","CDSGC","UTR3GC"]:
 length=gc.replace("GC","LEN");numer=(df[gc]*df[length]).groupby(df.Gene_ID).sum();denom=g[length].sum();out[gc]=numer.div(denom.replace(0,np.nan)).fillna(0)
cols=["UTR5LEN","CDSLEN","INTRONLEN","UTR3LEN","UTR5GC","CDSGC","UTR3GC","ORFEXONDENSITY"]
a.output_dir.mkdir(parents=True,exist_ok=True);df.to_csv(a.output_dir/"maize_transcript_features_raw.csv",index=False);out[cols].reset_index().rename(columns={"Gene_ID":"gene_id"}).to_csv(a.output_dir/"maize_gene_features_raw.csv",index=False)
(a.output_dir/"provenance.json").write_text(json.dumps({"method":"explicit historical aggregation recipe","n_transcripts":len(df),"n_genes":len(out),"common_table_replaced":False,"standardization_applied":False},indent=2))
