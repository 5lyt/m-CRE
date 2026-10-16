"""Create a row-aligned, masked NPZ using the inherited sequence/label/whitelist loader."""
from pathlib import Path
import argparse,importlib.util,numpy as np,os
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ['MCRE_DATA_DIR']=a.data_dir
source=Path(__file__).resolve().parents[1]/'model/DeepCRE_mRNA_BiLSTM_chr.py'
spec=importlib.util.spec_from_file_location('m4_source',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sp,_=m.resolve_task_parts(a.task);whitelist,_,_=m.load_validation_gene_whitelist(sp);base=m.load_species_feature_table(sp,whitelist);labels=m.load_expression_labels(a.task);d=m.make_task_dataframe(base,labels,a.task)
x=np.stack(d.encoded_seq.to_numpy()).astype(np.float32);x[:,1000:1003]=0;x[:,1500:1520]=0;x[:,2017:2020]=0
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,sequences=x,features=d[m.MRNA_COLS].to_numpy(np.float32),feature_names=np.asarray(m.MRNA_COLS,dtype=str),gene_ids=d.gene_id.to_numpy(dtype=str),chromosomes=d.chromosome.to_numpy(dtype=str),labels=d.label.to_numpy(np.int32),is_val_gene=d.is_val_gene.to_numpy(bool))
