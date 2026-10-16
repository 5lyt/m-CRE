from pathlib import Path
import argparse,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument("--parts-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
parts=[a.parts_root/"motif_inputs_part1",a.parts_root/"motif_inputs_part2"]
arrays=[np.load(d/"candidate_arrays.npz",allow_pickle=False) for d in parts]
ix=np.concatenate([z["row_index"] for z in arrays]);assert np.array_equal(ix,np.arange(1357)),"Missing, duplicated or reordered rows"
genes=np.concatenate([z["gene_ids"] for z in arrays]);meta=pd.concat([pd.read_csv(d/"gene_meta_part.csv") for d in parts],ignore_index=True);assert np.array_equal(genes,meta.gene_id.to_numpy())
o=np.concatenate([z["sequences"] for z in arrays]);s=np.concatenate([z["attributions"] for z in arrays]);assert o.shape==s.shape==(1357,3020,4)
a.output_dir.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output_dir/"ohe_1362.npz",sequences=o);np.savez_compressed(a.output_dir/"attr_1362.npz",attributions=s);meta.to_csv(a.output_dir/"gene_meta_1362_used.csv",index=False);print("Restored 1357 aligned signed input rows; legacy filenames retained")
