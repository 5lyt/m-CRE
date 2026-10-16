"""Portable new-run entry point for ten comparisons. Never overwrites supplied paper results."""
from pathlib import Path
import argparse,json,importlib.util,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--model',choices=['M0','M1','M2','M3','M4','M4-T8','M4-R8','M4-P','T8-only','R8-only'],required=True);p.add_argument('--control-features',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100);a=p.parse_args()
import tensorflow as tf
z=np.load(a.prepared,allow_pickle=False);seq=z['sequences'];features=z['features'];ids=z['gene_ids'];chrom=z['chromosomes'].astype(str);y=z['labels'];white=z['is_val_gene'];names=z['feature_names'].tolist()
assert seq.shape[1:]==(3020,4) and features.shape==(len(y),8) and len(np.unique(ids))==len(ids) and set(np.unique(y))<={0,1}
if a.model in ['M4-T8','M4-R8','T8-only','R8-only']:
 if a.control_features is None:raise ValueError('Control feature CSV is required')
 f=pd.read_csv(a.control_features).set_index('gene_id');assert f.index.is_unique and f.shape[1]==8
 if not set(ids)<=set(f.index):raise ValueError('Control table must cover the prepared cohort; do not silently change eligible genes')
 names=f.columns.tolist();features=f.loc[ids].to_numpy(np.float32)
a.output_dir.mkdir(parents=True,exist_ok=True)
if any(a.output_dir.iterdir()):raise FileExistsError('Use a new empty output directory for each model run')
(a.output_dir/'run_config.json').write_text(json.dumps({**vars(a),'prepared':str(a.prepared),'control_features':str(a.control_features) if a.control_features else None,'output_dir':str(a.output_dir),'scope':'new portable rerun; not a claim that historical results were regenerated'},indent=2))
source={'M1':'DeepCRE_chr.py','M2':'DeepCRE_BiLSTM_chr.py','M3':'DeepCRE_mRNA_chr.py'}.get(a.model,'DeepCRE_mRNA_BiLSTM_chr.py')
sp=importlib.util.spec_from_file_location('architecture_source',Path(__file__).with_name(source));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
metrics=[];oof=[]
for fold,ch in enumerate(sorted(np.unique(chrom),key=lambda x:(not x.isdigit(),int(x) if x.isdigit() else x)),1):
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(a.seed+fold);rng=np.random.RandomState(a.seed+fold)
 tr=np.flatnonzero(chrom!=ch);va=np.flatnonzero((chrom==ch)&white)
 if not len(va):continue
 n=min(np.sum(y[tr]==0),np.sum(y[tr]==1))
 if n==0:raise ValueError('A training fold lacks one class')
 bal=np.concatenate([rng.choice(tr[y[tr]==c],n,replace=False) for c in [0,1]]);rng.shuffle(bal)
 xt=features[bal].copy();xv=features[va].copy()
 if a.model=='M4-P':xt=xt[rng.permutation(len(xt))];xv=xv[rng.permutation(len(xv))]
 sc=StandardScaler().fit(xt);xt=sc.transform(xt).astype(np.float32);xv=sc.transform(xv).astype(np.float32)
 auxonly=a.model in ['M0','T8-only','R8-only'];seqonly=a.model in ['M1','M2']
 if auxonly:
  ip=tf.keras.Input((8,));h=tf.keras.layers.Dense(128,activation='relu')(ip);h=tf.keras.layers.Dropout(.25)(h);h=tf.keras.layers.Dense(64,activation='relu')(h);h=tf.keras.layers.Dropout(.25)(h);out=tf.keras.layers.Dense(1,activation='sigmoid')(h);model=tf.keras.Model(ip,out);model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='binary_crossentropy');train_x=xt;val_x=xv
 else:
  model=m.build_model(3020,8);train_x=seq[bal] if seqonly else [seq[bal],xt];val_x=seq[va] if seqonly else [seq[va],xv]
 state={'feature_order':names,'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'var':sc.var_.tolist(),'n_samples_seen':int(sc.n_samples_seen_),'held_out_chromosome':ch,'fit_population':'balanced_training_only'}
 (a.output_dir/f'preprocessing_chr{ch}.json').write_text(json.dumps(state,indent=2));pd.DataFrame({'gene_id':ids[bal],'label':y[bal]}).to_csv(a.output_dir/f'balanced_training_chr{ch}.csv',index=False)
 callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=5,factor=.1)]
 model.fit(train_x,y[bal],validation_data=(val_x,y[va]),epochs=a.epochs,batch_size=64,callbacks=callbacks,verbose=2);prob=model.predict(val_x,batch_size=64,verbose=0).ravel();pred=prob>=.5;model.save(a.output_dir/f'model_chr{ch}.keras')
 metrics.append({'model':a.model,'fold':fold,'chromosome':ch,'n_train_balanced':len(bal),'n_val':len(va),'n_val_high':int(y[va].sum()),'n_val_low':int((y[va]==0).sum()),'auc':roc_auc_score(y[va],prob) if len(np.unique(y[va]))==2 else None,'accuracy':accuracy_score(y[va],pred),'precision':precision_score(y[va],pred,zero_division=0),'recall':recall_score(y[va],pred,zero_division=0),'f1':f1_score(y[va],pred,zero_division=0)})
 oof.extend({'gene_id':ids[i],'fold':fold,'chromosome':ch,'label':int(y[i]),'pred_prob':float(pp)} for i,pp in zip(va,prob))
pd.DataFrame(metrics).to_csv(a.output_dir/'fold_metrics.csv',index=False);pd.DataFrame(oof).to_csv(a.output_dir/'oof_predictions.csv',index=False)
