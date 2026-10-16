from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=Path(__file__).resolve().parents[1];R=C;S=C/'data/global_attribution_summaries';O=Path.cwd();(O/'Supplementary_Data_7').mkdir(exist_ok=True);sp=['Zmay','Sbic','Slyc','Atha'];features=['UTR5LEN','CDSLEN','INTRONLEN','UTR3LEN','UTR5GC','CDSGC','UTR3GC','ORFEXONDENSITY']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(8,7.5))
for row,ti in enumerate(['leaf','root']):
 fv=[];sv=[]
 for s in sp:
  d=pd.read_csv(S/f'{s}_{ti}'/f'{s}_{ti}_mrna_feature_importance.csv').set_index('Feature').reindex(features);fv.append(d.SHAP.to_numpy())
  seq=pd.read_csv(S/f'{s}_{ti}'/f'{s}_{ti}_sequence_importance.csv');sv.append(seq.Total_SHAP.to_numpy())
 for col,(vs,typ) in enumerate([(fv,'Feature'),(sv,'Sequence')]):
  a=np.corrcoef(vs);ax=axs[row,col];im=ax.imshow(a,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(4),sp,rotation=30);ax.set_yticks(range(4),sp);ax.set_title(ti.title()+' '+typ.lower()+' similarity')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(a[i,j])>.6 else 'black')
  fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
  p=O/'Supplementary_Data_7'/f'S3_{ti}_{typ.lower()}_Pearson_similarity.csv';pd.DataFrame(a,index=sp,columns=sp).to_csv(p)
fig.tight_layout(pad=2);fig.savefig(O/'Supplementary_Figure_S3_recomputed.pdf',bbox_inches='tight');plt.close(fig)
print('S3 Pearson matrices independently regenerated with feature-name matching')
