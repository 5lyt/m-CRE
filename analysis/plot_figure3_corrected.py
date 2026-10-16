from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];S=R/'data/global_attribution_summaries';O=Path.cwd();O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
sp=['Zmay','Sbic','Slyc','Atha'];names=['Maize','Sorghum','Tomato','Arabidopsis'];colors=['#2479b4','#ef8b24','#399b4a','#d14949'];markers=['o','s','*','^']
features=['UTR3GC','CDSGC','ORFEXONDENSITY','UTR5GC','INTRONLEN','UTR3LEN','UTR5LEN','CDSLEN']
fig=plt.figure(figsize=(8.3,10.8));gs=fig.add_gridspec(4,2,height_ratios=[1.2,1,1,1],hspace=.52,wspace=.35)
for col,tissue in enumerate(['leaf','root']):
 ax=fig.add_subplot(gs[0,col])
 for ss,name,c,m in zip(sp,names,colors,markers):
  f=pd.read_csv(S/f'{ss}_{tissue}'/f'{ss}_{tissue}_mrna_feature_importance.csv').set_index('Feature').reindex(features)
  ax.scatter(f.SHAP,np.arange(8),label=name,c=c,marker=m,s=27,zorder=3)
 ax.set_yticks(range(8),features);ax.invert_yaxis();ax.set_xlabel('Mean absolute attribution');ax.set_title(f'{chr(97+col)}  {tissue.capitalize()} transcript features',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(fontsize=6.5,loc='lower right')
# The input contains separate boundary windows. Draw each on its own coordinate axis.
for row,tissue in enumerate(['leaf','root'],1):
 axes=[fig.add_subplot(gs[row,c]) for c in range(2)]
 for ss,name,c in zip(sp,names,colors):
  d=pd.read_csv(S/f'{ss}_{tissue}'/f'{ss}_{tissue}_sequence_importance.csv')
  y=d.Total_SHAP.to_numpy()
  axes[0].plot(np.arange(1500)-1000,y[:1500],color=c,lw=1,label=name)
  axes[1].plot(np.arange(1500)-500,y[1520:],color=c,lw=1,label=name)
 for j,ax in enumerate(axes):
  ax.axvline(0,color='grey',ls='--',lw=.8);ax.grid(alpha=.16);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.set_xlabel('Position relative to annotated '+('TSS' if j==0 else 'TTS')+' (nt)');ax.set_ylim(bottom=0)
  ax.set_title((chr(98+row)+'  ' if j==0 else '')+tissue.capitalize()+('  5 prime boundary' if j==0 else '  3 prime boundary'),loc='left',fontweight='bold')
 axes[0].set_ylabel('Mean absolute attribution');axes[1].legend(fontsize=6.5)
d=pd.read_csv(S/'Zmay_leaf/Zmay_leaf_sequence_importance.csv')
# Legacy dataframe labels used ATCG for arrays whose actual channel order is ACGT.
corrected=pd.DataFrame({'position_one_based':d.Position,'A':d.A,'C':d['T'],'G':d.C,'T':d.G,'channel_mean':d.Total_SHAP})
corrected.to_csv(O/'Figure3_Zmay_leaf_corrected_nucleotide_channels.csv',index=False)
for c,sl,x in [(0,slice(0,1500),np.arange(1500)-1000),(1,slice(1520,3020),np.arange(1500)-500)]:
 ax=fig.add_subplot(gs[3,c])
 for nt,color in zip('ACGT',['#2479b4','#ef8b24','#399b4a','#d14949']):ax.plot(x,corrected[nt].to_numpy()[sl],label=nt,c=color,lw=.85)
 ax.axvline(0,color='grey',ls='--',lw=.8);ax.grid(alpha=.16);ax.set_xlabel('Position relative to annotated '+('TSS' if c==0 else 'TTS')+' (nt)');ax.set_title(('e  ' if c==0 else '')+'Maize leaf nucleotide channels',loc='left',fontweight='bold');ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.set_ylim(bottom=0)
 if c==0:ax.set_ylabel('Mean absolute attribution')
 else:ax.legend(ncol=4,fontsize=7)
fig.subplots_adjust(left=.13,right=.97,top=.97,bottom=.055);fig.savefig(O/'Figure_3.pdf');fig.savefig(O/'Figure_3_preview.png',dpi=120);plt.close(fig)
print('Figure3 and corrected channel table written')
