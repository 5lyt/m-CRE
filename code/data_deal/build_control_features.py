"""Calculate documented transcript8 or regulatory8 definitions for a new rerun."""
from pathlib import Path
import argparse,itertools,math,re
import numpy as np,pandas as pd
T8=['TXLEN','EXONCOUNT','EXONLEN_SD','TX_GC','TX_CPG_FREQ','TX_TPA_FREQ','TX_AT_SKEW','TX_GC_SKEW']
def transcript_stats(s,lens):
 s=s.upper();counts={b:s.count(b) for b in 'ACGT'};n=sum(counts.values());pairs=[s[i:i+2] for i in range(len(s)-1) if set(s[i:i+2])<=set('ACGT')];den=len(pairs)
 return [len(s),len(lens),float(np.std(lens,ddof=0)),(counts['G']+counts['C'])/n if n else 0,pairs.count('CG')/den if den else 0,pairs.count('TA')/den if den else 0,(counts['A']-counts['T'])/(counts['A']+counts['T']) if counts['A']+counts['T'] else 0,(counts['G']-counts['C'])/(counts['G']+counts['C']) if counts['G']+counts['C'] else 0]
def window_stats(s):
 s=s.upper();hs=[]
 for k in [1,2,3]:
  vals=[s[i:i+k] for i in range(len(s)-k+1) if set(s[i:i+k])<=set('ACGT')]
  if not vals:hs.append(0);continue
  counts=np.unique(vals,return_counts=True)[1];p=counts/counts.sum();hs.append(float(-np.sum(p*np.log2(p))/(2*k)))
 longest=max([len(m.group()) for m in re.finditer(r'A+|C+|G+|T+',s)] or [0]);valid=sum(s.count(b) for b in 'ACGT');return hs+[longest/valid if valid else 0]
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['transcript8','regulatory8'],required=True);p.add_argument('--prepared',type=Path);p.add_argument('--gff',type=Path);p.add_argument('--fasta',type=Path);p.add_argument('--transcript-map',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.kind=='regulatory8':
  if not a.prepared:raise ValueError('--prepared is required')
  z=np.load(a.prepared,allow_pickle=False);rows=[]
  for gid,x in zip(z['gene_ids'],z['sequences']):
   bases=np.array(list('ACGT'));s=''.join(np.where(x.sum(-1)==1,bases[x.argmax(-1)],'N'));rows.append([gid]+window_stats(s[:1500])+window_stats(s[1520:]))
  cols=['gene_id']+[w+'_'+k for w in ['TSS','TTS'] for k in ['H1','H2','H3','MAX_HOMOPOLYMER_FRAC']]
 else:
  if not a.gff or not a.fasta:raise ValueError('--gff and --fasta are required')
  import gffutils
  from pyfaidx import Fasta
  db=gffutils.create_db(str(a.gff),dbfn=':memory:',force=True,keep_order=True,merge_strategy='merge',sort_attribute_values=True);fa=Fasta(str(a.fasta));choices={};mapping={}
  if a.transcript_map:
   mm=pd.read_csv(a.transcript_map);mapping=dict(zip(mm.gene_id,mm.transcript_id))
  txs=list(db.features_of_type('mRNA')) or list(db.features_of_type('transcript'))
  for tx in txs:
   parents=tx.attributes.get('Parent') or tx.attributes.get('gene_id')
   if not parents:continue
   gid=re.sub(r'^gene:','',parents[0]);exons=sorted(db.children(tx,featuretype='exon'),key=lambda q:q.start);lens=[q.end-q.start+1 for q in exons]
   if not lens:continue
   if gid in mapping and str(mapping[gid]) not in [tx.id,re.sub(r'^transcript:','',tx.id)]:continue
   key=(-sum(lens),tx.id)
   if gid not in choices or key<choices[gid][0]:choices[gid]=(key,tx,exons,lens)
  rows=[]
  for gid,(_,tx,exons,lens) in sorted(choices.items()):
   s=''.join(str(fa[tx.seqid][q.start-1:q.end]) for q in exons)
   if tx.strand=='-':s=s.translate(str.maketrans('ACGTacgt','TGCAtgca'))[::-1]
   rows.append([gid]+transcript_stats(s,lens))
  cols=['gene_id']+T8
 a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows,columns=cols).to_csv(a.output,index=False)
if __name__=='__main__':main()
