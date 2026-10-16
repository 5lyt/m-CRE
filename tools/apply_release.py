"""Copy the prepared repository into a clean checkout. No commits, pushes or deletes."""
from pathlib import Path
import argparse,subprocess,shutil,filecmp
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();source=a.source.resolve();target=a.target.resolve()
if source==target or source in target.parents or target in source.parents:raise ValueError('Use separate source and checkout directories')
if not (target/'.git').exists():raise ValueError('Target is not a Git checkout')
url=subprocess.check_output(['git','-C',str(target),'remote','get-url','origin'],text=True).strip()
if not (url.endswith('5lyt/m-CRE.git') or url.endswith('5lyt/m-CRE')):raise ValueError('Target origin must be 5lyt/m-CRE')
if subprocess.check_output(['git','-C',str(target),'status','--porcelain'],text=True).strip():raise ValueError('Checkout has local changes; review those first')
if (target/'LICENSE').read_bytes()!=(source/'LICENSE').read_bytes():raise ValueError('License differs; review explicitly before copying')
changed=[]
for f in sorted(source.rglob('*')):
 if not f.is_file() or '.git' in f.parts or '__pycache__' in f.parts:continue
 rel=f.relative_to(source);dest=target/rel
 if dest.exists() and filecmp.cmp(f,dest,shallow=False):continue
 changed.append(rel.as_posix())
 if not a.dry_run:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
print(('Would copy ' if a.dry_run else 'Copied ')+str(len(changed))+' files; no files deleted and no remote writes performed')
for n in changed:print(n)
