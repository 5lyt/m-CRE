from pathlib import Path
import argparse,subprocess,hashlib,json
def checksum(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument("--query",type=Path,required=True);p.add_argument("--database",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--executable",default="tomtom");a=p.parse_args()
for f in [a.query,a.database]:
 if not f.is_file():raise FileNotFoundError(f)
a.output_dir.mkdir(parents=True,exist_ok=True)
# These are explicit rerun settings; the historical command/database log has not been recovered.
cmd=[a.executable,"-oc",str(a.output_dir),"-dist","pearson","-thresh","0.01",str(a.query),str(a.database)]
version=subprocess.run([a.executable,"-version"],capture_output=True,text=True,check=True)
manifest={"command":cmd,"tomtom_version":version.stdout.strip() or version.stderr.strip(),"database_sha256":checksum(a.database),"query_sha256":checksum(a.query),"historical_run_reproduced":False}
(a.output_dir/"execution_manifest.json").write_text(json.dumps(manifest,indent=2));subprocess.run(cmd,check=True)
