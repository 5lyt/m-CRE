import sys,json,platform,importlib.metadata as md
names=["numpy","pandas","scipy","scikit-learn","tensorflow","keras","shap","modisco-lite","h5py","pyranges","pyfaidx"]
out={"python":sys.version,"platform":platform.platform(),"scope":"current execution environment, not historical training environment","packages":{}}
for n in names:
 try:out["packages"][n]=md.version(n)
 except md.PackageNotFoundError:out["packages"][n]=None
print(json.dumps(out,indent=2))
