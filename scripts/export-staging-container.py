#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
import shutil
root=Path.cwd();dest=Path(sys.argv[1]).resolve()
assert not dest.exists() and root not in dest.parents
subprocess.run(['git','diff','--quiet','HEAD'],check=True)
manifest=json.loads(Path('release/build-inputs.json').read_text())
dest.mkdir();digest=hashlib.sha256();files=[]
for name,expected in sorted(manifest['files'].items()):
 p=root/name; assert not p.is_symlink() and not any(x.startswith('.env') for x in p.parts)
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest();assert h==expected,name
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 digest.update(name.encode()+b'\0'+h.encode()+b'\n');files.append({'path':name,'sha256':h})
receipt={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_snapshot_sha256':digest.hexdigest(),'source_files':len(files),'files':files}
Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
