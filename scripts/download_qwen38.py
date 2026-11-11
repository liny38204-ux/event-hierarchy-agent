#!/usr/bin/env python3
"""Download and verify pinned model files into an explicit local directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,as_completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--transport',choices=['hub','curl'],default='hub')
    p.add_argument('--endpoint',choices=['https://huggingface.co','https://hf-mirror.com'],default='https://huggingface.co')
    p.add_argument('--workers',type=int,choices=[1,2,4],default=4)
    p.add_argument('--direct',action='store_true');a=p.parse_args()
    dest=a.destination.expanduser().resolve()
    dest.mkdir(parents=True,exist_ok=True)
    lock=json.loads(a.lock.read_text());started=time.time()
    if a.transport=='hub':
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=lock['repo_id'],revision=lock['revision'],
                          local_dir=dest,allow_patterns=[f['name'] for f in lock['files']],max_workers=4)
    else:
        # The pinned manifest already supplies names/sizes/hashes, so no HEAD
        # requests are needed. Resume only our partial files; verify before use.
        def download(item):
            file=dest/item['name']
            if file.exists() and file.stat().st_size==item['size']:
                print(json.dumps({'cached':item['name']}),flush=True);return
            partial=file.with_suffix(file.suffix+'.curl-part')
            url=f"{a.endpoint}/{lock['repo_id']}/resolve/{lock['revision']}/{item['name']}?download=true"
            command=['curl']+(['--noproxy','*'] if a.direct else [])+['--location','--fail','--silent','--show-error',
                '--connect-timeout','15','--max-time','1800','--retry','3','--retry-delay','2',
                '--speed-time','120','--speed-limit','1024','--continue-at','-',
                '--output',str(partial),url]
            result=subprocess.run(command)
            if result.returncode==33:
                # Some small metadata endpoints reject byte ranges. Keep the
                # abandoned partial and restart exactly this file from zero.
                if partial.exists():partial.replace(partial.with_name(partial.name+'.no-range'))
                command=command[:command.index('--continue-at')]+command[command.index('--continue-at')+2:]
                result=subprocess.run(command)
            result.check_returncode()
            if partial.stat().st_size!=item['size']:raise ValueError('Download size mismatch: '+item['name'])
            partial.replace(file);print(json.dumps({'downloaded':item['name'],'bytes':item['size']}),flush=True)
        failures=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(download,item):item['name'] for item in sorted(lock['files'],key=lambda f:f['name'].endswith('.safetensors'))}
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:failures.append({'file':futures[future],'error':repr(exc)})
        if failures:raise RuntimeError('Downloads incomplete: '+json.dumps(failures))
    checked=[]
    for item in lock['files']:
        file=dest/item['name']
        if file.stat().st_size!=item['size']:raise ValueError('Size mismatch: '+item['name'])
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
        h=digest.hexdigest()
        if item['sha256'] and h!=item['sha256']:raise ValueError('Hash mismatch: '+item['name'])
        if not item['sha256'] and item.get('git_blob_sha1'):
            git_hash=hashlib.sha1(b'blob '+str(file.stat().st_size).encode()+b'\0'+file.read_bytes()).hexdigest()
            if git_hash!=item['git_blob_sha1']:raise ValueError('Metadata Git blob mismatch: '+item['name'])
        checked.append(dict(name=item['name'],bytes=file.stat().st_size,sha256=h))
        print(json.dumps(checked[-1]),flush=True)
    (dest/'verified_model.json').write_text(json.dumps({'repo_id':lock['repo_id'],
        'revision':lock['revision'],'transport':a.transport,'endpoint':a.endpoint,'direct':a.direct,
        'files':checked,'elapsed_seconds':time.time()-started},indent=2)+'\n')


if __name__=='__main__':main()
