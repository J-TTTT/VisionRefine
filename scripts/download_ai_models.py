"""Fetch the three official default checkpoints into an isolated project cache.

Default Hugging Face commit IDs are pinned below and recorded in model_revisions.json.
Subsequent runs resume those same revisions; model paths never enter the repository.
"""
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

from download_ai_file import download

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'workspace/cache/ai'
MODELS=['google/owlv2-base-patch16-ensemble','facebook/sam2.1-hiera-tiny','Qwen/Qwen3-VL-2B-Instruct']
DEFAULT_REVISIONS={
    'google/owlv2-base-patch16-ensemble':'cfd3195ba4ea9592eec887ded089f4c08eff231d',
    'facebook/sam2.1-hiera-tiny':'de431c4043854a71d8101e17995dfe596bf101a5',
    'Qwen/Qwen3-VL-2B-Instruct':'89644892e4d85e24eaac8bacfd4f463576704203',
}


def json_url(url):
    with urllib.request.urlopen(url,timeout=90) as response:
        return json.load(response)


def mirror_file(model, name, expected_sha):
    """Use the official Qwen mirror only if its full file hash matches our pinned HF snapshot."""
    if os.environ.get('VISIONREFINE_MODEL_SOURCE') != 'modelscope' or not model.startswith('Qwen/') or not expected_sha:
        return None
    base='https://modelscope.cn/api/v1/models/'+model
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base+'/repo/files?Revision=master&Recursive=true',timeout=30) as response:
        data=json.load(response)
    row=next((item for item in data.get('Data',{}).get('Files',[]) if item.get('Path')==name),None)
    if not row or row.get('Sha256') != expected_sha:
        raise ValueError('Official mirror hash differs from the pinned Hugging Face file')
    return base+'/repo?'+urllib.parse.urlencode({'Revision':row['Revision'],'FilePath':name})


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    revisions_path=CACHE/'model_revisions.json'
    revisions=json.loads(revisions_path.read_text()) if revisions_path.exists() else {}
    paths_path=CACHE/'model_paths.json'
    paths=json.loads(paths_path.read_text()) if paths_path.exists() else {}
    for model in MODELS:
        configured=Path(paths[model]) if model in paths else None
        if configured and configured.is_dir() and (any(configured.glob('*.safetensors')) or any(configured.glob('*.pt'))):
            print('Model already available: '+model,flush=True)
            continue
        # Reuse this user's existing OWLv2 snapshot, without changing that cache.
        if model.startswith('google/') and model not in revisions:
            snapshots=Path.home()/'.cache/huggingface/hub'/('models--'+model.replace('/','--'))/'snapshots'
            existing=next((p for p in snapshots.glob('*') if (p/'config.json').is_file() and any(p.glob('*.safetensors'))),None)
            if existing:
                revisions[model]=existing.name;paths[model]=str(existing)
                revisions_path.write_text(json.dumps(revisions,indent=2));paths_path.write_text(json.dumps(paths,indent=2))
                print('Reusing existing official snapshot: '+model,flush=True)
                continue
        revision=revisions.get(model)
        if not revision:
            revision=DEFAULT_REVISIONS[model]
            revisions[model]=revision
            revisions_path.write_text(json.dumps(revisions,indent=2))
        info=json_url(f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true')
        folder=CACHE/'models'/model.replace('/','--')/revision
        folder.mkdir(parents=True,exist_ok=True)
        if 'sam2' in model:
            wanted=lambda name:name.endswith('.pt')
        else:
            wanted=lambda name:name.endswith(('.json','.txt','.jinja','.safetensors')) and '/' not in name
        files=[row for row in info['siblings'] if wanted(row['rfilename'])]
        if not files:raise ValueError('No model files found: '+model)
        needed=sum(row.get('size',0) for row in files if not (folder/row['rfilename']).exists())
        if shutil.disk_usage(CACHE).free<needed+512*1024*1024:
            raise ValueError('Not enough disk space for '+model)
        for row in files:
            name=row['rfilename'];target=folder/name
            url=f'https://huggingface.co/{model}/resolve/{revision}/{urllib.parse.quote(name)}'
            sha=(row.get('lfs') or {}).get('sha256')
            if (row.get('size') or 0)>8*1024*1024:
                mirror=mirror_file(model,name,sha)
                download(mirror or url,target,sha,workers=32,direct=bool(mirror))
            elif not target.exists():
                with urllib.request.urlopen(url,timeout=120) as response:
                    content=response.read(16*1024*1024)
                if row.get('size') is not None and len(content) != row['size']:
                    raise ValueError('Incomplete model file: '+name)
                temporary=target.with_suffix(target.suffix+'.tmp')
                temporary.write_bytes(content);temporary.replace(target)
        paths[model]=str(folder)
        paths_path.write_text(json.dumps(paths,indent=2))
        print('Model ready: '+model+' @ '+revision,flush=True)


if __name__=='__main__':main()
