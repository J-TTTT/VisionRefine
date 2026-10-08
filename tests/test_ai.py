"""AI proposals must stay separate, respect manual work and survive reloads."""
import copy
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_video import video_env
from test_video_segmentation import environment, polygon, save, track
from test_segmentation import project
from visionrefine import server
from visionrefine.core.ai import providers
from visionrefine.core.ai.api import create_ai_router
from visionrefine.core.ai.contracts import Configuration, JobInput, Provider, ReviewInput
from visionrefine.core.ai.service import AIService, TERMINAL
from visionrefine.core.dataset_io.service import effective_annotation, select_snapshot
from visionrefine.core.video.service import VideoConflict, VideoService, atomic_json


def wait(service, pid, jid):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        job = service.job(pid, jid)
        if job['status'] in TERMINAL:
            return job
        time.sleep(.01)
    raise AssertionError('AI worker did not finish')


def infer(monkeypatch, output):
    inputs = []
    def run(provider, payload, checkpoint, progress):
        inputs.append(copy.deepcopy(payload))
        checkpoint()
        return copy.deepcopy(output)
    monkeypatch.setattr(providers, 'infer', run)
    return inputs


def run_video(env, monkeypatch, cap, result, **options):
    _, store, pid, vid, _ = env
    service = AIService(store)
    inputs = infer(monkeypatch, result)
    task = service.start(pid, JobInput(capability=cap, video_id=vid, **options))
    job = wait(service, pid, task['id'])
    return service, job, inputs


def accept(service, pid, job, ids=None):
    return service.review(pid, job['id'], ReviewInput(action='accept', item_ids=ids or [i['id'] for i in job['items']]))


def test_default_config_bindings_and_validation(environment):
    _, store, pid, _, _ = environment
    app = FastAPI(); app.include_router(create_ai_router(lambda: store))
    with TestClient(app) as client:
        config = client.get('/api/ai/config').json()
        assert len(config['providers']) == 3
        assert config['providers'][2]['model'].endswith('2B-Instruct')
        config['providers'].append({**config['providers'][2], 'id':'custom', 'name':'My model'})
        assert client.put('/api/ai/config', json=config).status_code == 200
        assert client.put('/api/ai/config', json=config).status_code == 409
        binding = {'revision':0, 'providers':{'video_caption':'custom'}}
        response = client.put(f'/api/ai/projects/{pid}/bindings', json=binding)
        assert response.status_code == 200
        assert AIService(store).provider(pid, 'video_caption')['id'] == 'custom'
        invalid = {**response.json(), 'providers':{'video_segmentation':'custom'}}
        assert client.put(f'/api/ai/projects/{pid}/bindings', json=invalid).status_code == 400
        latest = client.get('/api/ai/config').json()
        latest['providers'].pop()
        assert client.put('/api/ai/config', json=latest).status_code == 400
    for url in ['file:///tmp/x', 'http://user:password@localhost:8030', 'http://localhost/?token=secret']:
        with pytest.raises(ValueError):
            Provider(**{**config['providers'][0], 'base_url':url})
    with pytest.raises(ValueError):
        Provider(**{**config['providers'][1], 'protocol':'openai_compatible'})


def test_caption_is_proposal_then_unreviewed_draft_with_provenance(environment, monkeypatch):
    client, store, pid, vid, base = environment
    assert client.put(base+'/annotations', json={'captions':[{'id':'human','text':'人工描述','reviewed':True}]}).status_code == 200
    service, job, inputs = run_video(environment, monkeypatch, 'video_caption', {'captions':[{'text':'可见物体正在移动'}]})
    assert job['status'] == 'completed', job
    assert [f['timestamp'] for f in inputs[0]['frames']] == [0,.1,.4,.8]
    assert len(VideoService(store).annotations(pid,vid)['captions']) == 1
    accepted = accept(service, pid, job)
    doc = client.get(base+'/annotations').json()
    assert doc['captions'][0]['reviewed'] is True
    new = doc['captions'][1]
    assert new['basis'] == 'visual' and new['reviewed'] is False
    assert new['provenance']['model'].endswith('2B-Instruct')
    assert doc['review']['captions'] == 'unreviewed'
    assert accept(service, pid, accepted) == accepted
    assert AIService(type(store)(store.root)).job(pid, job['id']) == accepted
    bundle = VideoService(store).export(pid, 'all')
    assert bundle['schema_version'] == '3.0'
    assert bundle['videos'][0]['annotations']['captions'][1]['provenance'] == new['provenance']
    reviewed = VideoService(store).export(pid, 'reviewed')
    assert [c['id'] for c in reviewed['videos'][0]['annotations']['captions']] == ['human']
    VideoService(store).import_annotations(pid, bundle)
    assert not any(c['reviewed'] for c in VideoService(store).annotations(pid,vid)['captions'])


def test_partial_accept_and_reject_and_stale_revision(environment, monkeypatch):
    client, _, pid, _, base = environment
    service, job, _ = run_video(environment, monkeypatch, 'video_events', {'events':[
        {'kind':'interval','start':0,'end':.3,'text':'动作一'}, {'kind':'point','start':.4,'text':'动作二'}]})
    assert job['status']=='completed', job
    accept(service,pid,job,[job['items'][0]['id']])
    doc=client.get(base+'/annotations').json()
    doc['events'][0]['text']='人工修正'
    assert client.put(base+'/annotations',json=doc).status_code==200
    with pytest.raises(VideoConflict):
        accept(service,pid,job,[job['items'][1]['id']])
    rejected=service.review(pid,job['id'],ReviewInput(action='reject',item_ids=[job['items'][1]['id']]))
    assert [i['status'] for i in rejected['items']]==['accepted','rejected']
    assert len(client.get(base+'/annotations').json()['events'])==1


@pytest.mark.parametrize('events',[
    [{'start':0,'end':2,'text':'beyond video'}],
    [{'start':0,'end':.5,'label_id':'invented','text':''}],
    [{'start':0,'end':.5,'text':'x','track_ids':['guessed-object']}],
    [{'start':.8,'end':.1,'text':'reversed'}],
])
def test_invalid_model_events_fail_without_publishing(environment,monkeypatch,events):
    _,store,pid,vid,_=environment
    _,job,_=run_video(environment,monkeypatch,'video_events',{'events':events})
    assert job['status']=='failed'
    assert VideoService(store).annotations(pid,vid)['revision']==0


def test_propagation_protects_anchors_absence_and_corrections(environment,monkeypatch):
    client,store,pid,vid,base=environment
    row=track(frames=(0,))
    row['visibility_ranges']=[{'start_frame':2,'end_frame':2,'visibility':'occluded','reviewed':True}]
    original=save(client,base,[row])
    outputs={'keyframes':[{'frame_index':i,'visibility':'visible','geometry':polygon()} for i in range(4)]}
    service,job,inputs=run_video(environment,monkeypatch,'video_segmentation',outputs,track_id=row['id'],seed_frame=0)
    assert job['status']=='completed',job
    assert [i['value']['frame_index'] for i in job['items']]==[1,3]
    assert inputs[0]['seeds'][0]['frame_index']==0
    accept(service,pid,job)
    document=client.get(base+'/segmentation').json()
    assert document['tracks'][0]['keyframes'][0]==original['tracks'][0]['keyframes'][0]
    assert document['tracks'][0]['visibility_ranges']==original['tracks'][0]['visibility_ranges']
    corrected=document['tracks'][0]['keyframes'][1]
    corrected['geometry']['polygons'][0][0]=[3,3]
    document=client.put(base+'/segmentation',json=document).json()
    assert 'provenance' not in document['tracks'][0]['keyframes'][1]
    _,next_job,next_inputs=run_video(environment,monkeypatch,'video_segmentation',outputs,track_id=row['id'],seed_frame=1)
    assert [s['frame_index'] for s in next_inputs[0]['seeds']]==[0,1]
    assert [i['value']['frame_index'] for i in next_job['items']]==[3]


def test_propagation_rejects_stale_track_revision(environment,monkeypatch):
    client,_,pid,_,base=environment
    save(client,base,[track(frames=(0,))])
    service,job,_=run_video(environment,monkeypatch,'video_segmentation',{'keyframes':[{'frame_index':1,'visibility':'visible','geometry':polygon()}]},track_id='person-1',seed_frame=0)
    save(client,base,[track(frames=(0,3))])
    with pytest.raises(VideoConflict):
        accept(service,pid,job)


def test_cancellation_restart_and_retry(environment,monkeypatch):
    _,store,pid,vid,_=environment
    ready=threading.Event()
    def slow(provider,payload,checkpoint,progress):
        ready.set()
        while True:
            time.sleep(.01);checkpoint()
    monkeypatch.setattr(providers,'infer',slow)
    service=AIService(store)
    job=service.start(pid,JobInput(capability='video_caption',video_id=vid))
    assert ready.wait(3)
    service.cancel(pid,job['id'])
    assert wait(service,pid,job['id'])['status']=='cancelled'
    infer(monkeypatch,{'captions':[]})
    retry=service.retry(pid,job['id'])
    assert wait(service,pid,retry['id'])['status']=='completed'
    orphan={**retry,'id':'orphan','status':'running'}
    atomic_json(service.path(pid,'orphan'),orphan)
    assert service.job(pid,'orphan')['status']=='interrupted'


def test_image_draft_preserves_human_objects_and_reviewed_export(project,monkeypatch):
    client,p=project
    pid=p['id'];url=f'/api/projects/{pid}/annotations'
    original={'id':'human','kind':'polygon','label':'cell','polygons':[[[1,1],[8,1],[8,8],[1,8]]]}
    human=client.put(url,json={'image':'sample.png','objects':[original]}).json()
    infer(monkeypatch,{'objects':[{**polygon(),'label':'cell'}]})
    service=AIService(server.store)
    req=JobInput(capability='image_segmentation',image='sample.png',label='cell',prompt={'points':[[5,5]],'point_labels':[1]})
    job=wait(service,pid,service.start(pid,req)['id'])
    assert job['status']=='completed',job
    assert client.get(url,params={'image':'sample.png'}).json()==human
    accept(service,pid,job)
    draft=client.get(url,params={'image':'sample.png'}).json()
    assert draft['status']=='ai_suggestion'
    assert draft['objects'][0]==human['objects'][0]
    assert len(draft['objects'])==2
    live=server.store.get(pid)
    snapshot,_=select_snapshot(server.store,live,'reviewed',[])
    assert len(snapshot.images[0].objects)==1
    snapshot,_=select_snapshot(server.store,live,'reviewed_or_ai',[])
    assert len(snapshot.images[0].objects)==2
    assert [o.source for o in snapshot.images[0].objects]==['human_reviewed','ai_suggestion']
    saved=client.put(url,json={'image':'sample.png','objects':draft['objects']}).json()
    assert saved['status']=='human_reviewed'
    assert client.get(url,params={'image':'sample.png'}).json()==saved
    assert saved['objects'][1]['provenance']['ai']['job_id']==job['id']


def test_image_accept_checks_source_and_geometry(project,monkeypatch):
    client,p=project;pid=p['id']
    infer(monkeypatch,{'objects':[{**polygon(),'label':'cell'}]})
    service=AIService(server.store)
    req=JobInput(capability='image_segmentation',image='sample.png',label='cell',prompt={'box':[1,1,20,20]})
    job=wait(service,pid,service.start(pid,req)['id'])
    client.put(f'/api/projects/{pid}/annotations',json={'image':'sample.png','objects':[]})
    with pytest.raises(VideoConflict):accept(service,pid,job)
    infer(monkeypatch,{'objects':[{'kind':'polygon','label':'cell','polygons':[[[0,0],[99999,0],[1,3]]]}]})
    failed=wait(service,pid,service.start(pid,req)['id'])
    assert failed['status']=='failed'


def test_accept_recovers_when_job_write_fails_after_annotation_commit(environment,monkeypatch):
    client,store,pid,vid,base=environment
    service,job,_=run_video(environment,monkeypatch,'video_caption',{'captions':[{'text':'恢复测试'}]})
    real_write=service.write_job
    def fail_once(*args):raise OSError('simulated exit after annotation commit')
    monkeypatch.setattr(service,'write_job',fail_once)
    with pytest.raises(OSError):accept(service,pid,job)
    assert len(client.get(base+'/annotations').json()['captions'])==1
    monkeypatch.setattr(service,'write_job',real_write)
    saved=accept(service,pid,job)
    assert saved['items'][0]['status']=='accepted'
    assert len(client.get(base+'/annotations').json()['captions'])==1
    assert 'items' not in service.jobs(pid)[0]
    assert service.jobs(pid)[0]['counts']['accepted']==1
