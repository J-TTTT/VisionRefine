"""HTTP adapters and worker transport tested without installing GPU dependencies."""
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionrefine.core.ai import providers
from visionrefine import ai_worker


def frame():
    buffer=io.BytesIO();Image.new('RGB',(40,24),'red').save(buffer,'JPEG')
    return {'frame_index':0,'timestamp':0,'jpeg':base64.b64encode(buffer.getvalue()).decode()}


@pytest.fixture
def endpoint():
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            calls.append(('GET',self.path,None))
            if self.path=='/capabilities':value={'models':[{'id':'test-model','capabilities':['video_caption'],'ready':True}]}
            elif self.path=='/models':value={'data':[{'id':'test-model'}]}
            else:value={'status':'completed','result':{'captions':[{'text':'Visible scene'}]}}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(value).encode())
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(('POST',self.path,payload))
            if self.path=='/jobs':value={'id':'job-1'}
            elif self.path.endswith('/cancel'):value={'status':'cancelled'}
            else:value={'choices':[{'message':{'content':'{"captions":[{"text":"A red image"}]}'}}]}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(value).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield {'base_url':f'http://127.0.0.1:{server.server_port}','protocol':'visionrefine_http','model':'test-model','capabilities':['video_caption'],'timeout':10},calls
    server.shutdown();thread.join();server.server_close()


def test_worker_and_chat_protocol_preserve_timestamps(endpoint):
    provider,calls=endpoint
    payload={'capability':'video_caption','frames':[frame()],'width':40,'height':24,'start':0,'end':1,'labels':[],'instruction':''}
    assert providers.check(provider)['ready']
    assert providers.infer(provider,payload,lambda:None,lambda *args:None)['captions']
    assert next(p for m,path,p in calls if path=='/jobs')['frames'][0]['timestamp']==0
    provider['protocol']='openai_compatible'
    assert providers.check(provider)['reachable']
    assert providers.infer(provider,payload,lambda:None,lambda *args:None)['captions'][0]['text']=='A red image'
    request=calls[-1][2]
    assert 'timestamp 0.000000s' in request['messages'][0]['content'][1]['text']
    assert request['messages'][0]['content'][2]['image_url']['url'].startswith('data:image/jpeg;base64,')


def test_no_key_leaks_and_no_fallback(endpoint,monkeypatch):
    provider,calls=endpoint
    provider['api_key_env']='NONEXISTENT_TEST_AI_KEY'
    monkeypatch.delenv('NONEXISTENT_TEST_AI_KEY',raising=False)
    with pytest.raises(ValueError,match='environment variable'):
        providers.check(provider)
    assert not calls


def test_worker_transport_cleanup_auth_and_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(ai_worker,'ROOT',tmp_path/'worker')
    monkeypatch.delenv('VISIONREFINE_AI_TOKEN',raising=False)
    def infer(jid,payload,checkpoint,progress):
        assert (ai_worker.path(jid).parent/'frames/000000.jpg').is_file()
        return {'captions':[{'text':'test'}]}
    monkeypatch.setattr(ai_worker,'run_inference',infer)
    body={'model':'Qwen/Qwen3-VL-2B-Instruct','capability':'video_caption','frames':[frame()],'width':40,'height':24}
    with TestClient(ai_worker.app) as client:
        assert client.post('/jobs',json={**body,'width':999999}).status_code==400
        response=client.post('/jobs',json=body)
        assert response.status_code==202,response.text
        jid=response.json()['id']
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            job=client.get('/jobs/'+jid).json()
            if job['status']=='completed':break
            time.sleep(.01)
        assert job['result']['captions'][0]['text']=='test'
        deadline=time.monotonic()+3
        while (ai_worker.path(jid).parent/'frames').exists() and time.monotonic()<deadline:time.sleep(.01)
        assert not (ai_worker.path(jid).parent/'frames').exists()
        monkeypatch.setenv('VISIONREFINE_AI_TOKEN','example-secret')
        assert client.get('/capabilities').status_code==401
        assert client.get('/capabilities',headers={'Authorization':'Bearer example-secret'}).status_code==200


def test_geometry_roundtrip_preserves_holes_and_scaling():
    import numpy as np
    mask=np.zeros((24,40),dtype=np.uint8);mask[2:20,3:30]=1;mask[6:10,8:12]=0
    geometry=ai_worker.output_geometry(mask,80,48)
    restored=np.asarray(ai_worker.geometry_mask(geometry,80,48))
    assert restored[5,7]==1 and restored[14,18]==0
    assert restored.sum()==mask.sum()*4
