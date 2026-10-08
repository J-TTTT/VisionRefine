"""Actual browser actions against durable jobs, with a deterministic model adapter."""
import os
import time

import pytest
from PIL import Image

from test_video_segmentation_browser import video_workspace
from visionrefine.core.ai import providers

pytestmark=pytest.mark.skipif(os.environ.get('VISIONREFINE_BROWSER_TESTS')!='1',reason='Opt-in browser acceptance')


def test_video_ai_review_and_settings(video_workspace,monkeypatch,tmp_path):
    page,pw,base,project_url,vid,store=video_workspace
    inputs=[]
    def infer(provider,payload,checkpoint,progress):
        inputs.append(payload)
        if payload['capability']=='image_segmentation':
            return {'objects':[{'kind':'polygon','label':'cell','polygons':[[[30,40],[120,40],[120,100],[30,100]]]}]}
        return {'captions':[{'text':'一个橙色方块从左向右移动。'}]}
    monkeypatch.setattr(providers,'infer',infer)
    page.locator('#videoAI').click()
    page.locator('#aiCapability').select_option('video_caption')
    page.locator('#aiGenerate').click()
    pw.expect(page.locator('#aiItems')).to_contain_text('一个橙色方块',timeout=10000)
    assert page.request.get(project_url+'/videos/'+vid+'/annotations').json()['captions']==[]
    page.locator('#aiSelectAll').check()
    page.locator('#aiAccept').click()
    pw.expect(page.locator('#aiStatus')).to_contain_text('已采纳',timeout=10000)
    doc=page.request.get(project_url+'/videos/'+vid+'/annotations').json()
    assert doc['captions'][0]['reviewed'] is False
    page.locator('#aiClose').click()
    page.reload()
    page.locator('#tabCaptions').click()
    pw.expect(page.locator('#recordsBody')).to_contain_text('一个橙色方块')
    page.locator('#videoAI').click()
    pw.expect(page.locator('#aiJobs')).to_contain_text('已完成')
    page.locator('#aiJobs [data-action=open]').click()
    pw.expect(page.locator('#aiItems')).to_contain_text('已采纳为草稿')
    # Closing/reopening while settings save is in flight must preserve the submitted range.
    page.locator('#aiEnd').fill('12')
    page.evaluate("""() => {
      const original=window.fetch.bind(window); window.realFetch=original;
      window.fetch=(url,options)=>String(url).endsWith('/bindings')&&options?.method==='PUT'
        ? new Promise(resolve=>{window.resumeBinding=()=>{window.fetch=original;resolve(original(url,options));};})
        : original(url,options);
    }""")
    page.locator('#aiGenerate').click()
    page.wait_for_function('Boolean(window.resumeBinding)')
    page.locator('#aiClose').click()
    page.locator('#videoAI').click()
    pw.expect(page.locator('#aiEnd')).to_have_value('24')
    page.evaluate('window.resumeBinding()')
    deadline=time.monotonic()+5
    while len(inputs)<2 and time.monotonic()<deadline:page.wait_for_timeout(50)
    assert inputs[-1]['end_frame']==11
    page.locator('#aiClose').click()
    page.goto(base+'/ai')
    pw.expect(page.locator('#aiProviders')).to_contain_text('Qwen3-VL 2B')
    # LAN HTTP URLs do not expose crypto.randomUUID (unlike localhost).
    page.evaluate('Object.defineProperty(crypto, "randomUUID", {value:undefined})')
    page.locator('#aiAddProvider').click()
    custom=page.locator('[data-provider]').last
    custom.locator('[name=name]').fill('自定义视觉模型')
    custom.locator('[name=model]').fill('my-model')
    page.get_by_role('button',name='保存模型配置').click()
    pw.expect(page.locator('#aiConfigStatus')).to_contain_text('已保存')

    source=tmp_path/'images';source.mkdir()
    Image.new('RGB',(640,480),'gray').save(source/'sample.png')
    project=page.request.post(base+'/api/projects',data={'name':'AI 图片验收','task':'instance_segmentation',
        'dataset_path':str(source),'labels':['cell'],'model_max_side':1536}).json()
    pid=project['id'];assert page.request.post(base+f'/api/projects/{pid}/analyze').ok
    page.goto(base+'/')
    page.get_by_role('button',name='AI 图片验收 实例分割').click()
    page.locator('#openWorkspace').click()
    page.wait_for_function("editor.image==='sample.png' && !editor.loading")
    page.locator('#imageAI').click()
    pw.expect(page.locator('#aiImageFields')).to_be_visible()
    canvas=page.locator('#aiCanvas');canvas.scroll_into_view_if_needed();bounds=canvas.bounding_box()
    page.mouse.click(bounds['x']+bounds['width']/4,bounds['y']+bounds['height']/4)
    page.locator('#aiGenerate').click()
    pw.expect(page.locator('#aiItems')).to_contain_text('cell',timeout=10000)
    assert inputs[-1]['prompt']['point_labels']==[1]
    assert inputs[-1]['prompt']['points'][0]==pytest.approx([160,120],abs=2)
    page.locator('[data-preview]').click()
    # Refining a preview must retain the original foreground prompt.
    page.locator('#aiPromptMode').select_option('negative')
    canvas.scroll_into_view_if_needed();bounds=canvas.bounding_box()
    page.mouse.click(bounds['x']+bounds['width']/2,bounds['y']+bounds['height']/2)
    page.locator('#aiGenerate').click()
    pw.expect(page.locator('#aiJobs .ai-job')).to_have_count(2,timeout=10000)
    pw.expect(page.locator('#aiItems')).to_contain_text('cell',timeout=10000)
    assert inputs[-1]['prompt']['point_labels']==[1,0]
    page.locator('#aiSelectAll').check();page.locator('#aiAccept').click()
    pw.expect(page.locator('#aiStatus')).to_contain_text('已采纳')
    page.locator('#aiClose').click()
    assert page.evaluate('editor.objects.length')==1
    page.locator('#saveAnnotations').click()
    pw.expect(page.locator('#workspaceStatus')).to_contain_text('已保存 1 个实例')
    saved=page.request.get(base+f'/api/projects/{pid}/annotations?image=sample.png').json()
    assert saved['status']=='human_reviewed'
    assert saved['objects'][0]['provenance']['ai']['model']=='facebook/sam2.1-hiera-tiny'
