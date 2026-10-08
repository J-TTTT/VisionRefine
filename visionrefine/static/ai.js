/* Shared review panel: no model result is silently committed as human-reviewed. */
window.VisionRefineAI = (() => {
  const names={image_detection:'图片目标检测',image_segmentation:'图片交互分割',video_segmentation:'视频轮廓传播',video_caption:'视频描述',video_events:'动作 / 事件'};
  const statuses={queued:'排队中',running:'处理中',cancelling:'取消中',cancelled:'已取消',completed:'已完成',failed:'失败',interrupted:'已中断'};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let dialog,ctx,config,bindings,timer,job,picture,preview=null,points=[],pointLabels=[],box=null,drag=null,serial=0,pictureToken=0,selection=new Set(),busy=false;
  const el=id=>dialog.querySelector('#'+id);
  const base=()=>`/api/ai/projects/${ctx.pid}`;
  async function api(path,body,method='POST'){const response=await fetch(path,body?{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
  function note(text,error=false){el('aiStatus').textContent=text;el('aiStatus').classList.toggle('error',error);}
  async function attempt(action){const turn=serial;try{await action();}catch(e){if(turn===serial)note(e.message,true);}}
  function mount(){
    if(dialog)return;
    dialog=document.createElement('dialog');dialog.className='ai-dialog';dialog.setAttribute('aria-labelledby','aiTitle');
    dialog.innerHTML=`<div class="ai-head"><div><h2 id="aiTitle">AI 辅助标注</h2><a href="/ai" target="_blank" rel="noopener">模型服务设置 ↗</a></div><button type="button" id="aiClose" aria-label="关闭 AI 面板">关闭</button></div>
      <div class="ai-layout"><form id="aiForm" class="ai-form"><label>生成内容<select id="aiCapability"></select></label><label>本项目使用的模型<select id="aiProvider"></select></label><p id="aiModelHint" class="ai-note"></p><div class="ai-row" id="aiRange"><label>起始帧（从 1 开始）<input id="aiStart" type="number" min="1" step="1"></label><label>结束帧<input id="aiEnd" type="number" min="1" step="1"></label></div><div id="aiTrackFields"><label>传播对象<select id="aiTrack"></select></label><label>起始关键帧<input id="aiSeed" type="number" min="1" step="1"></label><p class="ai-note">先保存该帧轮廓。传播使用区间内的人工关键帧，保留人工轮廓、已审核帧和遮挡／离开状态。修改轮廓后可重新提交此区间。</p></div><div id="aiImageFields"><label>对象类别<select id="aiLabel"></select></label><label>交互提示<select id="aiPromptMode"><option value="positive">前景点（点击对象）</option><option value="negative">背景点（点击排除区域）</option><option value="box">框选对象（拖动）</option></select></label><div class="ai-actions"><button type="button" id="aiClearPrompt">清除提示</button><button type="button" id="aiUseShape">使用当前选中轮廓</button></div><p id="aiPromptHint" class="ai-note">在右侧画面点击或框选。坐标自动换算为原始像素。</p></div><label id="aiSampleField">采样画面数量<input id="aiSamples" type="number" min="2" max="32" value="12"></label><label id="aiThresholdField">检测置信度<input id="aiThreshold" type="number" min="0.01" max="0.99" step="0.01" value="0.2"></label><label id="aiInstructionField">补充要求（可选）<textarea id="aiInstruction" rows="2" placeholder="例如：重点描述手部动作，不推测人物意图。"></textarea></label><button class="ai-primary" id="aiGenerate" type="submit">生成待审核建议</button><p class="ai-note">视频描述和事件只依据采样画面；快速动作可能漏检。采纳后仍需人工检查。</p></form>
      <section><div class="ai-stage"><canvas id="aiCanvas" width="720" height="405" aria-label="AI 提示与结果预览"></canvas></div><p id="aiPreviewLabel" class="ai-note">原始画面 · 建议轮廓可点击预览</p><div class="ai-status" id="aiStatus" role="status">选择模型并生成建议。</div><div class="ai-actions"><label><input type="checkbox" id="aiSelectAll">选择全部待审核建议</label><div><button id="aiReject">拒绝所选</button> <button id="aiAccept" class="ai-primary">采纳到草稿</button></div></div><div class="ai-row" id="aiReviewRange" hidden><label>筛选起始帧<input id="aiFilterStart" type="number" min="1"></label><label>筛选结束帧<input id="aiFilterEnd" type="number" min="1"></label></div><div id="aiItems" class="ai-cards"><div class="ai-empty">建议会显示在这里；现有标注会保留。</div></div><button id="aiMore" hidden>显示更多建议</button></section></div><section class="ai-jobs"><h3>此素材的 AI 任务</h3><div id="aiJobs"></div></section>`;
    document.body.append(dialog);
    el('aiClose').onclick=()=>dialog.close();dialog.addEventListener('close',()=>{clearTimeout(timer);serial++;});
    el('aiCapability').onchange=()=>{job=null;renderItems();fields();};
    el('aiProvider').onchange=()=>{el('aiModelHint').textContent='下次提交任务时保存这个项目的模型选择。'};
    el('aiTrack').onchange=()=>{const t=ctx.tracks.find(t=>t.id===el('aiTrack').value);if(t){el('aiStart').value=t.start_frame+1;el('aiEnd').value=t.end_frame+1;const seed=t.keyframes.find(k=>k.visibility==='visible'&&k.frame_index===ctx.frame)||t.keyframes.find(k=>k.visibility==='visible');el('aiSeed').value=(seed?.frame_index??ctx.frame)+1;}};
    el('aiClearPrompt').onclick=()=>{points=[];pointLabels=[];box=null;ctx.seedGeometry=null;preview=null;draw();promptHint();};
    el('aiUseShape').onclick=()=>{ctx.seedGeometry=structuredClone(ctx.selectedGeometry);preview=ctx.seedGeometry;draw();promptHint();};
    el('aiForm').onsubmit=e=>{e.preventDefault();attempt(generate);};
    el('aiAccept').onclick=()=>attempt(()=>review('accept'));el('aiReject').onclick=()=>attempt(()=>review('reject'));
    el('aiSelectAll').onchange=()=>{selection=new Set(el('aiSelectAll').checked?filtered().filter(i=>i.status==='pending').map(i=>i.id):[]);renderItems();};
    for(const id of ['aiFilterStart','aiFilterEnd'])el(id).oninput=()=>{selection.clear();renderItems();};
    el('aiMore').onclick=()=>{limit+=100;renderItems();};
    el('aiItems').onchange=e=>{const id=e.target.dataset.item;if(!id)return;if(e.target.checked)selection.add(id);else selection.delete(id);updateActions();};
    el('aiItems').onclick=e=>{const id=e.target.closest('[data-preview]')?.dataset.preview;if(id)attempt(()=>showItem(job.items.find(i=>i.id===id)));};
    el('aiJobs').onclick=e=>{const b=e.target.closest('[data-job]');if(!b)return;const turn=serial;attempt(async()=>{if(b.dataset.action==='open'){const value=await api(base()+'/jobs/'+b.dataset.job);if(turn!==serial)return;job=value;selection.clear();limit=100;renderItems();note(job.error||job.progress.message,!!job.error);}else{await api(base()+'/jobs/'+b.dataset.job+'/'+b.dataset.action,{});if(turn===serial)await poll();}});};
    const canvas=el('aiCanvas');
    const position=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(ctx.width,(e.clientX-r.left)/r.width*ctx.width)),Math.max(0,Math.min(ctx.height,(e.clientY-r.top)/r.height*ctx.height))];};
    canvas.onpointerdown=e=>{if(el('aiCapability').value!=='image_segmentation'||busy)return;preview=null;const p=position(e);if(el('aiPromptMode').value==='box'){drag=p;box=null;ctx.seedGeometry=null;canvas.setPointerCapture(e.pointerId);}else if(points.length<64){points.push(p);pointLabels.push(el('aiPromptMode').value==='negative'?0:1);}draw();promptHint();};
    canvas.onpointermove=e=>{if(drag){const p=position(e);box=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];draw();}};
    canvas.onpointerup=()=>{drag=null;if(box&&(box[2]-box[0]<1||box[3]-box[1]<1))box=null;promptHint();};
    canvas.onpointercancel=()=>{drag=null;};
  }
  function promptHint(){el('aiPromptHint').textContent=ctx.seedGeometry?`已使用选中轮廓 · ${points.length} 个修正点。`:`${points.length} 个点${box?' · 已框选对象':''}。在画面上继续添加前景／背景提示。`;}
  function fields(){
    const cap=el('aiCapability').value, video=cap.startsWith('video_'), propagation=cap==='video_segmentation', segment=cap==='image_segmentation';
    el('aiRange').hidden=!video;el('aiTrackFields').hidden=!propagation;el('aiImageFields').hidden=!segment;
    el('aiSampleField').hidden=!video||propagation;el('aiThresholdField').hidden=cap!=='image_detection';el('aiInstructionField').hidden=propagation||segment;
    el('aiProvider').innerHTML='<option value="">使用全局默认</option>'+config.providers.filter(p=>p.capabilities.includes(cap)).map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('');
    el('aiProvider').value=bindings.providers[cap]||'';
    el('aiModelHint').textContent='全局默认：'+(config.providers.find(p=>p.id===config.defaults[cap])?.name||'未配置');
    el('aiUseShape').disabled=!ctx.selectedGeometry;
    if(propagation)el('aiTrack').onchange();
  }
  async function loadPicture(url){const turn=serial,token=++pictureToken;const img=new Image();await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('预览画面加载失败'));img.src=url;});if(turn!==serial||token!==pictureToken)return false;picture=img;draw();return true;}
  function draw(){
    const canvas=el('aiCanvas');canvas.style.maxWidth=(390*ctx.width/ctx.height)+'px';canvas.width=Math.min(1000,ctx.width);canvas.height=Math.round(canvas.width*ctx.height/ctx.width);const c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);if(picture)c.drawImage(picture,0,0,canvas.width,canvas.height);c.save();c.scale(canvas.width/ctx.width,canvas.height/ctx.height);c.lineWidth=Math.max(2,ctx.width/350);c.strokeStyle='#71efbd';c.fillStyle='#4de6a655';
    if(preview?.kind==='bbox'){const b=preview.bbox;c.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
    else if(preview?.polygons){for(const ring of preview.polygons){c.beginPath();ring.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.closePath();c.fill();c.stroke();}}
    else if(preview?.mask){for(const tile of preview.mask.tiles){const temp=document.createElement('canvas');temp.width=temp.height=128;const tc=temp.getContext('2d'),data=tc.createImageData(128,128);let offset=0;tile.counts.forEach((n,i)=>{if(i%2)for(let p=offset;p<offset+n;p++){data.data[p*4]=62;data.data[p*4+1]=233;data.data[p*4+2]=152;data.data[p*4+3]=145;}offset+=n;});tc.putImageData(data,0,0);c.drawImage(temp,tile.x,tile.y);}}
    if(box){c.strokeStyle='#f8d56b';c.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
    points.forEach((p,i)=>{c.fillStyle=pointLabels[i]?'#5befaf':'#ff6a73';c.beginPath();c.arc(p[0],p[1],Math.max(3,ctx.width/120),0,Math.PI*2);c.fill();});c.restore();
  }
  let limit=100;
  function filtered(){return(job?.items||[]).filter(i=>i.kind!=='keyframes'||(i.value.frame_index+1>=Number(el('aiFilterStart').value||1)&&i.value.frame_index+1<=Number(el('aiFilterEnd').value||ctx.frameCount)));}
  function updateActions(){el('aiAccept').disabled=busy||!selection.size;el('aiReject').disabled=busy||!selection.size;const pending=filtered().filter(i=>i.status==='pending');el('aiSelectAll').checked=!!pending.length&&pending.every(i=>selection.has(i.id));}
  function renderItems(){
    el('aiReviewRange').hidden=job?.request.capability!=='video_segmentation';const rows=filtered();el('aiMore').hidden=rows.length<=limit;
    el('aiItems').innerHTML=rows.length?rows.slice(0,limit).map(i=>{const v=i.value;const title=i.kind==='keyframes'?`帧 ${v.frame_index+1} · ${v.visibility==='visible'?'可见轮廓':'未检测到可见部分（请核对状态）'}`:i.kind==='objects'?v.label:i.kind==='captions'?'描述建议':`事件 · ${v.start.toFixed(3)}s${v.end!=null?' – '+v.end.toFixed(3)+'s':''}`;return`<article class="ai-card"><header><label><input type="checkbox" data-item="${i.id}" ${selection.has(i.id)?'checked':''} ${i.status!=='pending'?'disabled':''}>${esc(title)}</label>${['objects','keyframes'].includes(i.kind)?`<button data-preview="${i.id}">预览</button>`:''}</header>${v.text?`<p>${esc(v.text)}</p>`:''}<small>${{pending:'待审核',accepted:'已采纳为草稿',rejected:'已拒绝'}[i.status]}${v.confidence!=null?' · 置信度 '+(v.confidence*100).toFixed(1)+'%':''}</small></article>`;}).join(''):`<div class="ai-empty">${job?.status==='completed'?'当前范围没有建议。可调整范围或提示后重新生成。':'完成任务后，在下方选择任务查看建议。'}</div>`;updateActions();
  }
  async function showItem(item){preview=item.value.geometry||item.value;if(item.kind==='keyframes'){points=[];pointLabels=[];box=null;if(!await loadPicture(`/api/video/projects/${ctx.pid}/videos/${ctx.videoId}/frames/${item.value.frame_index}`))return;el('aiPreviewLabel').textContent=`建议预览 · 帧 ${item.value.frame_index+1} · ${job.provider.model}`;}else{draw();el('aiPreviewLabel').textContent='建议预览 · '+job.provider.model;}}
  async function generate(){
    if(busy)return;
    const turn=serial,context=ctx,projectBase=base(),cap=el('aiCapability').value,selected=el('aiProvider').value;
    const request={capability:cap,instruction:el('aiInstruction').value};
    if(context.videoId){Object.assign(request,{video_id:context.videoId,start_frame:Number(el('aiStart').value)-1,end_frame:Number(el('aiEnd').value)-1,sample_count:Number(el('aiSamples').value)});if(cap==='video_segmentation')Object.assign(request,{track_id:el('aiTrack').value,seed_frame:Number(el('aiSeed').value)-1});}
    else{request.image=context.image;request.threshold=Number(el('aiThreshold').value);if(cap==='image_segmentation')Object.assign(request,{label:el('aiLabel').value,prompt:structuredClone({points,point_labels:pointLabels,box,geometry:context.seedGeometry||null})});}
    const submittedBindings=structuredClone(bindings);
    if(selected)submittedBindings.providers[cap]=selected;else delete submittedBindings.providers[cap];
    if(context.beforeApply&&!await context.beforeApply())throw new Error('请先保存当前标注，再生成 AI 建议。');
    if(turn!==serial)return;
    busy=true;el('aiGenerate').disabled=true;note('正在提交任务…');
    try{
      const savedBindings=await api(projectBase+'/bindings',submittedBindings,'PUT');
      const created=await api(projectBase+'/jobs',request);
      if(turn!==serial)return;
      bindings=savedBindings;job=created;selection.clear();renderItems();note('任务已提交，可以关闭面板继续标注。');await poll();
    }finally{if(turn===serial){busy=false;el('aiGenerate').disabled=false;updateActions();}}
  }
  async function review(action){
    if(busy||!job||!selection.size)return;
    const turn=serial,context=ctx,url=base()+'/jobs/'+job.id+'/review',ids=[...selection];
    if(action==='accept'&&context.beforeApply&&!await context.beforeApply())throw new Error('当前有未保存修改，请先保存。');
    if(turn!==serial)return;
    busy=true;updateActions();try{const saved=await api(url,{action,item_ids:ids});if(action==='accept')await context.onApplied?.();if(turn!==serial)return;job=saved;selection.clear();renderItems();note(action==='accept'?'已采纳到待审核草稿。请检查轮廓、时间与描述，再进行人工确认。':'所选建议已拒绝。');await poll();}finally{if(turn===serial){busy=false;updateActions();}}
  }
  async function poll(){
    clearTimeout(timer);if(!dialog.open)return;const turn=serial;const jobs=await api(base()+'/jobs');if(turn!==serial)return;
    const relevant=jobs.filter(j=>ctx.videoId?j.request.video_id===ctx.videoId:j.request.image===ctx.image);
    el('aiJobs').innerHTML=relevant.map(j=>`<div class="ai-job"><span>${esc(names[j.request.capability])} · ${esc(statuses[j.status])}<br>${esc(j.error||j.progress.message)}<br>${esc(j.provider.name)} · ${esc(j.created_at.slice(0,19).replace('T',' '))}</span><button data-job="${j.id}" data-action="open">查看</button>${['queued','running','cancelling'].includes(j.status)?`<button data-job="${j.id}" data-action="cancel">取消</button>`:['failed','cancelled','interrupted'].includes(j.status)?`<button data-job="${j.id}" data-action="retry">重试</button>`:''}</div>`).join('')||'<p class="ai-note">还没有 AI 任务。</p>';
    if(job&&job.status!=='completed'){const latest=relevant.find(j=>j.id===job.id);if(latest){if(latest.status==='completed'){const value=await api(base()+'/jobs/'+job.id);if(turn!==serial)return;job=value;renderItems();note(job.progress.message);}else{job={...job,...latest};note(job.error||job.progress.message,!!job.error);}}}
    timer=setTimeout(()=>attempt(poll),1800);
  }
  async function open(context){
    mount();clearTimeout(timer);ctx={...context};serial++;job=null;picture=null;preview=null;points=[];pointLabels=[];box=null;selection.clear();busy=false;limit=100;
    const turn=serial;
    dialog.showModal();note('正在读取模型配置…');
    await attempt(async()=>{
      const settings=await Promise.all([api('/api/ai/config'),api(base()+'/bindings')]);
      if(turn!==serial)return;
      [config,bindings]=settings;
      const caps=ctx.videoId?['video_caption','video_events','video_segmentation']:[ctx.task==='instance_segmentation'?'image_segmentation':'image_detection'];
      el('aiCapability').innerHTML=caps.map(cap=>`<option value="${cap}">${names[cap]}</option>`).join('');
      if(ctx.initialCapability)el('aiCapability').value=ctx.initialCapability;
      el('aiTrack').innerHTML=(ctx.tracks||[]).map(t=>`<option value="${t.id}">${esc(t.name||t.id)}</option>`).join('');if(ctx.trackId)el('aiTrack').value=ctx.trackId;
      el('aiLabel').innerHTML=(ctx.labels||[]).map(l=>`<option value="${esc(l)}">${esc(l)}</option>`).join('');
      if(ctx.labels?.includes(ctx.initialLabel))el('aiLabel').value=ctx.initialLabel;
      el('aiStart').value=1;el('aiEnd').value=ctx.frameCount||1;el('aiSeed').value=(ctx.frame||0)+1;el('aiFilterStart').value=1;el('aiFilterEnd').value=ctx.frameCount||1;
      el('aiInstruction').value='';el('aiGenerate').disabled=false;
      fields();renderItems();if(!await loadPicture(ctx.imageUrl))return;note('生成结果将先进入待审核区；采纳不会自动确认审核完成。');await poll();
    });
  }
  return {open};
})();
