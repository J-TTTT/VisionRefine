(() => {
  const names = {image_detection:'图片目标检测', image_segmentation:'图片交互分割', video_segmentation:'视频轮廓传播', video_caption:'视频描述', video_events:'动作 / 事件'};
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let config;
  const providerId = () => 'custom-' + (window.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`);
  const status = (value, error=false) => { const node = document.getElementById('aiConfigStatus'); node.textContent=value; node.classList.toggle('error',error); };
  async function api(path, body, method='POST') { const response=await fetch('/api/ai'+path, body ? {method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {}); const data=await response.json(); if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail)); return data; }
  function read() {
    for (const node of document.querySelectorAll('[data-provider]')) {
      const provider=config.providers[Number(node.dataset.provider)];
      for(const field of ['name','protocol','base_url','model','api_key_env','timeout']) provider[field]=node.querySelector(`[name=${field}]`).value;
      provider.api_key_env ||= null; provider.timeout=Number(provider.timeout);
      provider.capabilities=[...node.querySelectorAll('[data-cap]:checked')].map(input=>input.dataset.cap);
    }
    for(const input of document.querySelectorAll('[data-default]')) config.defaults[input.dataset.default]=input.value;
  }
  function render() {
    document.getElementById('aiDefaults').innerHTML=Object.entries(names).map(([cap,name])=>`<label>${name}<select data-default="${cap}">${config.providers.filter(p=>p.capabilities.includes(cap)).map(p=>`<option value="${p.id}" ${config.defaults[cap]===p.id?'selected':''}>${esc(p.name)}</option>`).join('')}</select></label>`).join('');
    document.getElementById('aiProviders').innerHTML=config.providers.map((p,i)=>`<section data-provider="${i}"><div class="ai-head"><h3>${esc(p.name)}</h3><button type="button" data-test="${i}">测试连接</button></div><div class="ai-profile"><label>显示名称<input name="name" required value="${esc(p.name)}"></label><label>服务协议<select name="protocol"><option value="visionrefine_http" ${p.protocol==='visionrefine_http'?'selected':''}>VisionRefine 模型服务（含分割）</option><option value="openai_compatible" ${p.protocol==='openai_compatible'?'selected':''}>OpenAI 兼容视觉接口</option></select></label><label class="ai-wide">服务地址<input type="url" required name="base_url" value="${esc(p.base_url)}" placeholder="http://127.0.0.1:8030 或 http://server:8000/v1"></label><label class="ai-wide">模型名称<input required name="model" value="${esc(p.model)}"></label><label>密钥环境变量（可选）<input name="api_key_env" value="${esc(p.api_key_env)}" placeholder="MY_MODEL_API_KEY"></label><label>任务超时（秒）<input name="timeout" type="number" min="10" max="3600" value="${p.timeout}"></label><div class="ai-wide ai-checks">${Object.entries(names).map(([cap,name])=>`<label><input type="checkbox" data-cap="${cap}" ${p.capabilities.includes(cap)?'checked':''}>${name}</label>`).join('')}</div><p class="ai-note ai-wide" data-test-result></p></div></section>`).join('');
  }
  document.getElementById('aiAddProvider').onclick=()=>{read();config.providers.push({id:providerId(), name:'自定义模型',protocol:'openai_compatible',base_url:'http://127.0.0.1:8000/v1',model:'',capabilities:['video_caption','video_events'],api_key_env:null,timeout:600});render();};
  document.getElementById('aiProviders').addEventListener('change',e=>{if(e.target.matches('[data-cap]')){read();render();}});
  document.getElementById('aiProviders').addEventListener('click',async e=>{const b=e.target.closest('[data-test]');if(!b)return;read();const target=b.closest('section').querySelector('[data-test-result]');b.disabled=true;target.textContent='正在测试…';try{const r=await api('/test',config.providers[Number(b.dataset.test)]);target.textContent=r.ready===false?'服务已连接；模型权重尚未就绪。'+(r.detail||''):'连接正常 · '+r.model+' · '+(r.detail||'模型可用');}catch(error){target.textContent='连接失败：'+error.message;}finally{b.disabled=false;}});
  document.getElementById('aiConfigForm').onsubmit=async e=>{e.preventDefault();read();const b=e.submitter;b.disabled=true;try{config=await api('/config',config,'PUT');render();status('模型配置已保存。正在运行的任务继续使用提交时的配置。');}catch(error){status(error.message,true);}finally{b.disabled=false;}};
  api('/config').then(value=>{config=value;render();status('填写服务地址后，可先测试连接，再保存配置。');}).catch(e=>status(e.message,true));
})();
