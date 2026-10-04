(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API = '/api/video';
  let segmentation = null;
  const hasUnsaved = () => state.dirty || Boolean(segmentation?.isDirty());
  const isSaving = () => state.saving || Boolean(segmentation?.isSaving());
  const palette = ['#146c4b', '#476baf', '#ae6542', '#8a5a9d', '#327d8d', '#9c7837'];
  const state = {projects: [], project: null, videos: [], labels: [], jobs: [], video: null, timestamps: [], annotations: null, tab: 'events', selected: null, frame: 0, time: 0, dirty: false, edits: 0, saving: false, projectToken: 0, videoToken: 0, poll: null, sample: [], sampleSelected: new Set(), labelDraft: [], importedLabels: [], importBundle: null};
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const uid = () => window.crypto?.randomUUID?.() || `v-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  const clone = value => JSON.parse(JSON.stringify(value));
  const round = value => Math.round(value * 1000000) / 1000000;
  const duration = () => Number(state.video?.duration || 0);
  const color = value => /^#[\da-f]{6}$/i.test(value || '') ? value : palette[0];
  const names = value => [...new Set(String(value).split(/[,，\n\r]+/).map(x => x.trim()).filter(Boolean))];
  const projectPath = () => `${API}/projects/${encodeURIComponent(state.project.id)}`;
  const videoPath = () => `${projectPath()}/videos/${encodeURIComponent(state.video.id)}`;
  const labelName = id => state.labels.find(label => label.id === id)?.name || '';
  const selectedRecord = () => state.annotations?.[state.selected?.type]?.find(item => item.id === state.selected.id);
  const isBusyJob = job => ['queued', 'running', 'pending', 'cancelling'].includes(job.status);
  const sameVideo = (projectId, videoId) => state.project?.id === projectId && state.video?.id === videoId;
  function timeText(value) { const n = Math.max(0, Number(value) || 0); return `${String(Math.floor(n / 60)).padStart(2, '0')}:${(n % 60).toFixed(3).padStart(6, '0')}`; }
  function intervalText(item) { return item.start == null ? '整段视频' : item.kind === 'point' ? timeText(item.start) : `${timeText(item.start)} – ${timeText(item.end)}`; }
  function notify(message, error = false) { $('notice').hidden = !message; $('notice').classList.toggle('error', error); $('notice').textContent = message; }
  async function api(path, options = {}) {
    const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
    const body = await response.text();
    let data; try { data = body ? JSON.parse(body) : {}; } catch { throw new Error('服务器返回了无法解析的数据。请检查服务日志。'); }
    if (!response.ok) { const detail = data.detail || data.error || data.message; throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `请求失败（${response.status}）`); }
    return data;
  }
  const post = (path, body) => api(path, {method: 'POST', body: JSON.stringify(body)});
  async function run(action) { try { await action(); } catch (error) { notify(error.message, true); } }
  function updateDirty() { $('saveState').textContent = isSaving() ? '正在保存 / 处理…' : hasUnsaved() ? '有未保存的修改' : '已保存'; $('saveState').classList.toggle('dirty', hasUnsaved()); $('save').disabled = !state.video || !hasUnsaved() || isSaving(); }
  function changed(type, item = null, reviewEdit = false) { state.dirty = true; state.edits++; if (type && !reviewEdit) { state.annotations.review[type] = 'unreviewed'; if (item) item.reviewed = false; } updateDirty(); renderRecords(); renderTimeline(); }
  function setLocation() { const url = new URL(window.location.href); ['project', 'video', 'frame'].forEach(key => url.searchParams.delete(key)); if (state.project) url.searchParams.set('project', state.project.id); if (state.video) url.searchParams.set('video', state.video.id); history.replaceState(null, '', url); }
  async function mayLeave() {
    if (isSaving()) { notify('正在保存或处理分割，请等待完成。'); return false; }
    if (!hasUnsaved()) return true;
    const decision = await new Promise(resolve => {
      const dialog = $('unsavedDialog');
      const click = event => { const choice = event.target.closest('[data-decision]'); if (choice) finish(choice.dataset.decision); };
      const cancel = event => { event.preventDefault(); finish('cancel'); };
      const finish = value => { dialog.removeEventListener('click', click); dialog.removeEventListener('cancel', cancel); dialog.close(); resolve(value); };
      dialog.addEventListener('click', click); dialog.addEventListener('cancel', cancel); dialog.showModal();
    });
    if (decision === 'cancel') return false;
    if (decision === 'save') return saveAll();
    return true;
  }
  function normalizeAnnotations(data) { return { ...data, revision: data.revision ?? 0, events: data.events || [], captions: data.captions || [], review: {events: 'unreviewed', captions: 'unreviewed', ...(data.review || {})} }; }
  function validateAnnotations() {
    const d = duration();
    for (const event of state.annotations.events) {
      if (!event.label_id && !event.text.trim()) throw new Error('每条事件至少需要一个标签或一段自由描述。');
      if (event.label_id && !state.labels.some(label => label.id === event.label_id)) throw new Error('事件引用了不存在的标签，请重新选择。');
      if (!Number.isFinite(event.start) || event.start < 0 || event.start > d) throw new Error('事件开始时间需要在视频范围内。');
      if (event.kind === 'interval' && (!Number.isFinite(event.end) || event.end <= event.start || event.end > d)) throw new Error('持续事件的结束时间必须晚于开始时间，且在视频范围内。');
    }
    for (const caption of state.annotations.captions) {
      if (!caption.text.trim()) throw new Error('请填写每条视频描述的文本。');
      if (caption.start != null && (!Number.isFinite(caption.start) || !Number.isFinite(caption.end) || caption.start < 0 || caption.end <= caption.start || caption.end > d)) throw new Error('片段描述需要有效的开始和结束时间。');
    }
  }
  async function saveAnnotations() {
    if (!state.video || state.saving) return false;
    if (!state.dirty) return true;
    try { validateAnnotations(); } catch (error) { notify(error.message, true); return false; }
    const pid = state.project.id, vid = state.video.id, edits = state.edits;
    const snapshot = clone(state.annotations); state.saving = true; updateDirty();
    try {
      const saved = normalizeAnnotations(await api(`${videoPath()}/annotations`, {method: 'PUT', body: JSON.stringify(snapshot)}));
      if (sameVideo(pid, vid)) {
        if (state.edits === edits) { state.annotations = saved; state.dirty = false; renderInspector(); }
        else state.annotations.revision = saved.revision;
        renderRecords(); notify(state.dirty ? '上一版本已保存；保存期间的新修改仍在草稿中。' : '标注已保存。');
      }
      return state.edits === edits;
    } catch (error) { notify(`保存失败：${error.message}\n当前修改仍保留在页面中。若发生版本冲突，请先复制重要文本，再刷新并重新编辑。`, true); return false; }
    finally { state.saving = false; updateDirty(); }
  }
  async function saveAll() {
    if (isSaving()) return false;
    if (segmentation && !await segmentation.save()) return false;
    if (!await saveAnnotations()) return false;
    if (hasUnsaved()) { notify('当前版本已保存；保存期间的新修改仍在草稿中，请再次保存。'); updateDirty(); return false; }
    notify('事件、描述与分割草稿已保存。'); updateDirty(); return true;
  }
  function renderProjects() {
    $('projectSelect').innerHTML = '<option value="">选择视频项目</option>' + state.projects.map(project => `<option value="${esc(project.id)}">${esc(project.name)}</option>`).join('');
    $('projectSelect').value = state.project?.id || '';
  }
  function renderVideos() {
    const query = $('videoSearch').value.toLowerCase().trim();
    const videos = state.videos.filter(video => String(video.name || video.path || video.id).toLowerCase().includes(query));
    $('videoCount').textContent = state.videos.length;
    $('videoList').innerHTML = videos.length ? videos.map(video => `<button class="video-item ${video.id === state.video?.id ? 'active' : ''}" data-video="${esc(video.id)}" ${video.id === state.video?.id ? 'aria-current="true"' : ''}><span class="video-icon">▷</span><div><strong>${esc(video.name || video.id)}</strong><small>${timeText(video.duration)} · ${video.width || '—'} × ${video.height || '—'}</small></div></button>`).join('') : `<p class="side-empty">${query ? '没有匹配的视频。' : state.project ? '视频导入后会出现在这里。请查看后台任务进度。' : '新建项目，导入第一段视频。'}</p>`;
  }
  async function loadProjects() { const result = await api(`${API}/projects`); state.projects = Array.isArray(result) ? result : result.projects || []; renderProjects(); }
  async function loadProject(id, initial = {}) {
    const token = ++state.projectToken; ++state.videoToken; clearTimeout(state.poll); segmentation?.reset();
    $('player').pause(); state.project = null; state.video = null; state.annotations = null; state.dirty = false; state.sample = []; $('workspace').hidden = true; $('welcome').hidden = false; $('save').disabled = true; $('jobsPanel').hidden = true;
    document.querySelectorAll('.project-action').forEach(button => button.disabled = true);
    const [result, objectLabels] = await Promise.all([api(`${API}/projects/${encodeURIComponent(id)}`), api(`${API}/projects/${encodeURIComponent(id)}/object-labels`)]);
    if (token !== state.projectToken) return;
    state.project = result.project; state.videos = result.videos || []; state.labels = result.labels || []; state.jobs = result.jobs || []; segmentation?.setLabels(objectLabels.labels || []);
    $('projectTitle').textContent = state.project.name; $('projectSubtitle').textContent = '标记事件、描述内容，并挑选需要进一步标注的帧。'; $('projectSelect').value = id;
    document.querySelectorAll('.project-action').forEach(button => button.disabled = false);
    renderVideos(); renderJobs(); setLocation();
    const selected = state.videos.find(video => video.id === initial.video) || state.videos[0];
    if (selected) await loadVideo(selected.id, initial.frame);
    else notify('项目已创建。视频正在后台导入，处理完成后会自动显示。');
    schedulePoll();
  }
  async function loadVideo(id, initialFrame) {
    const token = ++state.videoToken, projectId = state.project.id, base = `${projectPath()}/videos/${encodeURIComponent(id)}`;
    const video = state.videos.find(item => item.id === id); if (!video) return;
    $('player').pause(); $('workspace').hidden = true; $('save').disabled = true;
    state.video = null; state.annotations = null; state.sample = []; state.sampleSelected = new Set(); segmentation?.reset();
    const [index, annotations, segmentationDocument] = await Promise.all([api(`${base}/index`), api(`${base}/annotations`), api(`${base}/segmentation`)]);
    if (token !== state.videoToken || state.project?.id !== projectId) return;
    state.video = video; state.timestamps = index.timestamps || []; state.annotations = normalizeAnnotations(annotations); state.dirty = false; state.edits = 0; state.selected = null;
    if (!state.timestamps.length) throw new Error('视频没有可定位的帧索引。请查看导入任务是否成功。');
    segmentation?.setDocument(segmentationDocument);
    const requestedFrame = Number(initialFrame);
    const initialIndex = Math.max(0, Math.min(state.timestamps.length - 1, Number.isInteger(requestedFrame) ? requestedFrame : 0));
    state.pendingFrame = initialIndex;
    $('player').onloadedmetadata = () => {
      if (token !== state.videoToken || state.project?.id !== projectId) return;
      const frame = state.pendingFrame ?? initialIndex; state.pendingFrame = null;
      $('player').currentTime = state.timestamps[frame]; updateTime(state.timestamps[frame]);
    };
    const previewFormat = $('player').canPlayType('video/mp4; codecs="avc1.42E01E"') ? 'mp4' : 'webm';
    $('player').src = `${base}/media?format=${previewFormat}`; $('player').playbackRate = Number($('speed').value); $('exactFrame').hidden = true; $('exactBadge').hidden = true;
    $('videoTitle').textContent = video.name || video.id; $('videoMeta').textContent = `${video.width} × ${video.height} · ${timeText(video.duration)} · ${state.timestamps.length} 帧 · ${video.has_audio ? '含音轨' : '无音轨'} · ${video.split || '未指定划分'}`;
    $('seek').max = duration(); $('sampleStart').value = '0'; $('sampleEnd').value = duration();
    $('welcome').hidden = true; $('workspace').hidden = false;
    renderVideos(); setTab(state.tab); renderSample(); updateDirty(); setLocation();
    setFrame(initialIndex, true);
  }
  function frameAt(time) {
    const timestamps = state.timestamps; let lo = 0, hi = timestamps.length - 1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (timestamps[mid] <= time + 0.000001) lo = mid + 1; else hi = mid - 1; }
    return Math.max(0, Math.min(timestamps.length - 1, hi));
  }
  function updateTime(time) { state.time = Math.min(duration(), Math.max(0, time)); state.frame = frameAt(state.time); $('seek').value = state.time; $('timeReadout').textContent = `${timeText(state.time)} / ${timeText(duration())}`; $('frameReadout').textContent = `帧 ${state.frame + 1} / ${state.timestamps.length}`; segmentation?.onFrame(); }
  function setFrame(index, exact = true) {
    if (!state.video || !state.timestamps.length) return;
    index = Math.max(0, Math.min(state.timestamps.length - 1, index)); if (segmentation && !segmentation.beforeFrame(index)) return; $('player').pause();
    const time = state.timestamps[index]; updateTime(time); if ($('player').readyState < 1) state.pendingFrame = index; else { state.pendingFrame = null; $('player').currentTime = time; }
    if (exact) refreshExactFrame(index);
    else { $('exactFrame').hidden = true; $('exactBadge').hidden = true; }
  }
  function refreshExactFrame(index = state.frame) {
    if (!state.video || !$('player').paused) return;
    const src = `${videoPath()}/frames/${index}`;
    $('exactFrame').dataset.video = state.video.id; $('exactFrame').dataset.frame = String(index);
    if ($('exactFrame').getAttribute('src') === src && $('exactFrame').complete && $('exactFrame').naturalWidth) { $('exactFrame').hidden = false; $('exactBadge').hidden = false; $('exactBadge').textContent = '精确帧预览'; return; }
    $('exactFrame').hidden = true; $('exactBadge').hidden = false; $('exactBadge').textContent = '正在读取精确帧…'; $('exactFrame').src = src;
  }
  function seekTime(time, exact = true) { setFrame(frameAt(time), exact); }
  async function togglePlay() {
    if (!state.video) return;
    const player = $('player');
    if (player.paused && segmentation && !segmentation.beforeContextChange()) return;
    if (!player.paused) { player.pause(); setFrame(frameAt(player.currentTime)); return; }
    const item = selectedRecord(); if ($('loop').checked && item?.start != null && item.end > item.start && (player.currentTime < item.start || player.currentTime >= item.end)) player.currentTime = item.start;
    $('exactFrame').hidden = true; $('exactBadge').hidden = true;
    try { await player.play(); } catch { notify('浏览器无法播放这个预览。仍可用逐帧预览进行标注；请查看视频预览生成状态。', true); }
  }
  function setTab(tab) {
    if (state.tab === 'segmentation' && tab !== state.tab && segmentation && !segmentation.beforeContextChange()) return;
    state.tab = tab;
    document.querySelectorAll('[data-tab]').forEach(button => { const active = button.dataset.tab === tab; button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1; });
    $('recordsBody').hidden = !['events', 'captions'].includes(tab); $('recordTools').hidden = !['events', 'captions'].includes(tab); $('framesPanel').hidden = tab !== 'frames'; segmentation?.setActive(tab === 'segmentation');
    $('addRecord').textContent = tab === 'captions' ? '＋ 添加描述' : '＋ 添加事件';
    if (state.selected && state.selected.type !== tab) state.selected = null;
    renderRecords(); renderInspector(); renderTimeline();
  }
  function renderRecords() {
    if (!state.annotations) return;
    $('eventCount').textContent = state.annotations.events.length; $('captionCount').textContent = state.annotations.captions.length;
    if (!['events', 'captions'].includes(state.tab)) return;
    const reviewed = state.annotations.review[state.tab] === 'reviewed';
    $('taskReview').textContent = reviewed ? '✓ 整项已审核' : '整项待审核'; $('taskReview').classList.toggle('reviewed', reviewed);
    $('reviewTask').textContent = `确认${state.tab === 'events' ? '事件' : '描述'}审核完成`;
    const records = state.annotations[state.tab];
    $('recordsBody').innerHTML = records.length ? records.map((item, index) => `<button class="record ${state.selected?.id === item.id ? 'selected' : ''}" data-record="${esc(item.id)}"><span class="record-number">${String(index + 1).padStart(2, '0')}</span><span class="record-main"><span class="record-title"><span class="review-dot ${item.reviewed ? 'reviewed' : ''}" aria-label="${item.reviewed ? '已审核' : '草稿'}"></span>${esc(state.tab === 'events' ? labelName(item.label_id) || item.text || '未填写事件' : item.text || '未填写描述')}</span><span class="record-text">${esc(state.tab === 'events' ? item.label_id ? item.text || (item.kind === 'point' ? '瞬时事件' : '持续事件') : item.kind === 'point' ? '瞬时事件 · 自由描述' : '持续事件 · 自由描述' : item.basis === 'audiovisual' ? '依据：画面与声音' : '依据：仅画面')}</span></span><span class="record-time">${esc(intervalText(item))}</span></button>`).join('') : `<div class="empty-records">${state.tab === 'events' ? '这一段发生了什么？<br>添加事件，标记时间点或持续区间。' : '用文字描述画面。<br>可以概述整段视频，也可以描述某个片段。'}</div>`;
  }
  function selectRecord(type, id, seek = true) { setTab(type); state.selected = {type, id}; renderRecords(); renderInspector(); renderTimeline(); const item = selectedRecord(); if (seek && item?.start != null) seekTime(item.start); }
  function addRecord() {
    if (!state.annotations || !['events', 'captions'].includes(state.tab)) return;
    const start = Math.min(state.time, Math.max(0, duration() - 0.001));
    const item = state.tab === 'events' ? {id: uid(), kind: 'interval', start: round(start), end: round(Math.min(duration(), start + 1)), label_id: state.labels[0]?.id || null, text: '', track_ids: [], reviewed: false} : {id: uid(), start: null, end: null, text: '', basis: 'visual', reviewed: false};
    state.annotations[state.tab].push(item); state.selected = {type: state.tab, id: item.id}; changed(state.tab); renderInspector(); $('recordText')?.focus();
  }
  function renderInspector() {
    if (state.tab === 'segmentation') { segmentation?.renderDetails(); return; }
    const item = selectedRecord();
    $('inspectorTitle').textContent = state.tab === 'frames' ? '抽帧说明' : item ? state.selected.type === 'events' ? '编辑事件' : '编辑描述' : '标注详情';
    if (state.tab === 'frames') { $('inspectorBody').innerHTML = '<p class="hint">先预览候选帧，再选择需要的画面。</p><p class="hint">抽取结果进入独立图片项目，可进行目标检测或实例分割。每张图片都保留原视频与时间戳，方便返回上下文。</p><p class="hint">播放预览与精确帧可能使用不同编码。逐帧按钮显示原视频解码后的实际帧。</p>'; return; }
    if (!item) { $('inspectorBody').innerHTML = '<div class="inspector-empty">选择一条标注，<br>或在当前时间添加新标注。</div>'; return; }
    const event = state.selected.type === 'events';
    const times = item.start != null ? `<div class="field-row"><label><span class="field-caption"><span>${event && item.kind === 'point' ? '发生时间' : '开始'}（秒）</span><button id="useCurrentStart" type="button" class="text-button">当前帧</button></span><input id="recordStart" type="number" min="0" max="${duration()}" step="any" value="${esc(item.start)}"></label>${!event || item.kind === 'interval' ? `<label><span class="field-caption"><span>结束（秒）</span><button id="useCurrentEnd" type="button" class="text-button">当前帧</button></span><input id="recordEnd" type="number" min="0" max="${duration()}" step="any" value="${esc(item.end)}"></label>` : ''}</div>` : '';
    $('inspectorBody').innerHTML = `<form id="recordForm">${event ? `<label>事件类型<select id="eventKind"><option value="interval" ${item.kind === 'interval' ? 'selected' : ''}>持续事件（时间区间）</option><option value="point" ${item.kind === 'point' ? 'selected' : ''}>瞬时事件（时间点）</option></select></label><label>事件标签<select id="eventLabel"><option value="">不选标签，仅自由描述</option>${state.labels.map(label => `<option value="${esc(label.id)}" ${label.id === item.label_id ? 'selected' : ''}>${esc(label.name)}</option>`).join('')}</select><small>标签和自由描述至少填写一项。</small></label>` : `<label>描述范围<select id="captionScope"><option value="whole" ${item.start == null ? 'selected' : ''}>整段视频</option><option value="segment" ${item.start != null ? 'selected' : ''}>指定片段</option></select></label><label>描述依据<select id="captionBasis"><option value="visual" ${item.basis === 'visual' ? 'selected' : ''}>仅画面</option><option value="audiovisual" ${item.basis === 'audiovisual' ? 'selected' : ''}>画面与声音</option></select></label>`}${times}<label>${event ? '自由描述' : '描述文本'}<textarea id="recordText" rows="5" maxlength="20000" placeholder="${event ? '描述发生的动作或事件…' : '描述画面中可确认的内容…'}">${esc(item.text)}</textarea></label>${event ? `<label>关联对象（可选，多选）<select id="eventTracks" multiple size="3">${(segmentation?.getTracks() || []).map(object => `<option value="${esc(object.id)}" ${(item.track_ids || []).includes(object.id) ? 'selected' : ''}>${esc(object.name)}</option>`).join('')}</select><small>按 Ctrl / ⌘ 选择多个对象；不关联也可标注事件。</small></label>` : ''}<label class="check"><input type="checkbox" id="recordReviewed" ${item.reviewed ? 'checked' : ''}>这条标注已人工审核</label><div class="inspector-actions"><button class="secondary" id="playSelection" type="button">播放此${item.start == null ? '视频' : '片段'}</button><button class="secondary" id="duplicateRecord" type="button">复制</button><button class="text-button danger" id="deleteRecord" type="button">删除</button></div></form>`;
    $('recordForm').addEventListener('submit', e => e.preventDefault());
    $('recordText').addEventListener('input', e => { item.text = e.target.value; changed(state.selected.type, item); $('recordReviewed').checked = false; });
    for (const [id, key] of [['recordStart', 'start'], ['recordEnd', 'end']]) $(id)?.addEventListener('input', e => { item[key] = e.target.value === '' ? NaN : Number(e.target.value); changed(state.selected.type, item); $('recordReviewed').checked = false; });
    $('eventTracks')?.addEventListener('change', e => { item.track_ids = [...e.target.selectedOptions].map(option => option.value); changed('events', item); $('recordReviewed').checked = false; });
    $('recordReviewed').addEventListener('change', e => { item.reviewed = e.target.checked; if (!item.reviewed) state.annotations.review[state.selected.type] = 'unreviewed'; changed(state.selected.type, null, true); });
    $('useCurrentStart')?.addEventListener('click', () => { item.start = round(state.timestamps[state.frame]); changed(state.selected.type, item); renderInspector(); });
    $('useCurrentEnd')?.addEventListener('click', () => { item.end = round(state.timestamps[state.frame]); changed(state.selected.type, item); renderInspector(); });
    $('eventKind')?.addEventListener('change', e => { item.kind = e.target.value; item.end = item.kind === 'point' ? null : round(Math.min(duration(), item.start + 1)); if (item.kind === 'interval' && item.end <= item.start) item.start = round(Math.max(0, item.end - 1)); changed('events', item); renderInspector(); });
    $('eventLabel')?.addEventListener('change', e => { item.label_id = e.target.value || null; changed('events', item); $('recordReviewed').checked = false; });
    $('captionScope')?.addEventListener('change', e => { item.start = e.target.value === 'whole' ? null : round(Math.min(state.time, Math.max(0, duration() - 0.001))); item.end = item.start == null ? null : round(Math.min(duration(), item.start + 1)); changed('captions', item); renderInspector(); });
    $('captionBasis')?.addEventListener('change', e => { item.basis = e.target.value; changed('captions', item); $('recordReviewed').checked = false; });
    $('deleteRecord').addEventListener('click', () => { if (!confirm('删除这条标注？保存草稿后生效。')) return; const type = state.selected.type; state.annotations[type] = state.annotations[type].filter(record => record.id !== item.id); state.selected = null; changed(type); renderInspector(); });
    $('duplicateRecord').addEventListener('click', () => { const copy = {...clone(item), id: uid(), reviewed: false}; state.annotations[state.selected.type].push(copy); state.selected.id = copy.id; changed(state.selected.type); renderInspector(); });
    $('playSelection').addEventListener('click', () => { $('player').currentTime = item.start || 0; $('loop').checked = item.start != null && item.end > item.start; if (!$('player').paused) $('player').pause(); run(togglePlay); });
  }
  function renderTimeline() {
    if (state.tab === 'segmentation') { segmentation?.renderTimeline(); return; }
    if (!state.annotations) return;
    const d = duration() || 1;
    $('ruler').innerHTML = [0, .25, .5, .75, 1].map(fraction => `<span>${timeText(d * fraction)}</span>`).join('');
    const items = [...state.annotations.events.map(item => ({...item, type: 'events'})), ...state.annotations.captions.map(item => ({...item, type: 'captions'}))].filter(item => item.start == null || Number.isFinite(item.start));
    if (!items.length) { $('timeline').innerHTML = '<div class="timeline-empty">添加事件或描述后，可在这里查看时间范围。</div>'; return; }
    $('timeline').innerHTML = items.map(item => {
      const selected = item.id === state.selected?.id, start = Math.max(0, Math.min(d, item.start || 0));
      const end = item.start == null ? d : item.kind === 'point' ? start : Math.min(d, Number.isFinite(item.end) ? item.end : start);
      const title = item.type === 'events' ? labelName(item.label_id) || item.text || '未填写事件' : item.text || '未填写描述';
      const labelColor = item.type === 'events' ? color(state.labels.find(label => label.id === item.label_id)?.color) : '';
      const barColor = labelColor ? `;border-color:${labelColor};background:${labelColor}22;color:${labelColor}` : '';
      return `<div class="timeline-lane"><button class="timeline-bar ${item.type === 'captions' ? 'caption' : ''} ${selected ? 'selected' : ''} ${item.kind === 'point' ? 'point' : ''}" style="left:${start / d * 100}%;width:${Math.max(.4, (end - start) / d * 100)}%${barColor}" data-timeline-id="${esc(item.id)}" data-type="${item.type}" title="${esc(title)} · ${esc(intervalText(item))}" aria-label="${esc(title)}，${esc(intervalText(item))}">${selected && item.start != null ? '<span class="timeline-handle start" data-handle="start" aria-hidden="true"></span>' : ''}${item.kind === 'point' ? '' : esc(title)}${selected && item.start != null && item.kind !== 'point' ? '<span class="timeline-handle end" data-handle="end" aria-hidden="true"></span>' : ''}</button></div>`;
    }).join('');
  }
  function setupTimeline() {
    let suppressClick = false;
    $('timeline').addEventListener('click', event => {
      if (suppressClick) { suppressClick = false; return; }
      const button = event.target.closest('[data-timeline-id]');
      if (button) selectRecord(button.dataset.type, button.dataset.timelineId);
      else if (event.target.closest('.timeline-lane')) { const rect = event.target.closest('.timeline-lane').getBoundingClientRect(); seekTime((event.clientX - rect.left) / rect.width * duration()); }
    });
    $('timeline').addEventListener('pointerdown', event => {
      const handle = event.target.closest('[data-handle]'); if (!handle) return;
      event.preventDefault(); event.stopPropagation();
      const button = handle.closest('[data-timeline-id]'), type = button.dataset.type, item = state.annotations[type].find(record => record.id === button.dataset.timelineId), key = handle.dataset.handle;
      const rect = button.parentElement.getBoundingClientRect(), original = item[key]; let moved = false;
      handle.setPointerCapture(event.pointerId);
      const move = e => { moved = true; let time = Math.max(0, Math.min(duration(), (e.clientX - rect.left) / rect.width * duration())); time = state.timestamps[frameAt(time)];
        if (item.kind !== 'point') time = key === 'start' ? Math.min(time, item.end - 0.001) : Math.max(time, item.start + 0.001);
        item[key] = round(Math.max(0, Math.min(duration(), time)));
        button.style.left = `${item.start / duration() * 100}%`; if (item.kind !== 'point') button.style.width = `${Math.max(.4, (item.end - item.start) / duration() * 100)}%`;
        if ($(key === 'start' ? 'recordStart' : 'recordEnd')) $(key === 'start' ? 'recordStart' : 'recordEnd').value = item[key];
      };
      const finish = e => { handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', finish); handle.removeEventListener('pointercancel', cancel); suppressClick = moved; if (moved) { changed(type, item); renderInspector(); } if (handle.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId); };
      const cancel = e => { item[key] = original; finish(e); renderTimeline(); renderInspector(); };
      handle.addEventListener('pointermove', move); handle.addEventListener('pointerup', finish); handle.addEventListener('pointercancel', cancel);
    });
  }
  function renderSample() {
    const count = state.sampleSelected.size;
    $('sampleSummary').innerHTML = state.sample.length ? `候选 ${state.sample.length} 帧 · 已选 ${count} 帧<button type="button" class="text-button" id="selectAllFrames">全选</button><button type="button" class="text-button" id="clearFrames">清空选择</button>` : '';
    $('sampleFrames').innerHTML = state.sample.map(frame => `<label class="sample-card"><img src="${videoPath()}/frames/${frame.frame_index}" loading="lazy" alt="第 ${frame.frame_index + 1} 帧"><input type="checkbox" data-sample-frame="${frame.frame_index}" ${state.sampleSelected.has(frame.frame_index) ? 'checked' : ''} aria-label="选择第 ${frame.frame_index + 1} 帧"><span>${timeText(frame.timestamp)} · #${frame.frame_index + 1}</span>${segmentation?.frameHasGeometry(frame.frame_index) ? '<span class="seg-sample-mark">含分割标注</span>' : ''}</label>`).join('');
    $('extractForm').hidden = !state.sample.length;
    $('extractForm').querySelector('button[type=submit]').disabled = !count;
    $('selectAllFrames')?.addEventListener('click', () => { state.sampleSelected = new Set(state.sample.map(frame => frame.frame_index)); renderSample(); });
    $('clearFrames')?.addEventListener('click', () => { state.sampleSelected.clear(); renderSample(); });
  }
  function sampleOptions() {
    const mode = $('sampleMode').value;
    document.querySelectorAll('.sample-range').forEach(field => { field.hidden = !['interval', 'count'].includes(mode); field.querySelector('input').disabled = field.hidden; }); $('intervalField').hidden = mode !== 'interval'; $('sampleInterval').disabled = mode !== 'interval'; $('countField').hidden = mode !== 'count'; $('sampleCount').disabled = mode !== 'count';
  }
  async function previewFrames(event) {
    event.preventDefault(); if (!state.video) return;
    const mode = $('sampleMode').value;
    if (mode === 'events' && hasUnsaved() && !await saveAll()) return;
    const button = $('sampleForm').querySelector('button'); button.disabled = true;
    const pid = state.project.id, vid = state.video.id;
    try {
      const payload = {mode};
      if (mode === 'current') payload.frame_index = state.frame;
      if (['interval', 'count'].includes(mode)) { payload.start = Number($('sampleStart').value); payload.end = Number($('sampleEnd').value); if (payload.end <= payload.start || payload.end > duration()) throw new Error('采样结束时间需要晚于开始时间，并且不超过视频时长。'); }
      if (mode === 'interval') payload.interval = Number($('sampleInterval').value);
      if (mode === 'count') payload.count = Number($('sampleCount').value);
      const result = await post(`${videoPath()}/extract/preview`, payload);
      if (!sameVideo(pid, vid)) return;
      state.sample = result.frames || []; state.sampleSelected = new Set(state.sample.map(frame => frame.frame_index)); renderSample();
      notify(state.sample.length ? `已生成 ${state.sample.length} 个候选帧，可取消不需要的画面。` : '没有候选帧。请检查范围，或先添加并保存事件。');
    } finally { button.disabled = false; }
  }
  async function extractFrames(event) {
    event.preventDefault(); if (!state.video || !state.sampleSelected.size) return;
    const includeSegmentation = $('extractTask').value === 'instance_segmentation' && $('includeSegmentation').checked;
    if (includeSegmentation && !await saveAll()) return;
    const objectLabels = [...new Set([...names($('extractLabels').value), ...(includeSegmentation ? (segmentation?.getLabels() || []).map(label => label.name) : [])])]; if (!objectLabels.length) throw new Error('请至少填写一个图片对象类别，例如：人。');
    const button = $('extractForm').querySelector('button'); button.disabled = true;
    const projectId = state.project.id;
    try {
      const result = await post(`${videoPath()}/extract`, {frame_indices: [...state.sampleSelected].sort((a, b) => a - b), task: $('extractTask').value, labels: objectLabels, include_segmentation: includeSegmentation});
      if (state.project?.id !== projectId) return;
      const job = result.job || result; state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)]; renderJobs(); schedulePoll(); notify('抽帧任务已提交。完成后可从后台任务打开图片标注项目。');
    } finally { button.disabled = !state.sampleSelected.size; }
  }
  const jobTitles = {import: '导入视频', video_import: '导入视频', extract: '抽帧', frame_extract: '抽帧', extraction: '抽帧'};
  const jobStatuses = {queued: '排队中', pending: '排队中', running: '处理中', cancelling: '正在取消', completed: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断'};
  function renderJobs() {
    $('jobsPanel').hidden = !state.project; $('jobCount').textContent = state.jobs.length ? `(${state.jobs.length})` : '';
    $('jobs').innerHTML = state.jobs.length ? state.jobs.map(job => {
      const progress = job.progress || {}, total = Number(progress.total) || 0, current = Number(progress.current) || 0, targetProject = job.result?.project_id;
      const errors = job.result?.errors || job.errors;
      return `<div class="job"><div><strong>${esc(jobTitles[job.kind || job.type] || job.kind || job.type || '视频任务')} · ${esc(jobStatuses[job.status] || job.status)}</strong><small>${esc(progress.message || job.message || '')}${total ? ` · ${current} / ${total}` : ''}</small>${job.error ? `<small class="danger">${esc(typeof job.error === 'string' ? job.error : JSON.stringify(job.error))}</small>` : ''}${errors?.length ? `<small class="danger">${esc(JSON.stringify(errors))}</small>` : ''}${isBusyJob(job) ? `<progress ${total ? `max="${total}" value="${current}"` : ''} aria-label="任务进度"></progress>` : ''}</div><div class="job-actions">${job.status === 'completed' && targetProject ? `<a href="/?project=${encodeURIComponent(targetProject)}" data-leave>打开图片工作台 ↗</a>` : ''}${isBusyJob(job) ? `<button class="secondary" data-job="${esc(job.id)}" data-action="cancel">取消</button>` : ['failed', 'cancelled', 'interrupted'].includes(job.status) ? `<button class="secondary" data-job="${esc(job.id)}" data-action="retry">重试</button>` : ''}</div></div>`;
    }).join('') : '<p class="hint">暂无后台任务。</p>';
  }
  function schedulePoll(delay = 1200) { clearTimeout(state.poll); if (state.project && state.jobs.some(isBusyJob)) state.poll = setTimeout(() => run(pollJobs), delay); }
  async function pollJobs() {
    if (!state.project) return;
    const token = state.projectToken, projectId = state.project.id;
    try {
      const previous = new Map(state.jobs.map(job => [job.id, job.status]));
      const result = await api(`${projectPath()}/jobs`);
      if (token !== state.projectToken) return;
      state.jobs = Array.isArray(result) ? result : result.jobs || []; renderJobs();
      const finished = state.jobs.some(job => job.status === 'completed' && previous.get(job.id) !== 'completed');
      const importing = state.jobs.some(job => isBusyJob(job) && ['import', 'video_import'].includes(job.kind || job.type));
      if (finished || importing || !state.videos.length) {
        const details = await api(`${API}/projects/${encodeURIComponent(projectId)}`); if (token !== state.projectToken) return;
        state.videos = details.videos || []; renderVideos();
        if (!state.video && state.videos.length) await loadVideo(state.videos[0].id);
      }
      schedulePoll();
    } catch (error) { if (token === state.projectToken) { notify(`后台任务状态暂时无法获取：${error.message}`, true); schedulePoll(4000); } }
  }
  async function jobAction(event) {
    const button = event.target.closest('[data-job]'); if (!button) return; button.disabled = true;
    const projectId = state.project.id;
    try { const result = await post(`${projectPath()}/jobs/${encodeURIComponent(button.dataset.job)}/${button.dataset.action}`, {}); if (state.project?.id !== projectId) return; const job = result.job || result; state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)]; await pollJobs(); schedulePoll(); }
    finally { button.disabled = false; }
  }
  function renderLabelDraft() {
    $('labelRows').innerHTML = state.labelDraft.map((label, index) => `<div class="label-row"><input type="color" value="${color(label.color)}" data-label-color="${index}" aria-label="标签 ${index + 1} 颜色"><input type="text" value="${esc(label.name)}" data-label-name="${index}" maxlength="160" aria-label="标签 ${index + 1} 名称"><button type="button" class="icon-button danger" data-delete-label="${index}" aria-label="删除标签 ${esc(label.name)}">×</button></div>`).join('');
  }
  function openLabels(scope = 'events') { if (scope === 'objects' && segmentation && !segmentation.beforeContextChange()) return; state.labelScope = typeof scope === 'string' ? scope : 'events'; state.labelDraft = clone(state.labelScope === 'objects' ? segmentation.getLabels() : state.labels); $('labelsDialogTitle').textContent = state.labelScope === 'objects' ? '管理分割对象类别' : '管理事件标签'; $('newLabelName').placeholder = state.labelScope === 'objects' ? '新增对象类别' : '新增事件标签'; state.importedLabels = []; $('labelsError').textContent = ''; $('labelsFile').value = ''; $('labelPreview').hidden = true; renderLabelDraft(); $('labelsDialog').showModal(); }
  function addDraftLabel(name, customColor) { name = name.trim(); if (!name) return; if (state.labelDraft.some(label => label.name === name)) throw new Error(`标签“${name}”已存在。`); state.labelDraft.push({id: uid(), name, color: color(customColor || palette[state.labelDraft.length % palette.length])}); }
  async function readLabels(event) {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('标签文件不能超过 2 MB。');
      const text = await file.text(); let entries;
      if (file.name.toLowerCase().endsWith('.json')) { const parsed = JSON.parse(text); entries = Array.isArray(parsed) ? parsed : parsed.labels; if (!Array.isArray(entries)) throw new Error('JSON 需要为数组或包含 labels 数组。'); }
      else entries = text.split(/\r?\n/).map(name => name.trim()).filter(Boolean);
      state.importedLabels = entries.map(entry => typeof entry === 'string' ? {name: entry} : entry).filter(entry => entry && typeof entry.name === 'string' && entry.name.trim());
      $('labelPreview').hidden = false; $('labelPreview').innerHTML = `识别到 ${state.importedLabels.length} 个标签：${esc(state.importedLabels.map(entry => entry.name).join('、'))}<button type="button" class="secondary" id="applyLabelImport">添加到标签草稿（跳过同名标签）</button>`;
      $('applyLabelImport').addEventListener('click', () => { for (const label of state.importedLabels) if (!state.labelDraft.some(item => item.name === label.name.trim())) addDraftLabel(label.name, label.color); renderLabelDraft(); $('labelPreview').hidden = true; });
      $('labelsError').textContent = '';
    } catch (error) { $('labelsError').textContent = error.message; $('labelPreview').hidden = true; }
  }
  async function saveLabels(event) {
    event.preventDefault(); const button = $('labelsForm').querySelector('button[type=submit]'); button.disabled = true;
    try {
      state.labelDraft.forEach(label => label.name = label.name.trim());
      if (state.labelDraft.some(label => !label.name)) throw new Error('标签名称不能为空。');
      if (new Set(state.labelDraft.map(label => label.name)).size !== state.labelDraft.length) throw new Error('标签名称不能重复。');
      const result = await api(`${projectPath()}/${state.labelScope === 'objects' ? 'object-labels' : 'labels'}`, {method: 'PUT', body: JSON.stringify({labels: state.labelDraft})});
      if (state.labelScope === 'objects') segmentation.setLabels(result.labels || state.labelDraft); else state.labels = result.labels || state.labelDraft; $('labelsDialog').close(); renderRecords(); renderTimeline(); renderInspector(); notify(state.labelScope === 'objects' ? '分割对象类别已保存。' : '事件标签已保存。');
    } catch (error) { $('labelsError').textContent = error.message; } finally { button.disabled = false; }
  }
  async function createProject(event) {
    event.preventDefault(); if (!await mayLeave()) return; const form = $('projectForm'), button = form.querySelector('button[type=submit]'); button.disabled = true; $('projectError').textContent = '';
    try {
      const values = new FormData(form); const result = await post(`${API}/projects`, {name: values.get('name').trim(), paths: values.get('paths').split(/\r?\n/).map(path => path.trim()).filter(Boolean), labels: names(values.get('labels')), split: values.get('split')});
      $('projectDialog').close(); form.reset(); await loadProjects(); await loadProject(result.project.id);
    } catch (error) { $('projectError').textContent = error.message; if (!$('projectDialog').open) notify(error.message, true); } finally { button.disabled = false; }
  }
  async function reviewTask() {
    if (!state.annotations || !['events', 'captions'].includes(state.tab)) return;
    const type = state.tab, title = type === 'events' ? '事件' : '描述';
    if (!confirm(`确认已完整检查这段视频的${title}标注？\n${state.annotations[type].length ? '现有条目将全部标记为已审核。' : '当前没有条目，将明确记录为已审核的空标注。'}\n确认后将保存当前视频的全部修改。`)) return;
    state.annotations[type].forEach(item => item.reviewed = true); state.annotations.review[type] = 'reviewed'; changed(type, null, true); renderInspector(); await saveAll();
  }
  async function exportAnnotations() {
    if (!state.project) return;
    if (hasUnsaved() && !await saveAll()) return;
    const project = state.project, button = $('export'); button.disabled = true; $('transferError').textContent = '';
    try {
      const bundle = await api(`${projectPath()}/export?policy=${encodeURIComponent($('exportPolicy').value)}`);
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {type: 'application/json'}), url = URL.createObjectURL(blob), anchor = document.createElement('a');
      anchor.href = url; anchor.download = `${project.name.replace(/[\\/:*?"<>|]/g, '_')}-video-annotations.json`; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (error) { $('transferError').textContent = error.message; } finally { button.disabled = false; }
  }
  async function readAnnotations(event) {
    const file = event.target.files[0]; state.importBundle = null; $('importAnnotations').disabled = true; $('annotationPreview').hidden = true;
    if (!file) return;
    try {
      if (file.size > 50 * 1024 * 1024) throw new Error('标注文件不能超过 50 MB。');
      const bundle = JSON.parse(await file.text()); if (!bundle || !Array.isArray(bundle.videos)) throw new Error('需要包含 videos 数组的 VisionRefine 视频原生标注文件。');
      state.importBundle = bundle; $('annotationPreview').hidden = false; $('annotationPreview').textContent = `${file.name}\n包含 ${bundle.videos.length} 段视频、${bundle.labels?.length || 0} 个标签。\n目标项目：${state.project.name}\n导入内容将重置为待审核；素材匹配或版本冲突会由服务器阻止。`;
      $('importAnnotations').disabled = false; $('transferError').textContent = '';
    } catch (error) { $('transferError').textContent = error.message; }
  }
  async function importAnnotations() {
    if (!state.importBundle || !state.project) return;
    if (!await mayLeave()) return;
    if (!confirm('将导入标注到当前项目。匹配视频的已有标注可能被更新。确认已保存必要备份并继续？')) return;
    const button = $('importAnnotations'); button.disabled = true;
    try {
      const expected = {}, expectedSegmentation = {};
      // Read every current revision before mutation so concurrent updates cannot be overwritten.
      const revisions = await Promise.all(state.videos.map(async video => { const base = `${projectPath()}/videos/${encodeURIComponent(video.id)}`; const [annotations, segmentationDoc] = await Promise.all([api(`${base}/annotations`), api(`${base}/segmentation`)]); return [video.id, annotations.revision, segmentationDoc.revision]; }));
      revisions.forEach(([id, revision, segmentationRevision]) => { expected[id] = revision; expectedSegmentation[id] = segmentationRevision; });
      const projectId = state.project.id, videoId = state.video?.id;
      await post(`${projectPath()}/annotations/import`, {bundle: state.importBundle, expected_revisions: expected, expected_segmentation_revisions: expectedSegmentation});
      $('transferDialog').close(); state.importBundle = null; $('annotationsFile').value = ''; await loadProject(projectId, {video: videoId}); notify('标注已导入，请重新审核导入内容。');
    } catch (error) { $('transferError').textContent = error.message; } finally { button.disabled = !state.importBundle; }
  }
  function bind() {
    document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
    document.addEventListener('click', event => { const anchor = event.target.closest('a[data-leave]'); if (!anchor || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return; event.preventDefault(); run(async () => { if (await mayLeave()) { state.dirty = false; segmentation?.clearDirty(); location.href = anchor.href; } }); });
    window.addEventListener('beforeunload', event => { if (hasUnsaved() || isSaving()) { event.preventDefault(); event.returnValue = ''; } });
    const openProject = () => { $('projectError').textContent = ''; $('projectDialog').showModal(); };
    $('newProject').addEventListener('click', openProject); $('welcomeCreate').addEventListener('click', openProject);
    $('projectForm').addEventListener('submit', createProject);
    $('projectSelect').addEventListener('change', event => run(async () => { const id = event.target.value, oldId = state.project?.id || ''; event.target.value = oldId; if (id && id !== oldId && await mayLeave()) await loadProject(id); }));
    $('videoSearch').addEventListener('input', renderVideos);
    $('videoList').addEventListener('click', event => run(async () => { const button = event.target.closest('[data-video]'); if (button && button.dataset.video !== state.video?.id && await mayLeave()) await loadVideo(button.dataset.video); }));
    $('save').addEventListener('click', saveAll);
    $('play').addEventListener('click', () => run(togglePlay));
    $('previousFrame').addEventListener('click', () => setFrame(state.frame - 1)); $('nextFrame').addEventListener('click', () => setFrame(state.frame + 1));
    $('seek').addEventListener('input', event => seekTime(Number(event.target.value)));
    $('speed').addEventListener('change', event => $('player').playbackRate = Number(event.target.value));
    $('player').addEventListener('timeupdate', () => { if (!state.video || state.pendingFrame != null) return; const item = selectedRecord(); if (!$('player').paused && $('loop').checked && item?.start != null && item.end > item.start && $('player').currentTime >= item.end) $('player').currentTime = item.start; updateTime($('player').currentTime); if ($('player').paused && !$('player').seeking) refreshExactFrame(); });
    $('player').addEventListener('play', () => { if (segmentation?.isDraftDirty() && !segmentation.beforeContextChange()) { $('player').pause(); return; } $('play').textContent = '暂停'; $('exactFrame').hidden = true; $('exactBadge').hidden = true; segmentation?.onPlayback(); });
    $('player').addEventListener('pause', () => { $('play').textContent = '播放'; if (state.video && state.pendingFrame == null) { updateTime($('player').currentTime); if (!$('player').seeking) refreshExactFrame(); } segmentation?.onPlayback(); });
    $('player').addEventListener('seeking', () => { if (!state.video || state.pendingFrame != null) return; const target = frameAt($('player').currentTime); if (segmentation && !segmentation.beforeFrame(target)) { $('player').currentTime = state.timestamps[state.frame]; return; } $('exactFrame').hidden = true; $('exactBadge').hidden = true; segmentation?.onSeeking(); });
    $('player').addEventListener('seeked', () => { if (!state.video || state.pendingFrame != null) return; updateTime($('player').currentTime); if ($('player').paused) refreshExactFrame(); segmentation?.onPlayback(); });
    $('player').addEventListener('ended', () => { const item = selectedRecord(); if ($('loop').checked && item?.start != null && item.end > item.start) { $('player').currentTime = item.start; run(togglePlay); } });
    $('player').addEventListener('error', () => { if (state.video) notify('播放预览暂时不可用。可使用逐帧按钮继续标注；请查看后台任务或重试导入。', true); });
    $('exactFrame').addEventListener('load', () => { if ($('player').paused && !$('player').seeking && state.video && $('exactFrame').dataset.video === state.video.id && Number($('exactFrame').dataset.frame) === state.frame) { $('exactFrame').hidden = false; $('exactBadge').hidden = false; $('exactBadge').textContent = '精确帧预览'; } });
    $('exactFrame').addEventListener('error', () => { $('exactFrame').hidden = true; $('exactBadge').hidden = true; if (state.video) notify('精确帧读取失败，请重新选择帧或检查视频源文件。', true); });
    document.querySelectorAll('[data-tab]').forEach(button => { button.addEventListener('click', () => setTab(button.dataset.tab)); button.addEventListener('keydown', event => { if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); const tabs = [...document.querySelectorAll('[data-tab]')], next = tabs[(tabs.indexOf(button) + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]; setTab(next.dataset.tab); next.focus(); }); });
    $('addRecord').addEventListener('click', addRecord);
    $('recordsBody').addEventListener('click', event => { const button = event.target.closest('[data-record]'); if (button) selectRecord(state.tab, button.dataset.record); });
    $('reviewTask').addEventListener('click', () => run(reviewTask));
    setupTimeline();
    $('extractTask').addEventListener('change', () => { $('includeSegmentationField').hidden = $('extractTask').value !== 'instance_segmentation'; $('extractLabels').required = !$('includeSegmentation').checked || $('extractTask').value !== 'instance_segmentation'; });
    $('includeSegmentation').addEventListener('change', () => { $('extractLabels').required = !$('includeSegmentation').checked; });
    $('sampleMode').addEventListener('change', sampleOptions); $('sampleForm').addEventListener('submit', event => run(() => previewFrames(event))); $('extractForm').addEventListener('submit', event => run(() => extractFrames(event)));
    $('sampleFrames').addEventListener('change', event => { if (!event.target.matches('[data-sample-frame]')) return; const index = Number(event.target.dataset.sampleFrame); if (event.target.checked) state.sampleSelected.add(index); else state.sampleSelected.delete(index); renderSample(); });
    $('jobs').addEventListener('click', event => run(() => jobAction(event))); $('refreshJobs').addEventListener('click', event => { event.preventDefault(); run(pollJobs); });
    $('appendVideos').addEventListener('click', () => { $('appendError').textContent = ''; $('appendDialog').showModal(); });
    $('appendForm').addEventListener('submit', async event => {
      event.preventDefault(); const form = $('appendForm'), button = form.querySelector('button[type=submit]'); button.disabled = true;
      try { const paths = new FormData(form).get('paths').split(/\r?\n/).map(path => path.trim()).filter(Boolean); const result = await post(`${projectPath()}/import`, {paths}); const job = result.job || result; state.jobs = [job, ...state.jobs.filter(item => item.id !== job.id)]; renderJobs(); schedulePoll(); $('appendDialog').close(); form.reset(); notify('视频追加任务已提交，可继续标注当前素材。'); }
      catch (error) { $('appendError').textContent = error.message; } finally { button.disabled = false; }
    });
    $('manageLabels').addEventListener('click', openLabels); $('labelsForm').addEventListener('submit', saveLabels);
    $('addLabel').addEventListener('click', () => { try { addDraftLabel($('newLabelName').value); $('newLabelName').value = ''; renderLabelDraft(); $('labelsError').textContent = ''; } catch (error) { $('labelsError').textContent = error.message; } });
    $('newLabelName').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); $('addLabel').click(); } });
    $('labelRows').addEventListener('input', event => { if (event.target.dataset.labelName != null) state.labelDraft[Number(event.target.dataset.labelName)].name = event.target.value; if (event.target.dataset.labelColor != null) state.labelDraft[Number(event.target.dataset.labelColor)].color = event.target.value; });
    $('labelRows').addEventListener('click', event => { const button = event.target.closest('[data-delete-label]'); if (!button) return; const index = Number(button.dataset.deleteLabel), label = state.labelDraft[index]; if (state.labelScope === 'objects' ? segmentation.getTracks().some(item => item.label_id === label.id) : state.annotations?.events.some(item => item.label_id === label.id)) { $('labelsError').textContent = '当前视频已有标注使用此类别，请先修改对应标注并保存。其他视频的引用也会在保存时检查。'; return; } state.labelDraft.splice(index, 1); renderLabelDraft(); });
    $('labelsFile').addEventListener('change', readLabels);
    $('transfer').addEventListener('click', () => { $('transferError').textContent = ''; $('transferDialog').showModal(); });
    $('export').addEventListener('click', exportAnnotations); $('annotationsFile').addEventListener('change', readAnnotations); $('importAnnotations').addEventListener('click', () => run(importAnnotations));
    document.addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (!document.querySelector('dialog[open]')) saveAll(); return; }
      if (!state.video || document.querySelector('dialog[open]') || event.target.closest('input,textarea,select,button,canvas,[contenteditable=true]')) return;
      if (event.code === 'Space') { event.preventDefault(); run(togglePlay); }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); setFrame(state.frame + (event.key === 'ArrowLeft' ? -1 : 1)); }
    });
  }
  async function init() {
    segmentation = window.VideoSegmentationWorkspace.create({state, uid, api, post, videoPath, projectPath, notify, run, setFrame, updateDirty, saveAll, openLabels, replaceAnnotations(value) { if (!value) return; state.annotations = normalizeAnnotations(value); state.dirty = false; renderRecords(); renderInspector(); }});
    bind(); sampleOptions();
    const params = new URLSearchParams(location.search);
    const results = await Promise.allSettled([api(`${API}/capabilities`), loadProjects()]);
    if (results[1].status === 'rejected') throw results[1].reason;
    $('connectionStatus').textContent = '本地服务已连接';
    if (results[0].status === 'fulfilled') {
      const capabilities = results[0].value;
      if (capabilities.available === false || capabilities.ffmpeg === false) notify(capabilities.error || '视频处理依赖尚未就绪，请安装项目的视频扩展依赖后重启服务。', true);
    }
    const id = params.get('project') || state.projects[0]?.id;
    if (id) await loadProject(id, {video: params.get('video'), frame: params.get('frame')});
  }
  run(init);
})();
