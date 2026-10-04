(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  const statusNames = {visible: '可见 · 已分割', occluded: '完全遮挡', outside: '离开画面', missing: '尚未标注', inactive: '对象范围外'};
  const emptyDocument = () => ({revision: 0, tracks: [], reviewed_ranges: [], updated_at: null});
  window.VideoSegmentationWorkspace = {create(context) {
    const ctx = context;
    let document = emptyDocument(), labels = [], trackId = null, dirty = false, edits = 0, saving = false, active = false;
    let canvas = null, canvasKey = null, canvasStatus = {}, pendingPreview = null, previewBusy = false, generation = 0, currentTool = 'polygon';
    const state = () => ctx.state;
    const track = () => document.tracks.find(item => item.id === trackId);
    const frameCount = () => state().timestamps.length;
    const frame = () => state().frame;
    const endpoint = () => `${ctx.videoPath()}/segmentation`;
    const label = id => labels.find(item => item.id === id);
    const safeColor = value => /^#[0-9a-f]{6}$/i.test(value || '') ? value : '#146c4b';
    const key = () => `${state().video?.id}:${trackId || ''}:${frame()}`;
    const busy = () => saving || previewBusy;
    const canEditCanvas = () => Boolean(active && state().video && $('player').paused && !$('player').seeking && track() && frame() >= track().start_frame && frame() <= track().end_frame && canvasStatus.ready && !busy() && !pendingPreview);
    function frameState(object, index) {
      if (!object || index < object.start_frame || index > object.end_frame) return {visibility: 'inactive'};
      const explicit = object.keyframes.find(item => item.frame_index === index);
      if (explicit) return explicit;
      const range = object.visibility_ranges.find(item => item.start_frame <= index && item.end_frame >= index);
      return range ? {...range, fromRange: true, geometry: null} : {visibility: 'missing', geometry: null, reviewed: false};
    }
    function intervals(object, reviewedOnly = false, start = object.start_frame, end = object.end_frame) {
      const source = [
        ...object.keyframes.filter(item => !reviewedOnly || item.reviewed).map(item => [item.frame_index, item.frame_index]),
        ...object.visibility_ranges.filter(item => !reviewedOnly || item.reviewed).map(item => [item.start_frame, item.end_frame])
      ].map(([a, b]) => [Math.max(a, start), Math.min(b, end)]).filter(([a, b]) => a <= b).sort((a, b) => a[0] - b[0]);
      const merged = [];
      for (const [a, b] of source) { const last = merged.at(-1); if (last && a <= last[1] + 1) last[1] = Math.max(last[1], b); else merged.push([a, b]); }
      return merged;
    }
    const coveredCount = (object, reviewedOnly = false) => intervals(object, reviewedOnly).reduce((total, [a, b]) => total + b - a + 1, 0);
    function gap(object, start, end, reviewedOnly = true) {
      start = Math.max(start, object.start_frame); end = Math.min(end, object.end_frame); if (start > end) return null;
      let cursor = start; for (const [a, b] of intervals(object, reviewedOnly, start, end)) { if (a > cursor) return cursor; cursor = Math.max(cursor, b + 1); }
      return cursor <= end ? cursor : null;
    }
    function clearFrameSpan(object, start, end) {
      object.keyframes = object.keyframes.filter(item => item.frame_index < start || item.frame_index > end);
      object.visibility_ranges = object.visibility_ranges.flatMap(range => {
        if (range.end_frame < start || range.start_frame > end) return [range];
        const result = [];
        if (range.start_frame < start) result.push({...range, end_frame: start - 1});
        if (range.end_frame > end) result.push({...range, start_frame: end + 1});
        return result;
      });
    }
    function changed(object = null, {keepReview = false, renderInspector = true} = {}) {
      dirty = true; edits++;
      if (!keepReview) { document.reviewed_ranges = []; if (object) object.reviewed = false; }
      ctx.updateDirty(); renderList(); renderTimeline(); if (renderInspector) renderDetails();
    }
    function ensureCanvas() {
      if (canvas || !window.VideoSegmentationCanvas) return canvas;
      canvas = window.VideoSegmentationCanvas.create({canvas: $('segmentationCanvas'), onChange: geometryChanged,
        onStatus(value) { canvasStatus = value; renderCanvasTools(); ctx.updateDirty(); },
        onSelection(value) { const id = typeof value === 'string' ? value : value?.id; if (id && id !== trackId) selectTrack(id); }});
      canvas.setTool(currentTool); return canvas;
    }
    function geometryChanged(geometry, metadata = {}) {
      const object = track();
      if (!active || !object || !state().video || !$('player').paused || metadata.key && metadata.key !== key()) return;
      if (frame() < object.start_frame || frame() > object.end_frame) return;
      clearFrameSpan(object, frame(), frame());
      if (geometry) object.keyframes.push({frame_index: frame(), visibility: 'visible', geometry: clone(geometry), reviewed: false});
      object.keyframes.sort((a, b) => a.frame_index - b.frame_index);
      pendingPreview = null; changed(object); canvasKey = key();
    }
    async function syncFrame(force = false) {
      const visible = active && state().video && $('player').paused && !$('player').seeking;
      $('segmentationCanvas').hidden = !visible; $('segCanvasTools').hidden = !active;
      if (!visible) { canvas?.setEnabled(false); renderCanvasTools(); return; }
      if (!ensureCanvas()) { ctx.notify('分割画布组件未加载，请刷新页面。', true); return; }
      const nextKey = key(), object = track(), current = frameState(object, frame());
      if (canvasKey === nextKey && !force) { canvas.setEnabled(Boolean(canvasStatus.ready && object && current.visibility !== 'inactive' && !busy() && !pendingPreview)); $('segmentationCanvas').style.visibility = canvasStatus.ready ? 'visible' : 'hidden'; renderCanvasTools(); return; }
      canvasKey = nextKey; canvas.setEnabled(false); $('segmentationCanvas').style.visibility = 'hidden'; pendingPreview = null;
      const overlays = document.tracks.filter(item => item.id !== trackId).flatMap(item => { const value = frameState(item, frame()); return value.visibility === 'visible' && value.geometry ? [{id: item.id, color: safeColor(label(item.label_id)?.color), geometry: value.geometry}] : []; });
      const token = generation;
      try {
        const ready = await canvas.load({imageUrl: `${ctx.videoPath()}/frames/${frame()}`, width: state().video.width, height: state().video.height,
          geometry: current.visibility === 'visible' ? current.geometry : null, overlays, key: nextKey, color: safeColor(label(object?.label_id)?.color)});
        if (token !== generation || nextKey !== key() || !active || !$('player').paused || $('player').seeking) return;
        if (!ready) canvasKey = null;
        $('segmentationCanvas').style.visibility = ready ? 'visible' : 'hidden'; canvas.setEnabled(Boolean(ready && object && current.visibility !== 'inactive' && !busy() && !pendingPreview)); renderCanvasTools(); renderDetails();
      } catch (error) { if (token === generation) ctx.notify(`分割画布加载失败：${error.message}`, true); }
    }
    function renderCanvasTools() {
      if (!$('segCanvasStatus')) return;
      const object = track(), current = frameState(object, frame()), editing = active && $('player').paused && object && current.visibility !== 'inactive' && canvasStatus.ready && !busy() && !pendingPreview;
      documentQuery('[data-seg-tool]').forEach(button => { button.disabled = !editing; button.classList.toggle('active', button.dataset.segTool === (canvasStatus.tool || currentTool)); });
      for (const id of ['segFinish', 'segCancelDraft', 'segAppendContour', 'segDeleteVertex', 'segDeleteContour', 'segRadius']) $(id).disabled = !editing;
      $('segUndo').disabled = !editing || !canvasStatus.canUndo; $('segRedo').disabled = !editing || !canvasStatus.canRedo;
      $('segZoom').textContent = canvasStatus.zoom ? `${Math.round(canvasStatus.zoom * 100)}%` : '';
      $('segCanvasStatus').textContent = !active ? '' : !$('player').paused ? '播放中暂停几何编辑，暂停后显示对应精确帧。' : !object ? '新建或选择对象，然后在当前帧标注可见像素。' : current.visibility === 'inactive' ? '当前帧在该对象范围之外。请定位到对象范围内，或调整范围。' : canvasStatus.error || !canvasStatus.ready ? canvasStatus.error || '正在读取精确帧…' : pendingPreview ? '正在预览轮廓调整。应用或取消后继续编辑。' : canvasStatus.dirtyDraft ? '轮廓尚未完成：按 Enter 完成，Esc 取消。' : `第 ${frame() + 1} 帧 · ${statusNames[current.visibility]} · 滚轮缩放，空格 + 拖动平移。`;
    }
    function documentQuery(selector) { return window.document.querySelectorAll(selector); }
    function beforeContextChange() {
      if (busy()) { ctx.notify('正在处理分割修改，请等待完成。'); return false; }
      if (pendingPreview) { if (!confirm('当前轮廓调整仍在预览。取消预览并继续？')) return false; cancelPreview(); }
      if (canvas?.isDraftDirty()) { if (!confirm('当前轮廓还没有闭合。放弃未完成的轮廓并继续？已完成的分割会保留。')) return false; canvas.cancelDraft(); }
      return true;
    }
    function selectTrack(id) {
      if (id === trackId) return;
      if (!beforeContextChange()) return;
      trackId = id; canvasKey = null; renderList(); renderDetails(); renderTimeline(); syncFrame();
    }
    function renderList() {
      $('segTrackCount').textContent = document.tracks.length;
      const all = document.tracks.reduce((sum, object) => sum + object.end_frame - object.start_frame + 1, 0);
      const count = document.tracks.reduce((sum, object) => sum + coveredCount(object, true), 0);
      $('segOverallProgress').textContent = `${document.tracks.length} 个对象 · ${count} / ${all} 对象帧已审核`;
      $('segTrackList').innerHTML = document.tracks.length ? document.tracks.map(object => {
        const total = object.end_frame - object.start_frame + 1, annotated = coveredCount(object), reviewed = coveredCount(object, true), current = frameState(object, frame());
        return `<button class="seg-track ${object.id === trackId ? 'selected' : ''}" data-seg-track="${esc(object.id)}"><span class="seg-track-color" style="background:${safeColor(label(object.label_id)?.color)}"></span><span class="seg-track-main"><strong>${esc(object.name)}</strong><small>${esc(label(object.label_id)?.name || '未分类')} · #${object.start_frame + 1}–${object.end_frame + 1} · ${statusNames[current.visibility]}</small><span class="seg-progress"><i style="width:${total ? reviewed / total * 100 : 0}%"></i></span></span><span class="seg-track-count">已标 ${annotated}/${total}<br>已审 ${reviewed}/${total}${object.reviewed ? ' ✓' : ''}</span></button>`;
      }).join('') : '<div class="empty-records">创建一个对象，在不同帧持续使用同一个 ID。<br>每帧的轮廓单独编辑，遮挡时仅标注状态。</div>';
      $('segReviewedRanges').innerHTML = (document.reviewed_ranges || []).map((range, index) => `<span class="seg-range-chip">已审核 #${range.start_frame + 1}–${range.end_frame + 1}<button class="text-button" data-remove-review="${index}" aria-label="撤销区间审核">×</button></span>`).join('');
    }
    function renderTimeline() {
      if (!active || !state().video) return;
      const total = frameCount() || 1;
      $('ruler').innerHTML = [0, .25, .5, .75, 1].map(fraction => `<span>#${Math.min(total, Math.floor((total - 1) * fraction) + 1)}</span>`).join('');
      $('timeline').innerHTML = `<div class="seg-timeline-legend"><span>空白：未标</span><span class="visible">实线：分割</span><span class="occluded">斜纹：遮挡</span><span class="outside">虚线：离开</span><span>✓：已审核</span></div>` + (document.tracks.length ? document.tracks.map(object => {
        const color = safeColor(label(object.label_id)?.color);
        const ranges = object.visibility_ranges.map(range => `<button class="seg-timeline-range ${range.visibility} ${range.reviewed ? 'reviewed' : ''}" style="left:${range.start_frame / total * 100}%;width:${(range.end_frame - range.start_frame + 1) / total * 100}%" data-seg-frame="${range.start_frame}" data-object="${esc(object.id)}" title="${esc(object.name)}：${statusNames[range.visibility]} #${range.start_frame + 1}–${range.end_frame + 1}${range.reviewed ? ' ✓' : ''}"></button>`).join('');
        const frames = object.keyframes.map(item => `<button class="seg-timeline-key ${item.visibility} ${item.reviewed ? 'reviewed' : ''}" style="left:${item.frame_index / total * 100}%;width:${Math.max(.4, 100 / total)}%;--track-color:${color}" data-seg-frame="${item.frame_index}" data-object="${esc(object.id)}" title="${esc(object.name)}：#${item.frame_index + 1} ${statusNames[item.visibility]}${item.reviewed ? ' ✓' : ''}"></button>`).join('');
        return `<div class="seg-timeline-label">${esc(object.name)}</div><div class="timeline-lane seg-timeline-lane ${object.id === trackId ? 'selected' : ''}"><div class="seg-lifetime" style="left:${object.start_frame / total * 100}%;width:${(object.end_frame - object.start_frame + 1) / total * 100}%"></div>${ranges}${frames}<div class="seg-playhead" style="left:${frame() / total * 100}%"></div></div>`;
      }).join('') : '<div class="timeline-empty">尚未创建分割对象。</div>');
    }
    function renderDetails() {
      if (!active) return;
      const object = track(); $('inspectorTitle').textContent = '对象与逐帧分割';
      if (!object) { $('inspectorBody').innerHTML = '<p class="hint">先配置对象类别，再创建对象。每个对象有独立的名称、稳定 ID 和出现范围。</p><p class="hint">只描出画面中可见的部分。被其他物体遮挡后分开的可见区域，可以使用多个轮廓。</p>'; return; }
      const current = frameState(object, frame()), explicit = current.visibility !== 'missing' && current.visibility !== 'inactive';
      const disable = busy() ? 'disabled' : '';
      $('inspectorBody').innerHTML = `<fieldset class="seg-detail-fields" ${disable}><label>对象名称<input id="segTrackName" value="${esc(object.name)}" maxlength="160"></label><label>对象类别<select id="segTrackLabel">${labels.map(value => `<option value="${esc(value.id)}" ${value.id === object.label_id ? 'selected' : ''}>${esc(value.name)}</option>`).join('')}</select></label><small class="seg-id">ID：${esc(object.id)}</small><div class="field-row"><label>起始帧<input id="segTrackStart" type="number" min="1" max="${frameCount()}" step="1" value="${object.start_frame + 1}"></label><label>结束帧<input id="segTrackEnd" type="number" min="1" max="${frameCount()}" step="1" value="${object.end_frame + 1}"></label></div><button id="segApplyLifetime" class="secondary full">应用对象范围</button>
        <div class="seg-detail-section"><h3>当前帧 #${frame() + 1}</h3><div class="seg-key-nav"><button id="segPreviousKey" class="secondary">← 上一标注帧</button><button id="segNextKey" class="secondary">下一标注帧 →</button></div><p class="seg-current-status">${statusNames[current.visibility]}${current.fromRange ? '（来自状态区间）' : ''}</p><label>可见状态<select id="segVisibility" ${current.visibility === 'inactive' ? 'disabled' : ''}><option value="visible" ${['visible', 'missing'].includes(current.visibility) ? 'selected' : ''}>可见：在画布上标注轮廓</option><option value="occluded" ${current.visibility === 'occluded' ? 'selected' : ''}>完全遮挡：没有可见像素</option><option value="outside" ${current.visibility === 'outside' ? 'selected' : ''}>离开画面</option></select></label><label class="check"><input id="segFrameReviewed" type="checkbox" ${current.reviewed ? 'checked' : ''} ${!explicit || !$('player').paused || !canvasStatus.ready ? 'disabled' : ''}>当前帧已人工审核</label><button id="segCopyPrevious" class="secondary full">复制上一分割到当前帧</button><button id="segDeleteCurrent" class="text-button danger" ${!explicit ? 'disabled' : ''}>删除当前帧标注</button></div>
        <details class="seg-detail-section"><summary>遮挡 / 离开区间</summary><div class="field-row"><label>起始帧<input id="segStateStart" type="number" min="${object.start_frame + 1}" max="${object.end_frame + 1}" step="1" value="${Math.max(object.start_frame, Math.min(object.end_frame, frame())) + 1}"></label><label>结束帧<input id="segStateEnd" type="number" min="${object.start_frame + 1}" max="${object.end_frame + 1}" step="1" value="${Math.max(object.start_frame, Math.min(object.end_frame, frame())) + 1}"></label></div><label>区间状态<select id="segRangeVisibility"><option value="occluded">完全遮挡</option><option value="outside">离开画面</option></select></label><label class="check"><input id="segStateReviewed" type="checkbox">已逐帧检查此区间</label><button id="segApplyStateRange" class="secondary full">设置状态区间</button><small class="hint">该区间中的轮廓会移除。再次可见时，可在原对象下继续绘制。</small></details>
        <details class="seg-detail-section" ${pendingPreview ? 'open' : ''}><summary>轮廓调整</summary><label>调整方式<select id="segRefineOperation"><option value="smooth">平滑边界</option><option value="shrink">向内收缩</option><option value="expand">向外扩张</option><option value="snap">贴合图像边缘</option></select></label><div class="field-row"><label>半径（px）<input id="segRefineRadius" type="number" min="1" max="64" step="1" value="3"></label><label>吸附距离（px）<input id="segSnapDistance" type="number" min="1" max="32" step="1" value="10"></label></div><label class="check"><input id="segProtectHoles" type="checkbox" checked>保留孔洞</label><button id="segPreviewRefine" class="secondary full" ${!current.geometry || previewBusy ? 'disabled' : ''}>预览调整</button>${pendingPreview ? '<div class="seg-refine-preview"><span>预览尚未应用</span><button id="segApplyRefine" class="primary">应用</button><button id="segCancelRefine" class="secondary">取消</button></div>' : ''}</details>
        <details class="seg-detail-section"><summary>轨迹拆分 / 合并 / 删除</summary><button id="segSplitTrack" class="secondary full">从当前帧拆为新对象</button><label>合并另一个对象到当前对象<select id="segMergeOther"><option value="">选择同类对象</option>${document.tracks.filter(other => other.id !== object.id && other.label_id === object.label_id).map(other => `<option value="${esc(other.id)}">${esc(other.name)}</option>`).join('')}</select></label><button id="segMergeTrack" class="secondary full">合并对象与事件关联</button><div class="field-row"><label>删除起始帧<input id="segDeleteStart" type="number" min="1" max="${frameCount()}" step="1" value="${frame() + 1}"></label><label>删除结束帧<input id="segDeleteEnd" type="number" min="1" max="${frameCount()}" step="1" value="${frame() + 1}"></label></div><button id="segDeleteRange" class="secondary full danger">删除区间中的标注</button><button id="segDeleteTrack" class="text-button danger">删除整个对象</button></details>
        <div class="seg-detail-section"><button id="segReviewTrack" class="secondary full">${object.reviewed ? '✓ 整个对象已审核' : '确认整个对象审核完成'}</button><p class="hint">每个活动帧都需明确标注并审核。只完成关键帧不能算作整条轨迹已审核。</p></div></fieldset>`;
      bindDetails(object);
    }
    function bindDetails(object) {
      const action = (id, callback) => $(id)?.addEventListener('click', () => ctx.run(callback));
      $('segTrackName').addEventListener('input', event => { object.name = event.target.value; changed(object, {renderInspector: false}); });
      $('segTrackLabel').addEventListener('change', event => { object.label_id = event.target.value; object.keyframes.forEach(item => item.reviewed = false); object.visibility_ranges.forEach(item => item.reviewed = false); changed(object); syncFrame(true); });
      action('segApplyLifetime', () => {
        if (!beforeContextChange()) return;
        const start = Number($('segTrackStart').value) - 1, end = Number($('segTrackEnd').value) - 1;
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= frameCount()) throw new Error('对象范围需要有效的整数帧号，且结束帧不早于起始帧。');
        if (object.keyframes.some(item => item.frame_index < start || item.frame_index > end) || object.visibility_ranges.some(item => item.start_frame < start || item.end_frame > end)) throw new Error('新范围之外仍有标注。请先明确删除对应区间，再缩小对象范围。');
        object.start_frame = start; object.end_frame = end; changed(object); syncFrame(true);
      });
      for (const [id, direction] of [['segPreviousKey', -1], ['segNextKey', 1]]) action(id, () => {
        const positions = [...object.keyframes.map(item => item.frame_index), ...object.visibility_ranges.flatMap(item => [item.start_frame, item.end_frame])].filter(value => direction < 0 ? value < frame() : value > frame()).sort((a, b) => direction < 0 ? b - a : a - b);
        if (!positions.length) return ctx.notify(direction < 0 ? '前面没有已标注帧。' : '后面没有已标注帧。'); ctx.setFrame(positions[0]);
      });
      $('segVisibility').addEventListener('change', event => {
        const value = event.target.value;
        if (!beforeContextChange()) { renderDetails(); return; }
        const current = frameState(object, frame());
        if (value !== 'visible' && current.geometry && !confirm('标记为完全遮挡或离开画面，会移除当前帧的轮廓。继续？')) { renderDetails(); return; }
        if (value === 'visible') { if (current.visibility !== 'visible') { clearFrameSpan(object, frame(), frame()); changed(object); syncFrame(true); } ctx.notify('请在当前帧画出对象的可见轮廓。完成轮廓后会记录为可见。'); }
        else { clearFrameSpan(object, frame(), frame()); object.keyframes.push({frame_index: frame(), visibility: value, geometry: null, reviewed: false}); object.keyframes.sort((a, b) => a.frame_index - b.frame_index); changed(object); syncFrame(true); }
      });
      $('segFrameReviewed').addEventListener('change', event => {
        const value = event.target.checked, current = frameState(object, frame()); if (!['visible', 'occluded', 'outside'].includes(current.visibility)) return;
        if (canvas?.isDraftDirty() || pendingPreview) { event.target.checked = !value; ctx.notify('请先完成轮廓或确认调整预览，再审核当前帧。', true); return; }
        if (current.fromRange) { clearFrameSpan(object, frame(), frame()); object.keyframes.push({frame_index: frame(), visibility: current.visibility, geometry: null, reviewed: value}); }
        else current.reviewed = value;
        if (!value) { document.reviewed_ranges = []; object.reviewed = false; }
        changed(object, {keepReview: value});
      });
      action('segCopyPrevious', () => {
        if (!beforeContextChange()) return;
        if (frame() < object.start_frame || frame() > object.end_frame) throw new Error('当前帧不在对象范围内。');
        const previous = object.keyframes.filter(item => item.frame_index < frame() && item.visibility === 'visible' && item.geometry).sort((a, b) => b.frame_index - a.frame_index)[0];
        if (!previous) throw new Error('前面没有可复制的分割帧。');
        if (frameState(object, frame()).geometry && !confirm('用上一分割帧的轮廓替换当前帧？')) return;
        clearFrameSpan(object, frame(), frame()); object.keyframes.push({frame_index: frame(), visibility: 'visible', geometry: clone(previous.geometry), reviewed: false}); changed(object); syncFrame(true); ctx.notify('轮廓已复制，请按当前画面调整边界并重新审核。');
      });
      action('segDeleteCurrent', () => { if (!beforeContextChange() || !confirm(`删除对象“${object.name}”第 ${frame() + 1} 帧的标注？该帧将变为未标注。`)) return; clearFrameSpan(object, frame(), frame()); changed(object); syncFrame(true); });
      action('segApplyStateRange', () => {
        if (!beforeContextChange()) return;
        const start = Number($('segStateStart').value) - 1, end = Number($('segStateEnd').value) - 1, visibility = $('segRangeVisibility').value, reviewed = $('segStateReviewed').checked;
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < object.start_frame || end > object.end_frame || start > end) throw new Error('状态区间必须位于该对象范围内。');
        if (object.keyframes.some(item => item.frame_index >= start && item.frame_index <= end && item.geometry) && !confirm('这个区间中已有可见轮廓。设置遮挡／离开状态会移除它们。继续？')) return;
        clearFrameSpan(object, start, end); object.visibility_ranges.push({start_frame: start, end_frame: end, visibility, reviewed}); object.visibility_ranges.sort((a, b) => a.start_frame - b.start_frame); changed(object); syncFrame(true);
      });
      action('segPreviewRefine', previewGeometry); action('segApplyRefine', applyPreview); action('segCancelRefine', cancelPreview);
      action('segSplitTrack', async () => {
        if (frame() <= object.start_frame || frame() > object.end_frame) throw new Error('拆分位置需在对象起始帧之后，且位于对象范围内。');
        if (!confirm(`从第 ${frame() + 1} 帧拆出一个新对象？现有分割将按时间分开，相关事件关联由系统同步调整。`)) return;
        await operation({operation: 'split', track_id: object.id, frame_index: frame()});
      });
      action('segMergeTrack', async () => {
        const other = $('segMergeOther').value; if (!other) throw new Error('请选择要合并的同类对象。');
        if (!confirm('将所选对象合并到当前对象，保留当前对象 ID，并更新事件关联。存在冲突的帧会阻止合并。继续？')) return;
        await operation({operation: 'merge', track_id: object.id, other_track_id: other});
      });
      action('segDeleteRange', async () => {
        const start = Number($('segDeleteStart').value) - 1, end = Number($('segDeleteEnd').value) - 1;
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < object.start_frame || end > object.end_frame || end < start) throw new Error('删除范围必须是对象范围内的有效帧区间。');
        if (!confirm(`删除对象“${object.name}”第 ${start + 1}–${end + 1} 帧的全部轮廓与可见状态？对象 ID 和活动范围会保留，区间变为未标注。`)) return;
        await operation({operation: 'delete_range', track_id: object.id, start_frame: start, end_frame: end});
      });
      action('segDeleteTrack', async () => {
        if (state().annotations.events.some(event => (event.track_ids || []).includes(object.id))) throw new Error('这个对象仍被事件关联。请先到事件面板取消对应关联并保存，再删除对象。');
        if (!beforeContextChange() || !confirm(`删除整个对象“${object.name}”及其全部帧标注？此操作在保存草稿后生效。`)) return;
        if (!await ctx.saveAll()) return;
        document.tracks = document.tracks.filter(item => item.id !== object.id); trackId = document.tracks[0]?.id || null; changed(); canvasKey = null; syncFrame();
      });
      action('segReviewTrack', async () => {
        const missing = gap(object, object.start_frame, object.end_frame);
        if (missing != null) throw new Error(`第 ${missing + 1} 帧尚未明确标注并审核。请先补齐分割或遮挡／离开状态。`);
        if (!confirm(`确认对象“${object.name}”的全部活动帧已经审核完成？`)) return;
        object.reviewed = true; changed(object, {keepReview: true}); await ctx.saveAll();
      });
    }
    async function previewGeometry() {
      if (previewBusy || !track() || !canvas?.canSave()) throw new Error('请先完成当前轮廓并等待精确帧加载。');
      if (pendingPreview) cancelPreview();
      const geometry = frameState(track(), frame()).geometry; if (!geometry) throw new Error('当前帧没有可调整的分割。');
      const original = clone(geometry), requestKey = key(), token = generation;
      const payload = {frame_index: frame(), geometry: original, operation: $('segRefineOperation').value, radius: Number($('segRefineRadius').value), snap_distance: Number($('segSnapDistance').value), protect_holes: $('segProtectHoles').checked};
      previewBusy = true; canvas.setEnabled(false); renderCanvasTools(); ctx.updateDirty();
      try {
        const result = await ctx.post(`${endpoint()}/preview`, payload);
        if (generation !== token || key() !== requestKey || !active) return;
        pendingPreview = {key: requestKey, original, geometry: result.geometry}; canvas.setGeometry(result.geometry, {silent: true}); renderDetails(); ctx.notify('轮廓调整已生成预览。确认效果后点击“应用”。');
      } finally { previewBusy = false; canvas.setEnabled(canEditCanvas()); renderDetails(); renderCanvasTools(); ctx.updateDirty(); }
    }
    function applyPreview() {
      if (!pendingPreview || pendingPreview.key !== key()) return;
      const value = pendingPreview; pendingPreview = null; canvas.setGeometry(value.original, {silent: true}); canvas.setGeometry(value.geometry, {silent: false}); canvas.setEnabled(canEditCanvas()); renderDetails(); renderCanvasTools(); ctx.updateDirty();
    }
    function cancelPreview() {
      if (pendingPreview && pendingPreview.key === key()) canvas?.setGeometry(pendingPreview.original, {silent: true});
      pendingPreview = null; canvas?.setEnabled(canEditCanvas()); renderDetails(); renderCanvasTools(); ctx.updateDirty();
    }
    async function operation(payload) {
      if (!beforeContextChange() || !await ctx.saveAll()) return;
      const token = generation; saving = true; renderDetails(); canvas?.setEnabled(false); ctx.updateDirty();
      try {
        const result = await ctx.post(`${endpoint()}/operations`, {...payload, revision: document.revision, annotations_revision: state().annotations.revision});
        if (token !== generation) return;
        document = result.segmentation; dirty = false; edits++; ctx.replaceAnnotations(result.annotations);
        if (!track()) trackId = document.tracks[0]?.id || null; canvasKey = null; renderList(); renderTimeline(); renderDetails(); await syncFrame(); ctx.notify('对象轨迹已更新并保存。');
      } finally { saving = false; renderDetails(); canvas?.setEnabled(canEditCanvas()); ctx.updateDirty(); }
    }
    async function save() {
      if (saving || previewBusy) return false;
      if (pendingPreview || canvas?.isDraftDirty() || canvasStatus.ready && !canvas?.canSave()) { ctx.notify('分割还有未完成的轮廓或调整预览，请完成或取消后保存。', true); return false; }
      if (!dirty) return true;
      for (const object of document.tracks) {
        if (!object.name.trim() || !labels.some(value => value.id === object.label_id)) { ctx.notify('请填写对象名称，并选择有效的对象类别。', true); return false; }
      }
      const token = generation, counter = edits, snapshot = clone(document); saving = true; ctx.updateDirty();
      try {
        const saved = await ctx.api(endpoint(), {method: 'PUT', body: JSON.stringify(snapshot)});
        if (generation !== token) return false;
        if (edits === counter) { document = saved; dirty = false; renderDetails(); } else document.revision = saved.revision;
        renderList(); renderTimeline(); return edits === counter;
      } catch (error) { ctx.notify(`分割保存失败：${error.message}\n修改仍保留在页面中。`, true); return false; }
      finally { saving = false; renderDetails(); renderCanvasTools(); ctx.updateDirty(); }
    }
    function bind() {
      $('segAddTrack').addEventListener('click', () => {
        if (!beforeContextChange()) return;
        if (!labels.length) { ctx.notify('请先新增或导入至少一个对象类别。'); ctx.openLabels('objects'); return; }
        $('segNewLabel').innerHTML = labels.map(value => `<option value="${esc(value.id)}">${esc(value.name)}</option>`).join('');
        $('segNewName').value = `${labels[0].name} ${String(document.tracks.length + 1).padStart(2, '0')}`;
        $('segNewStart').value = frame() + 1; $('segNewEnd').value = frameCount(); $('segNewStart').max = frameCount(); $('segNewEnd').max = frameCount(); $('segNewError').textContent = ''; $('segNewTrackDialog').showModal();
      });
      $('segNewTrackForm').addEventListener('submit', event => {
        event.preventDefault();
        const start = Number($('segNewStart').value) - 1, end = Number($('segNewEnd').value) - 1, name = $('segNewName').value.trim();
        if (!name || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= frameCount()) { $('segNewError').textContent = '请填写名称和有效的对象帧范围。'; return; }
        const object = {id: ctx.uid(), name, label_id: $('segNewLabel').value, start_frame: start, end_frame: end, keyframes: [], visibility_ranges: [], reviewed: false};
        document.tracks.push(object); trackId = object.id; $('segNewTrackDialog').close(); changed(object); canvasKey = null; if (frame() < start || frame() > end) ctx.setFrame(start); else syncFrame();
      });
      $('segManageLabels').addEventListener('click', () => ctx.openLabels('objects'));
      $('segTrackList').addEventListener('click', event => { const button = event.target.closest('[data-seg-track]'); if (button) selectTrack(button.dataset.segTrack); });
      $('timeline').addEventListener('click', event => {
        if (!active) return;
        const button = event.target.closest('[data-seg-frame]'); if (button) { event.preventDefault(); event.stopImmediatePropagation(); if (!beforeContextChange()) return; trackId = button.dataset.object; canvasKey = null; ctx.setFrame(Number(button.dataset.segFrame)); renderList(); renderDetails(); renderTimeline(); return; }
        const lane = event.target.closest('.seg-timeline-lane'); if (lane) { event.preventDefault(); event.stopImmediatePropagation(); const rect = lane.getBoundingClientRect(); ctx.setFrame(Math.max(0, Math.min(frameCount() - 1, Math.floor((event.clientX - rect.left) / rect.width * frameCount())))); }
      }, true);
      documentQuery('[data-seg-tool]').forEach(button => button.addEventListener('click', () => { if (pendingPreview) return ctx.notify('请先应用或取消调整预览。'); currentTool = button.dataset.segTool; canvas?.setTool(currentTool); renderCanvasTools(); }));
      for (const [id, method] of [['segFinish', 'finishPolygon'], ['segCancelDraft', 'cancelDraft'], ['segAppendContour', 'appendContour'], ['segDeleteVertex', 'deleteVertex'], ['segDeleteContour', 'deleteContour'], ['segUndo', 'undo'], ['segRedo', 'redo'], ['segFit', 'fit']]) $(id).addEventListener('click', () => { if (pendingPreview && id !== 'segFit') return ctx.notify('请先应用或取消调整预览。'); canvas?.[method](); });
      $('segRadius').addEventListener('change', event => canvas?.setRadius(Math.max(1, Math.min(256, Number(event.target.value) || 12))));
      $('segReviewRange').addEventListener('click', () => ctx.run(async () => {
        const start = Number($('segReviewStart').value) - 1, end = Number($('segReviewEnd').value) - 1;
        if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= frameCount()) throw new Error('请填写视频范围内的有效整数帧区间。');
        for (const object of document.tracks) { const missing = gap(object, start, end); if (missing != null) throw new Error(`对象“${object.name}”第 ${missing + 1} 帧尚未明确标注并审核，不能确认此区间。`); }
        if (!confirm(`确认已检查第 ${start + 1}–${end + 1} 帧中的全部对象？没有活动对象的帧将记为已确认无对象。`)) return;
        const ranges = [...document.reviewed_ranges, {start_frame: start, end_frame: end}].sort((a, b) => a.start_frame - b.start_frame), merged = [];
        for (const range of ranges) { const last = merged.at(-1); if (last && range.start_frame <= last.end_frame + 1) last.end_frame = Math.max(last.end_frame, range.end_frame); else merged.push({...range}); }
        document.reviewed_ranges = merged; changed(null, {keepReview: true}); await ctx.saveAll();
      }));
      $('segReviewedRanges').addEventListener('click', event => { const button = event.target.closest('[data-remove-review]'); if (button) { document.reviewed_ranges.splice(Number(button.dataset.removeReview), 1); changed(null, {keepReview: true}); } });
    }
    bind();
    return {
      getTracks: () => document.tracks,
      getLabels: () => labels,
      setLabels(value) { labels = value || []; renderList(); renderDetails(); if (active) syncFrame(true); },
      getDocument: () => document,
      isDirty: () => dirty || Boolean(pendingPreview) || Boolean(canvas?.isDraftDirty()),
      isSaving: () => saving || previewBusy,
      isDraftDirty: () => Boolean(canvas?.isDraftDirty()),
      reset() { generation++; document = emptyDocument(); trackId = null; dirty = false; edits = 0; pendingPreview = null; canvasKey = null; canvas?.cancelDraft(); canvas?.setEnabled(false); $('segmentationCanvas').hidden = true; },
      setDocument(value) { generation++; document = value || emptyDocument(); document.tracks.forEach(object => { object.keyframes ||= []; object.visibility_ranges ||= []; }); document.reviewed_ranges ||= []; trackId = document.tracks[0]?.id || null; dirty = false; edits = 0; canvasKey = null; pendingPreview = null; $('segReviewStart').value = 1; $('segReviewEnd').value = frameCount(); $('segReviewStart').max = frameCount(); $('segReviewEnd').max = frameCount(); renderList(); renderDetails(); },
      setActive(value) { active = value; $('segmentationPanel').hidden = !value; $('segCanvasTools').hidden = !value; if (!value) { $('segmentationCanvas').hidden = true; canvas?.setEnabled(false); } else { renderList(); renderTimeline(); renderDetails(); syncFrame(); } },
      onFrame() { if (active) { renderList(); renderTimeline(); if ($('player').paused) syncFrame(); else { $('segmentationCanvas').hidden = true; canvas?.setEnabled(false); renderCanvasTools(); } } },
      onSeeking() { if (active) { $('segmentationCanvas').hidden = true; canvas?.setEnabled(false); } },
      onPlayback() { if (active) { if (!$('player').paused) { $('segmentationCanvas').hidden = true; canvas?.setEnabled(false); } else syncFrame(); renderCanvasTools(); } },
      beforeContextChange,
      beforeFrame(next) { return !active || next === frame() || beforeContextChange(); },
      renderDetails, renderTimeline, save,
      frameHasGeometry(index) { return document.tracks.some(object => frameState(object, index).visibility === 'visible' && frameState(object, index).geometry); },
      clearDirty() { dirty = false; pendingPreview = null; canvas?.cancelDraft(); }
    };
  }};
})();
