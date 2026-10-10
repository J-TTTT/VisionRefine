/* Dataset transactions are shared by new-project and append workflows. */
let importSession = null;
let importSourceSerial = 0;

function invalidateImport() {
  if (!importSession) return;
  importSession.preview = null;
  $("commitDatasetImport").disabled = true;
  $("datasetImportStatus").textContent = "配置已更改，请检查并重新预览。";
}

function addImportSource(seed = {}) {
  const row = document.createElement("section");
  row.className = "import-source";
  const id = seed.id || `${importSession.projectId ? importSession.sourcePrefix : "source"}${++importSourceSerial}`;
  row.innerHTML = `<div class="source-heading"><h3>数据来源</h3><button type="button" class="secondary remove-source">移除此来源</button></div>
    <div class="source-grid">
      <label>来源标识<input data-field="id" value="${escapeHtml(id)}" placeholder="例如 train2026，仅英文、数字、-、_"></label>
      <label>输入格式<select data-field="format">${datasetFormats.filter(f => f.can_import && f.tasks.includes(importSession.task)).map(f => `<option value="${escapeHtml(f.id)}">${escapeHtml(f.title)}</option>`).join("")}</select></label>
      <label>图片根目录<input data-field="root" placeholder="/path/to/dataset"></label>
      <label class="source-annotation">标注文件 / 目录 / 原生 ZIP<input data-field="annotation_path"></label>
      <label>默认划分<select data-field="split"><option value="unspecified">未指定</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select></label>
      <label class="source-labels">图片模式类别 / 空标注备用类别<input data-field="labels" placeholder="例如 person, car"></label>
    </div><label class="native-trust" hidden><input type="checkbox" data-field="trust_reviewed"> 信任原生包并恢复其审核状态（默认不信任）</label><p class="hint source-hint"></p><div class="source-mappings"></div>`;
  for (const key of ["format", "root", "annotation_path", "split"]) {
    if (seed[key]) row.querySelector(`[data-field="${key}"]`).value = seed[key];
  }
  row.querySelector('[data-field="labels"]').value = (seed.labels || []).join(", ");
  const update = () => {
    const format = row.querySelector('[data-field="format"]').value;
    const spec = datasetFormats.find(f => f.id === format)?.input || {};
    row.querySelector(".source-annotation").hidden = format === "images";
    row.querySelector(".source-labels").hidden = !spec.labels;
    row.querySelector('[data-field="split"]').disabled = !!spec.split_from_source;
    row.querySelector('[data-field="annotation_path"]').placeholder = spec.placeholder || "";
    row.querySelector(".source-hint").textContent = spec.hint || "";
    row.querySelector(".native-trust").hidden = format !== "visionrefine";
  };
  row.querySelector('[data-field="format"]').onchange = update;
  row.addEventListener("input", event => {
    if (event.target.matches("[data-field]")) row.querySelector(".source-mappings").replaceChildren();
    invalidateImport();
  });
  row.querySelector(".remove-source").onclick = () => { row.remove(); invalidateImport(); };
  $("importSources").append(row);
  update();
  invalidateImport();
}

function openDatasetImport(append) {
  if (editor.dirty && !$("annotationWorkspace").hidden) {
    alert("请先保存当前图像的人工修改，再导入数据。");
    return false;
  }
  const form = $("projectForm").elements;
  if (!append && !form.name.value.trim()) {
    $("formError").textContent = "请先填写项目名称。";
    form.name.focus();
    return false;
  }
  importSession = {projectId: append ? current.id : null,
    name: append ? current.name : form.name.value.trim(),
    task: append ? current.task : form.task.value,
    maxSide: append ? current.model_max_side : Number(form.model_max_side.value), preview: null, busy: false,
    sourcePrefix: `batch${Date.now()}_`};
  importSourceSerial = 0;
  $("importSources").replaceChildren();
  $("datasetImportPreview").hidden = true;
  $("importConflicts").replaceChildren();
  $("duplicatePolicy").value = "keep_existing";
  $("datasetImportTitle").textContent = append ? `追加到：${current.name}` : `导入到新项目：${importSession.name}`;
  $("commitDatasetImport").textContent = append ? "确认追加" : "确认创建并导入";
  addImportSource(append ? {labels: current.labels} : {
    format: form.dataset_format.value, root: form.dataset_path.value,
    annotation_path: form.annotation_path.value, split: form.split.value, labels: createLabels
  });
  $("datasetImportStatus").textContent = "先预览来源、类别和冲突；确认后才会写入项目。";
  $("datasetImportDialog").showModal();
  return true;
}

function importPayload() {
  const sources = [...$("importSources").children].map(row => {
    const value = key => row.querySelector(`[data-field="${key}"]`).value.trim();
    const mapping = {};
    row.querySelectorAll("[data-source-category]").forEach(input => { mapping[input.dataset.sourceCategory] = input.value.trim(); });
    return {id: value("id"), root: value("root"), format: value("format"),
      annotation_path: value("format") === "images" ? null : value("annotation_path") || null,
      split: value("split"), labels: value("labels").split(/[,，\n]/).map(s => s.trim()).filter(Boolean), category_mapping: mapping,
      trust_reviewed: value("format") === "visionrefine" && row.querySelector('[data-field="trust_reviewed"]').checked};
  });
  const resolutions = {};
  $("importConflicts").querySelectorAll("[data-conflict]").forEach(select => { resolutions[select.dataset.conflict] = select.value; });
  return {project_id: importSession.projectId, name: importSession.name, task: importSession.task,
    model_max_side: importSession.maxSide, sources, duplicate_policy: $("duplicatePolicy").value, conflict_resolutions: resolutions};
}

function importBusy(busy) {
  importSession.busy = busy;
  $("datasetImportInputs").disabled = busy;
  $("previewDatasetImport").disabled = busy;
  $("closeDatasetImport").disabled = busy;
  $("commitDatasetImport").disabled = busy || !importSession.preview?.commit_allowed;
}

function renderImportPreview(preview) {
  const s = preview.summary;
  $("datasetImportPreview").hidden = false;
  $("importPreviewSummary").textContent = `新增 ${s.added_images} 张 · 更新粗标注 ${s.updated_coarse} 张 · 跳过重复 ${s.skipped_duplicates} 张。合并后 ${s.image_count} 张 / ${s.object_count} ${importSession.task==="instance_segmentation"?"个实例":"个框"} / ${s.category_count} 类。划分：${Object.entries(s.splits).map(([k,v]) => `${k} ${v}`).join(" · ")}`;
  for (const source of preview.sources) {
    const row = [...$("importSources").children].find(r => r.querySelector('[data-field="id"]').value.trim() === source.id);
    const area = row.querySelector(".source-mappings");
    area.innerHTML = "<h4>类别映射：原类别 → 项目类别</h4>";
    for (const mapping of source.mappings) {
      const label = document.createElement("label");
      label.textContent = `${mapping.source_name} → `;
      const input = document.createElement("input");
      input.dataset.sourceCategory = mapping.source_name;
      input.value = mapping.target_name;
      input.setAttribute("aria-label", `${source.id}: ${mapping.source_name} 目标类别`);
      label.append(input);
      area.append(label);
    }
  }
  $("importConflicts").replaceChildren();
  for (const conflict of preview.conflicts) {
    const label = document.createElement("label");
    label.textContent = `${conflict.incoming} → 已有 ${conflict.existing}（保留 ${conflict.existing_split}；传入 ${conflict.incoming_split}；${importSession.task==="instance_segmentation"?"实例":"框"} ${conflict.existing_objects} → ${conflict.incoming_objects}）`;
    const select = document.createElement("select");
    select.dataset.conflict = conflict.incoming;
    select.innerHTML = '<option value="keep_existing">保留原数据</option><option value="update_coarse">只更新导入粗标注</option><option value="error">阻止提交</option>';
    select.value = conflict.action;
    select.onchange = invalidateImport;
    label.append(select);
    $("importConflicts").append(label);
  }
  $("importSourceReports").textContent = preview.sources.map(source => {
    const r = source.report;
    return `${source.id}：读取 ${r.image_count} 张 / ${r.object_count} ${importSession.task==="instance_segmentation"?"个实例":"个框"}，跳过 ${r.skipped_images} 张 / ${r.skipped_objects} ${importSession.task==="instance_segmentation"?"个实例":"个框"}\n` + r.issues.map(i => `[${i.severity}] ${i.location}: ${i.message}`).join("\n");
  }).join("\n\n");
  const issues = preview.sources.reduce((n, s) => n + s.report.issues.length, 0);
  $("datasetImportStatus").textContent = `${issues ? `有 ${issues} 项警告 / 错误，请展开来源检查报告。` : "来源检查完成。"}${preview.commit_allowed ? "核对类别映射和冲突后即可确认。" : "有阻止提交的冲突，请调整策略并重新预览。"}`;
}

async function previewDatasetImport() {
  importSession.preview = null;
  importBusy(true);
  $("datasetImportStatus").textContent = "正在读取所选格式中的图像和标注，并检查图片是否存在…";
  try {
    const preview = await api("/api/dataset-imports/preview", {method: "POST", body: JSON.stringify(importPayload())});
    importSession.preview = preview;
    renderImportPreview(preview);
  } catch (error) {
    $("datasetImportStatus").textContent = error.message;
  } finally { importBusy(false); }
}

$("commitDatasetImport").onclick = async () => {
  const preview = importSession.preview;
  if (!preview?.commit_allowed) return;
  importBusy(true);
  $("datasetImportStatus").textContent = "正在确认图片仍然存在并提交…";
  let committed = false;
  try {
    const project = await api(`/api/dataset-imports/${preview.preview_id}/commit`, {method: "POST"});
    committed = true;
    $("datasetImportStatus").textContent = "导入已提交，正在分析图像…";
    await api(`/api/projects/${project.id}/analyze`, {method: "POST"});
    editor.dirty = false;
    await loadProjects();
    await showProject(project.id);
    $("datasetImportDialog").close();
  } catch (error) {
    $("datasetImportStatus").textContent = (committed ? "导入已保存，后续分析失败，可关闭后重新分析：" : "导入未提交，请重新预览：") + error.message;
    importSession.preview = null;
    if (committed) await loadProjects();
  } finally { importBusy(false); }
};
$("previewDatasetImport").onclick = previewDatasetImport;
$("newDatasetImport").onclick = () => openDatasetImport(false);
$("appendDataset").onclick = () => openDatasetImport(true);
$("addImportSource").onclick = () => addImportSource({labels: importSession.projectId ? current.labels : createLabels});
$("duplicatePolicy").onchange = () => { $("importConflicts").replaceChildren(); invalidateImport(); };
$("closeDatasetImport").onclick = () => $("datasetImportDialog").close();
$("datasetImportDialog").addEventListener("cancel", event => { if (importSession?.busy) event.preventDefault(); });
$("showDatasetHistory").onclick = async () => {
  const target = $("datasetHistory"), id = current.id;
  if (!target.hidden) { target.hidden = true; return; }
  target.hidden = false;
  target.textContent = "读取历史…";
  try {
    const history = await api(`/api/projects/${id}/dataset/history`);
    if (current?.id !== id) return;
    target.innerHTML = history.map(row => `<details><summary>${escapeHtml(row.imported_at || "历史导入")} · ${row.operation.kind === "append" ? "追加" : "创建 / 导入"} · ${row.image_count} 张</summary><pre>${escapeHtml(JSON.stringify(row, null, 2))}</pre></details>`).join("");
  } catch (error) { target.textContent = error.message; }
};
