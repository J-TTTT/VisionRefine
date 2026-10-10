/* Bounded image lists shared by the project table and annotation workspace. */
const imageBrowsers = {};

function confirmImageChange() {
  return !(editor.dirty || (Segmentation.active() && Segmentation.dirty())) ||
    confirm("当前图像有未保存的修改，确定切换图像并放弃这些修改吗？");
}

function resetImageBrowsers() {
  cancelEditorLoad();
  editor.image = null; editor.meta = null; editor.objects = []; editor.images = []; editor.dirty = false;
  editor.annotationStatus = "unreviewed";
  Segmentation.reset();
  closeBoxLabelPanel();
  $("workspaceCurrentImage").textContent = "";
  $("workspaceStatus").textContent = "";
  const canvas = $("detailCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  for (const [name, browser] of Object.entries(imageBrowsers)) {
    browser.controller?.abort();
    browser.sequence++;
    browser.offset = 0; browser.total = 0; browser.query = "";
    browser.items = []; browser.loading = false;
    $(name + "ImageQuery").value = "";
    $(name + "ImageStatus").textContent = "";
    updateImageNavigation(name);
  }
  $("imageRows").replaceChildren();
  $("workspaceImage").replaceChildren();
}

function updateImageNavigation(name) {
  const browser = imageBrowsers[name];
  const pages = Math.max(1, Math.ceil(browser.total / browser.limit));
  $(name + "ImagePrev").disabled = browser.loading || browser.offset === 0;
  $(name + "ImageNext").disabled = browser.loading || browser.offset + browser.limit >= browser.total;
  const page = $(name + "ImagePage");
  page.value = Math.floor(browser.offset / browser.limit) + 1;
  page.max = pages;
  $(name + "ImageCount").textContent = `共 ${browser.total.toLocaleString()} 张 · ${pages} 页`;
}

async function loadImagePage(name, offset = 0, query = imageBrowsers[name].query) {
  if (!current) return;
  const workspace = name === "workspace", browser = imageBrowsers[name];
  if (workspace && !confirmImageChange()) {
    $(name + "ImageQuery").value = browser.query;
    updateImageNavigation(name);
    return;
  }
  browser.controller?.abort();
  const controller = new AbortController(), sequence = ++browser.sequence, projectId = current.id;
  browser.controller = controller; browser.loading = true;
  const active = () => sequence === browser.sequence && current?.id === projectId;
  const status = $(name + "ImageStatus");
  status.textContent = "正在加载…";
  updateImageNavigation(name);
  if (workspace) { cancelEditorLoad(); setEditorLoading(true); }
  try {
    const params = new URLSearchParams({offset, limit: browser.limit, q: query});
    const page = await api(`/api/projects/${projectId}/images?${params}`, {signal: controller.signal});
    if (!active()) return;
    if (workspace && page.items.length) {
      const loaded = await loadEditorImage(page.items[0].path, page.items[0]);
      if (!loaded || !active()) return;
    }
    Object.assign(browser, {offset: page.offset, total: page.total, items: page.items, query});
    if (workspace) {
      editor.images = page.items;
      $("workspaceImage").innerHTML = page.items.map(row =>
        `<option value="${escapeHtml(row.path)}">${escapeHtml(row.path)}</option>`).join("");
      $("workspaceImage").disabled = !page.items.length;
    } else {
      $("imageRows").innerHTML = page.items.map(row => `
        <tr><td title="${escapeHtml(row.path)}">${escapeHtml(row.path)}</td><td>${row.width} × ${row.height}</td><td>${row.megapixels}</td><td><span class="strategy">${routeNames[row.route.strategy] || row.route.strategy}${row.route.estimated_tiles > 1 ? ` · ${row.route.estimated_tiles}片` : ""}</span></td></tr>
      `).join("");
    }
    status.textContent = page.items.length ? "" : (workspace && editor.image ? "没有匹配图像，画布保留当前图像。" : "没有匹配图像。");
  } catch (error) {
    if (active() && error.name !== "AbortError") status.textContent = `加载失败：${error.message}。可点击搜索或跳转重试。`;
  } finally {
    if (active()) {
      browser.loading = false;
      updateImageNavigation(name);
      if (workspace) setEditorLoading(false);
    }
  }
}

function initImageBrowsers() {
  for (const name of ["project", "workspace"]) {
    const browser = imageBrowsers[name] = {offset: 0, total: 0, limit: 50, query: "", items: [], sequence: 0};
    $(name + "ImageSearch").onsubmit = event => {
      event.preventDefault();
      loadImagePage(name, 0, $(name + "ImageQuery").value.trim());
    };
    $(name + "ImageJump").onsubmit = event => {
      event.preventDefault();
      loadImagePage(name, (Number($(name + "ImagePage").value) - 1) * browser.limit);
    };
    $(name + "ImagePrev").onclick = () => loadImagePage(name, Math.max(0, browser.offset - browser.limit));
    $(name + "ImageNext").onclick = () => loadImagePage(name, browser.offset + browser.limit);
    updateImageNavigation(name);
  }
}
