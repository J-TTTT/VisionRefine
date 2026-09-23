/* Instance editor: original-pixel contours and sparse 128px binary mask tiles. */
const Segmentation = (() => {
  const SIZE = 128, PIXELS = SIZE * SIZE, MAX_TILES = 4096, MAX_RUNS = 1000000;
  const state = {draft: null, vertex: null, undo: [], redo: [], before: null, cursor: null,
    multi: new Set(), cutLine: null, preview: null, busy: false};
  let tileCache = new WeakMap();
  const active = () => current?.task === "instance_segmentation";
  const clone = value => JSON.parse(JSON.stringify(value));
  const selected = () => editor.objects[editor.selected];
  const message = text => { $("workspaceStatus").textContent = text; };
  const snapshot = () => ({objects: clone(editor.objects), selected: editor.selected});
  function remember(before) {
    if (!before) return;
    state.undo.push(before);
    // Bound history by both action count and serialized size.
    while (state.undo.length > 25 || (state.undo.length > 1 && JSON.stringify(state.undo).length > 16000000)) state.undo.shift();
    state.redo = [];
    cancelPreview();
    sync();
  }
  function checkpoint() { if (active()) remember(snapshot()); }
  function restore(value) {
    editor.objects = clone(value.objects); editor.selected = value.selected; editor.tool = "edit";
    state.vertex = null; state.draft = null; editor.interaction = null; state.before = null;
    tileCache = new WeakMap(); state.multi.clear(); state.cutLine = null; cancelPreview();
    closeBoxLabelPanel(); markEditorDirty(); sync(); drawDetail();
  }
  function undo(redo = false) {
    if (editor.interaction) return;
    if (state.draft) { state.draft = null; sync(); drawDetail(); return; }
    const from = redo ? state.redo : state.undo, to = redo ? state.undo : state.redo;
    if (!from.length) return;
    to.push(snapshot()); restore(from.pop());
  }
  function reset() {
    state.draft = state.vertex = state.before = state.cursor = null;
    state.undo = []; state.redo = []; tileCache = new WeakMap();
    state.multi.clear(); state.cutLine = null; state.preview = null; state.busy = false;
    configure();
  }
  function configure() {
    const on = active();
    $("segmentationTools").hidden = !on;
    $("segmentationOperations").hidden = !on;
    $("drawBoxTool").textContent = on ? "＋ 多边形" : "＋ 新增框";
    $("editBoxTool").textContent = on ? "选择 / 顶点" : "编辑框";
    $("drawBoxTool").title = on ? "绘制新实例（N）" : "新增框（N）";
    $("editBoxTool").title = on ? "选择实例并编辑顶点（V）" : "编辑框（V）";
    $("deleteBox").textContent = on ? "删除实例" : "删除选中框";
    $("workspaceLabelCaption").textContent = on ? "新实例标签" : "新增框标签";
    $("runInitialDetection").hidden = on;
    $("detailCanvas").setAttribute("aria-label", on ? "实例分割编辑画布" : "检测框编辑画布");
    const help = document.querySelector(".workspace-help");
    if (!help.dataset.detectionHelp) help.dataset.detectionHelp = help.textContent;
    help.textContent = on
      ? "多边形：逐点单击，Enter 或点击起点闭合，Esc 取消。选择 / 顶点：拖动顶点，Shift+点击边插入顶点，拖动内部移动多边形。追加轮廓可标记同一实例的分离部分。画笔补边，橡皮擦可形成孔洞；首次涂改多边形会转为像素掩码，可撤销恢复顶点。滚轮缩放，右键平移，Ctrl/⌘+Z 撤销。"
      : help.dataset.detectionHelp;
    sync();
  }
  function sync() {
    if (!$("segmentationTools")) return;
    const object = selected(), drafting = !!state.draft;
    const multiCount = state.multi.size;
    $("appendContour").disabled = !object || object.kind !== "polygon" || drafting;
    $("finishContour").disabled = !drafting || state.draft.points.length < 3;
    $("cancelContour").disabled = !drafting;
    $("deleteVertex").disabled = !object || object.kind !== "polygon" || !state.vertex || drafting;
    $("deleteContour").disabled = !object || object.kind !== "polygon" || !state.vertex || drafting;
    $("segUndo").disabled = !state.undo.length && !drafting;
    $("segRedo").disabled = !state.redo.length || drafting;
    $("eraseMask").disabled = !object || drafting;
    $("brushMask").disabled = drafting;
    $("newMask").disabled = drafting;
    $("segMerge").disabled = multiCount < 2 || drafting || state.busy;
    for (const id of ["segSmooth", "segShrink", "segExpand", "segSnap", "segSplit", "segCutTool"])
      $(id).disabled = !object || drafting || state.busy;
    $("segCutTool").classList.toggle("active", editor.tool === "cut");
    $("segCutTool").setAttribute("aria-pressed", String(editor.tool === "cut"));
    $("segApplyPreview").disabled = !state.preview || state.busy;
    $("segOperationPreview").hidden = !state.preview;
    if (state.preview) $("segOperationSummary").textContent = state.preview.summary;
    $("drawBoxTool").classList.toggle("active", editor.tool === "draw");
    $("editBoxTool").classList.toggle("active", editor.tool === "edit");
    $("drawBoxTool").setAttribute("aria-pressed", String(editor.tool === "draw"));
    $("editBoxTool").setAttribute("aria-pressed", String(editor.tool === "edit"));
    for (const [id, tool] of [["brushMask", "brush"], ["eraseMask", "erase"]]) {
      $(id).classList.toggle("active", editor.tool === tool);
      $(id).setAttribute("aria-pressed", String(editor.tool === tool));
    }
    $("segSelection").textContent = object
      ? `${object.label} · ${object.kind === "polygon" ? `${object.polygons.length} 段轮廓` : "像素掩码"}`
      : "未选中实例";
    if (multiCount > 1) $("segSelection").textContent = `已选 ${multiCount} 个实例（Ctrl/⌘+点击增减）`;
  }
  function setTool(tool) {
    if (editor.interaction) return;
    if (state.draft && tool !== editor.tool) { message("请先完成或取消当前轮廓。"); return; }
    editor.tool = tool; state.vertex = null; closeBoxLabelPanel();
    if (tool !== "cut") state.cutLine = null;
    cancelPreview();
    if (tool === "draw") { editor.selected = -1; state.multi.clear(); }
    $("detailCanvas").style.cursor = tool === "edit" ? "default" : "crosshair";
    updateDeleteButton(); sync(); drawDetail();
  }
  function inside(ring, x, y) {
    let result = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [ax, ay] = ring[i], [bx, by] = ring[j];
      if ((ay > y) !== (by > y) && x < (bx - ax) * (y - ay) / (by - ay) + ax) result = !result;
    }
    return result;
  }
  const cross = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  function intersects(a,b,c,d) {
    if (Math.max(a[0],b[0]) < Math.min(c[0],d[0]) || Math.max(c[0],d[0]) < Math.min(a[0],b[0]) ||
        Math.max(a[1],b[1]) < Math.min(c[1],d[1]) || Math.max(c[1],d[1]) < Math.min(a[1],b[1])) return false;
    return cross(a,b,c)*cross(a,b,d) <= 0 && cross(c,d,a)*cross(c,d,b) <= 0;
  }
  function validatePolygons(parts) {
    if (parts.length > 256 || parts.reduce((n,p)=>n+p.length,0) > 2048) throw Error("每个实例最多 256 段轮廓、2048 个顶点。");
    const edges = [];
    for (let r = 0; r < parts.length; r++) {
      const p = parts[r];
      if (p.length < 3 || new Set(p.map(v=>v.join(","))).size !== p.length) throw Error("轮廓至少需要三个不同顶点。");
      const area = p.reduce((s,v,i)=>s+p[(i+p.length-1)%p.length][0]*v[1]-v[0]*p[(i+p.length-1)%p.length][1],0);
      if (Math.abs(area)<1e-8) throw Error("轮廓面积为零，请调整顶点。");
      p.forEach((a,i)=>{
        const b=p[(i+1)%p.length], c=p[(i+2)%p.length];
        if(cross(a,b,c)===0 && (a[0]-b[0])*(c[0]-b[0])+(a[1]-b[1])*(c[1]-b[1])>0) throw Error("轮廓边不能折返重叠。");
        edges.push({r,i,a,b});
      });
    }
    for(let n=0;n<edges.length;n++) for(let m=n+1;m<edges.length;m++) {
      const a=edges[n], b=edges[m];
      if(a.r===b.r && (Math.abs(a.i-b.i)===1 || Math.abs(a.i-b.i)===parts[a.r].length-1)) continue;
      if(intersects(a.a,a.b,b.a,b.b)) throw Error("轮廓不能交叉或重叠；需要孔洞时请使用橡皮擦。");
    }
    for(let i=0;i<parts.length;i++) for(let j=i+1;j<parts.length;j++) {
      if(inside(parts[i],...parts[j][0]) || inside(parts[j],...parts[i][0])) throw Error("追加轮廓用于分离部分；孔洞请使用橡皮擦。");
    }
  }
  function bounds(object) {
    if (object.kind === "polygon") {
      const p=object.polygons.flat();
      object.bbox=[Math.min(...p.map(v=>v[0])),Math.min(...p.map(v=>v[1])),Math.max(...p.map(v=>v[0])),Math.max(...p.map(v=>v[1]))];
      object.area=object.polygons.reduce((sum,ring)=>sum+Math.abs(ring.reduce((s,p,i)=>s+ring[(i+ring.length-1)%ring.length][0]*p[1]-p[0]*ring[(i+ring.length-1)%ring.length][1],0))/2,0);
      return;
    }
    let left=Infinity, top=Infinity, right=0, bottom=0, area=0;
    for(const tile of object.mask.tiles) {
      let offset=0;
      tile.counts.forEach((length,index)=>{
        if(index%2 && length) {
          const sy=Math.floor(offset/SIZE), ey=Math.floor((offset+length-1)/SIZE);
          left=Math.min(left,tile.x+(sy===ey?offset%SIZE:0)); right=Math.max(right,tile.x+(sy===ey?(offset+length-1)%SIZE+1:SIZE));
          top=Math.min(top,tile.y+sy); bottom=Math.max(bottom,tile.y+ey+1); area+=length;
        }
        offset+=length;
      });
    }
    object.bbox=[left,top,right,bottom]; object.area=area;
  }
  function decode(tile) {
    const values=new Uint8Array(PIXELS); let at=0;
    tile.counts.forEach((n,i)=>{if(i%2)values.fill(1,at,at+n);at+=n;}); return values;
  }
  function encode(values,x,y) {
    const counts=[]; let bit=0,n=0,filled=0;
    for(const v of values){filled+=v;if(v===bit)n++;else{counts.push(n);n=1;bit=v;}}
    counts.push(n); return filled?{x,y,counts}:null;
  }
  function tileCanvas(tile, color) {
    let cache=tileCache.get(tile);
    if(cache?.color===color)return cache.canvas;
    const canvas=document.createElement("canvas");canvas.width=canvas.height=SIZE;
    const ctx=canvas.getContext("2d"), pixels=ctx.createImageData(SIZE,SIZE), values=decode(tile);
    const rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16));
    for(let i=0;i<PIXELS;i++)if(values[i]){pixels.data.set([...rgb,110],i*4);}
    ctx.putImageData(pixels,0,0);tileCache.set(tile,{color,canvas});return canvas;
  }
  function drawObject(ctx,object,index,fit) {
    const color=categoryColor(object.label), view=editor.view;
    ctx.save();ctx.translate(fit.x-view.x*fit.scale,fit.y-view.y*fit.scale);ctx.scale(fit.scale,fit.scale);
    if(object.kind==="polygon") {
      ctx.beginPath();
      object.polygons.forEach(ring=>{ring.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();});
      ctx.fillStyle=color;ctx.globalAlpha=.27;ctx.fill();ctx.globalAlpha=1;
      if(index===editor.selected){ctx.strokeStyle="#fff";ctx.lineWidth=5/fit.scale;ctx.stroke();}
      ctx.strokeStyle=color;ctx.lineWidth=2/fit.scale;ctx.stroke();
    } else if(object.kind==="mask") {
      ctx.imageSmoothingEnabled=false;
      for(const tile of object.mask.tiles){
        if(tile.x+SIZE<view.x||tile.y+SIZE<view.y||tile.x>view.x+view.width||tile.y>view.y+view.height)continue;
        ctx.drawImage(tileCanvas(tile,color),tile.x,tile.y);
      }
      if(index===editor.selected){const [x,y,r,b]=object.bbox;ctx.strokeStyle="#fff";ctx.lineWidth=1/fit.scale;ctx.setLineDash([4/fit.scale,4/fit.scale]);ctx.strokeRect(x,y,r-x,b-y);}
    }
    ctx.restore();
  }
  function drawHandles(ctx,fit) {
    const object=selected(), view=editor.view;
    if(object?.kind==="polygon" && editor.tool==="edit")object.polygons.forEach((ring,r)=>ring.forEach(([x,y],i)=>{
      const px=fit.x+(x-view.x)*fit.scale,py=fit.y+(y-view.y)*fit.scale;
      ctx.fillStyle=state.vertex?.r===r&&state.vertex?.i===i?"#ffcb47":"#fff";
      ctx.strokeStyle=categoryColor(object.label);ctx.lineWidth=2;ctx.fillRect(px-4,py-4,8,8);ctx.strokeRect(px-4,py-4,8,8);
    }));
  }
  function drawOverlay(ctx,fit) {
    if(state.draft) {
      ctx.save();ctx.strokeStyle="#20e080";ctx.fillStyle="#fff";ctx.lineWidth=2;ctx.setLineDash([5,3]);ctx.beginPath();
      state.draft.points.forEach(([x,y],i)=>{const px=fit.x+(x-editor.view.x)*fit.scale,py=fit.y+(y-editor.view.y)*fit.scale;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
      if(state.cursor)ctx.lineTo(fit.x+(state.cursor[0]-editor.view.x)*fit.scale,fit.y+(state.cursor[1]-editor.view.y)*fit.scale);
      ctx.stroke();ctx.setLineDash([]);
      state.draft.points.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(fit.x+(x-editor.view.x)*fit.scale,fit.y+(y-editor.view.y)*fit.scale,i?3:6,0,2*Math.PI);ctx.fill();ctx.stroke();});ctx.restore();
    }
    if(["brush","erase"].includes(editor.tool)&&state.cursor){
      ctx.save();ctx.strokeStyle=editor.tool==="erase"?"#ff544b":"#fff";ctx.lineWidth=1.5;ctx.beginPath();
      ctx.arc(fit.x+(state.cursor[0]-editor.view.x)*fit.scale,fit.y+(state.cursor[1]-editor.view.y)*fit.scale,radius()*fit.scale,0,Math.PI*2);ctx.stroke();ctx.restore();
    }
  }
  function drawPreview(ctx, fit) {
    if (state.preview) {
      const view = editor.view;
      ctx.save();
      ctx.translate(fit.x-view.x*fit.scale,fit.y-view.y*fit.scale);
      ctx.scale(fit.scale,fit.scale);
      for (const object of state.preview.result.objects) {
        if (!state.preview.result.affected_ids.includes(object.id)) continue;
        if (object.kind === "mask") {
          for (const tile of object.mask.tiles) {
            if (tile.x+SIZE<view.x||tile.y+SIZE<view.y||tile.x>view.x+view.width||tile.y>view.y+view.height) continue;
            ctx.drawImage(tileCanvas(tile,"#18e879"),tile.x,tile.y);
          }
        } else {
          ctx.beginPath();
          for (const ring of object.polygons) {ring.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();}
          ctx.fillStyle="rgba(24,232,121,.42)";ctx.fill();
          ctx.strokeStyle="#18e879";ctx.lineWidth=2/fit.scale;ctx.stroke();
        }
      }
      ctx.restore();
    }
    if (state.cutLine?.length) {
      ctx.save();ctx.strokeStyle="#31d8ed";ctx.lineWidth=3;ctx.setLineDash([8,4]);ctx.beginPath();
      state.cutLine.forEach(([x,y],i)=>{
        const px=fit.x+(x-editor.view.x)*fit.scale,py=fit.y+(y-editor.view.y)*fit.scale;
        i?ctx.lineTo(px,py):ctx.moveTo(px,py);
      });ctx.stroke();ctx.restore();
    }
  }
  function cancelPreview() {
    state.preview = null;
    if ($("segOperationPreview")) $("segOperationPreview").hidden = true;
  }
  const editRadius = () => Math.max(1,Math.min(64,Number($("segEditRadius").value)||3));
  const snapDistance = () => Math.max(1,Math.min(32,Number($("segSnapDistance").value)||10));
  function reportText(report) {
    const names={smooth:"平滑",shrink:"收缩",expand:"扩张",snap:"智能贴边",merge:"合并",split:"拆分"};
    const bits=[`${names[report.operation]}预览`];
    if(report.source_count)bits.push(`${report.source_count} → ${report.result_count} 个实例`);
    if(report.before_area!=null)bits.push(`面积 ${Math.round(report.before_area)} → ${Math.round(report.after_area)} px²`);
    if(report.pixel_changes!=null)bits.push(`${report.pixel_changes} 像素改变`);
    if(report.before_holes!=null)bits.push(`孔洞 ${report.before_holes} → ${report.after_holes}`);
    if(report.before_components!=null)bits.push(`分离区域 ${report.before_components} → ${report.after_components}`);
    if(report.edge_score_before!=null)bits.push(`边缘吻合度 ${report.edge_score_before} → ${report.edge_score_after}`);
    return bits.join(" · ");
  }
  async function startPreview(operation) {
    if (!active() || !editor.image || state.busy || state.draft || editor.interaction) return;
    const ids=operation==="merge"?[...state.multi]:selected()?[selected().id]:[];
    if (!ids.length) {message("请先选中实例。");return;}
    let targetLabel=null;
    if(operation==="merge"){
      if(ids.length<2){message("按 Ctrl 或 ⌘ 点击选择至少两个实例。");return;}
      const selectedObjects=editor.objects.filter(o=>ids.includes(o.id));
      if(new Set(selectedObjects.map(o=>o.label)).size>1){
        targetLabel=$("workspaceLabel").value;
        if(!confirm(`所选实例属于不同类别。合并后类别设为“${targetLabel}”，是否继续预览？`))return;
      }
    }
    const before=JSON.stringify(editor.objects),image=editor.image,projectId=current.id;
    state.busy=true;cancelPreview();sync();
    message("正在生成轮廓预览…");
    try{
      const response=await api(`/api/projects/${projectId}/segmentation/preview`,{
        method:"POST",body:JSON.stringify({image,objects:JSON.parse(before),selected_ids:ids,operation,
          radius:editRadius(),distance:snapDistance(),preserve_holes:$("segProtectHoles").checked,
          cut_line:operation==="split"?state.cutLine||[]:[],target_label:targetLabel})
      });
      if(current?.id!==projectId||editor.image!==image||JSON.stringify(editor.objects)!==before){message("标注已改变，请重新预览。");return;}
      state.preview={result:response,summary:reportText(response.report)};
      message("绿色为调整后的区域；确认效果后点击“应用预览”。");
    }catch(error){message(`预览失败：${error.message}`);}
    finally{state.busy=false;sync();drawDetail();}
  }
  function applyPreview(){
    if(!state.preview||state.busy)return;
    const result=state.preview.result;
    remember(snapshot());
    editor.objects=clone(result.objects);
    editor.selected=editor.objects.findIndex(o=>o.id===result.affected_ids[0]);
    editor.tool="edit";state.vertex=null;state.multi.clear();state.cutLine=null;
    cancelPreview();markEditorDirty();updateDeleteButton();sync();drawDetail();
  }
  function hit(x,y) {
    for(let i=editor.objects.length-1;i>=0;i--){
      const o=editor.objects[i], [l,t,r,b]=o.bbox;if(x<l||y<t||x>=r||y>=b)continue;
      if(o.kind==="polygon"&&o.polygons.some(p=>inside(p,x,y)))return i;
      if(o.kind==="mask"){
        const tx=Math.floor(x/SIZE)*SIZE,ty=Math.floor(y/SIZE)*SIZE,tile=o.mask.tiles.find(t=>t.x===tx&&t.y===ty);
        if(tile&&decode(tile)[(Math.floor(y)-ty)*SIZE+Math.floor(x)-tx])return i;
      }
    }return -1;
  }
  function vertexAt(x,y) {
    const o=selected();if(o?.kind!=="polygon")return null;
    const tolerance=9/canvasFit($("detailCanvas"),editor.view.width,editor.view.height).scale;
    let best=null,distance=tolerance;
    o.polygons.forEach((ring,r)=>ring.forEach((p,i)=>{const d=Math.hypot(x-p[0],y-p[1]);if(d<distance){best={r,i};distance=d;}}));return best;
  }
  function edgeAt(x,y) {
    const o=selected();if(o?.kind!=="polygon")return null;
    let distance=9/canvasFit($("detailCanvas"),editor.view.width,editor.view.height).scale,best=null;
    o.polygons.forEach((ring,r)=>ring.forEach((a,i)=>{
      const b=ring[(i+1)%ring.length],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)));
      const p=[a[0]+t*dx,a[1]+t*dy],d=Math.hypot(p[0]-x,p[1]-y);
      if(t>0&&t<1&&d<distance){distance=d;best={r,i,point:p};}
    }));return best;
  }
  function finish() {
    const draft=state.draft;if(!draft||draft.points.length<3)return;
    const object=draft.target?editor.objects.find(o=>o.id===draft.target):null;
    const polygons=[...(object?.polygons||[]),clone(draft.points)];
    try{validatePolygons(polygons);}catch(error){message(error.message);return;}
    remember(snapshot());
    if(object){object.polygons=polygons;bounds(object);editor.selected=editor.objects.indexOf(object);}
    else{const o={id:crypto.randomUUID?.()||`instance-${Date.now()}-${Math.random()}`,kind:"polygon",label:$("workspaceLabel").value,polygons,confidence:null,source:"human_reviewed"};bounds(o);editor.objects.push(o);editor.selected=editor.objects.length-1;}
    state.draft=null;editor.tool="edit";state.vertex=null;markEditorDirty();updateDeleteButton();sync();drawDetail();
  }
  function append() {if(selected()?.kind!=="polygon")return;state.draft={target:selected().id,points:[]};editor.tool="draw";sync();drawDetail();}
  function cancelDraft(){state.draft=null;sync();drawDetail();}
  function removeVertex(part=false) {
    const object=selected(),v=state.vertex;if(!object||!v||object.kind!=="polygon")return;
    if(!part&&object.polygons[v.r].length<=3){message("轮廓至少保留三个顶点；可使用“删除此轮廓”。");return;}
    const before=snapshot();
    if(part)object.polygons.splice(v.r,1);else object.polygons[v.r].splice(v.i,1);
    try{if(object.polygons.length)validatePolygons(object.polygons);}catch(error){editor.objects=before.objects;message(error.message);drawDetail();return;}
    remember(before);
    if(!object.polygons.length){editor.objects.splice(editor.selected,1);editor.selected=-1;}else bounds(object);
    state.vertex=null;markEditorDirty();updateDeleteButton();sync();drawDetail();
  }
  function deleteObject(){if(editor.selected<0||state.draft)return;remember(snapshot());state.multi.delete(selected().id);editor.objects.splice(editor.selected,1);editor.selected=-1;state.vertex=null;markEditorDirty();closeBoxLabelPanel();sync();updateDeleteButton();drawDetail();}
  function polygonToMask(object) {
    const locations=new Map();
    for(const ring of object.polygons){
      const xs=ring.map(p=>p[0]),ys=ring.map(p=>p[1]);
      const left=Math.floor(Math.min(...xs)/SIZE)*SIZE,top=Math.floor(Math.min(...ys)/SIZE)*SIZE;
      const right=Math.max(...xs),bottom=Math.max(...ys);
      if(Math.ceil((right-left)/SIZE)*Math.ceil((bottom-top)/SIZE)>MAX_TILES)throw Error("该轮廓范围过大，暂不能转为画笔掩码；请继续使用顶点编辑，或将对象划为较小实例。");
      for(let y=top;y<bottom;y+=SIZE)for(let x=left;x<right;x+=SIZE){
        locations.set(`${x},${y}`,[x,y]);
        if(locations.size>MAX_TILES)throw Error("实例超过画笔编辑容量，请继续使用顶点编辑或拆分实例。");
      }
    }
    const canvas=document.createElement("canvas");canvas.width=canvas.height=SIZE;const ctx=canvas.getContext("2d",{willReadFrequently:true}),tiles=[];
    for(const [x,y] of locations.values()){
      ctx.clearRect(0,0,SIZE,SIZE);ctx.beginPath();object.polygons.forEach(p=>{p.forEach(([px,py],i)=>i?ctx.lineTo(px-x,py-y):ctx.moveTo(px-x,py-y));ctx.closePath();});ctx.fill();
      const rgba=ctx.getImageData(0,0,SIZE,SIZE).data,values=new Uint8Array(PIXELS);
      for(let n=0;n<PIXELS;n++)if(x+n%SIZE<editor.meta.width&&y+Math.floor(n/SIZE)<editor.meta.height)values[n]=rgba[n*4+3]>=128?1:0;
      const tile=encode(values,x,y);if(tile)tiles.push(tile);
    }
    object.kind="mask";object.mask={encoding:"tile-rle-row-v1",tile_size:SIZE,tiles};object.polygons=null;
  }
  const radius=()=>Math.max(1,Math.min(256,Number($("brushRadius").value)||12));
  function paint(object,a,b,erase) {
    const r=radius(),x0=Math.max(0,Math.floor((Math.min(a[0],b[0])-r)/SIZE)*SIZE),y0=Math.max(0,Math.floor((Math.min(a[1],b[1])-r)/SIZE)*SIZE);
    const x1=Math.min(editor.meta.width,Math.max(a[0],b[0])+r),y1=Math.min(editor.meta.height,Math.max(a[1],b[1])+r);
    if(Math.ceil((x1-x0)/SIZE)*Math.ceil((y1-y0)/SIZE)>MAX_TILES)throw Error("单次笔画范围过大，请放大后分段绘制。");
    const tiles=new Map(object.mask.tiles.map(t=>[`${t.x},${t.y}`,t]));
    const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
    for(let y=y0;y<y1;y+=SIZE)for(let x=x0;x<x1;x+=SIZE){
      const key=`${x},${y}`,old=tiles.get(key);if(erase&&!old)continue;
      const data=old?decode(old):new Uint8Array(PIXELS);
      for(let py=0;py<SIZE&&y+py<editor.meta.height;py++)for(let px=0;px<SIZE&&x+px<editor.meta.width;px++){
        const cx=x+px+.5,cy=y+py+.5,t=length?Math.max(0,Math.min(1,((cx-a[0])*dx+(cy-a[1])*dy)/length)):0;
        if((cx-a[0]-t*dx)**2+(cy-a[1]-t*dy)**2<=r*r)data[py*SIZE+px]=erase?0:1;
      }
      const tile=encode(data,x,y);if(tile)tiles.set(key,tile);else tiles.delete(key);
    }
    if(tiles.size>MAX_TILES||[...tiles.values()].reduce((n,t)=>n+t.counts.length,0)>MAX_RUNS)throw Error("此实例已达到掩码编辑容量上限，请拆分实例或使用多边形。");
    object.mask.tiles=[...tiles.values()].sort((a,b)=>a.y-b.y||a.x-b.x);bounds(object);
  }
  function pointerDown(event) {
    if(!editor.meta||event.button>2||editor.interaction||state.busy)return;
    const [x,y]=detailImagePoint(event);state.cursor=[x,y];
    if(event.button===2||event.button===1||editor.spacePressed){editor.interaction={kind:"pan",clientX:event.clientX,clientY:event.clientY,viewX:editor.view.x,viewY:editor.view.y};event.currentTarget.setPointerCapture(event.pointerId);event.preventDefault();return;}
    if(editor.tool==="cut"){
      if(!selected()){message("请先选中要拆分的实例。");return;}
      state.cutLine=[[x,y]];cancelPreview();
      editor.interaction={kind:"cut"};event.currentTarget.setPointerCapture(event.pointerId);
      drawDetail();return;
    }
    if(editor.tool==="draw"){
      if(!state.draft)state.draft={target:null,points:[]};
      const points=state.draft.points,tol=9/canvasFit($("detailCanvas"),editor.view.width,editor.view.height).scale;
      if(points.length>=3&&Math.hypot(x-points[0][0],y-points[0][1])<tol){finish();return;}
      if(points.length>=2048){message("一个实例最多 2048 个顶点。");return;}
      points.push([x,y]);sync();drawDetail();return;
    }
    if(["brush","erase"].includes(editor.tool)){
      if(editor.tool==="erase"&&!selected())return;
      state.before=snapshot();
      try{
        if(!selected()) {editor.objects.push({id:`instance-${Date.now()}-${Math.random()}`,kind:"mask",label:$("workspaceLabel").value,confidence:null,source:"human_reviewed",mask:{encoding:"tile-rle-row-v1",tile_size:SIZE,tiles:[]},bbox:[x,y,x+1,y+1]});editor.selected=editor.objects.length-1;}
        if(selected().kind==="polygon")polygonToMask(selected());
        paint(selected(),[x,y],[x,y],editor.tool==="erase");
        editor.interaction={kind:"paint",last:[x,y]};event.currentTarget.setPointerCapture(event.pointerId);
      }catch(error){editor.objects=state.before.objects;editor.selected=state.before.selected;state.before=null;message(error.message);}
      sync();drawDetail();return;
    }
    if(event.ctrlKey||event.metaKey){
      const [cx,cy]=canvasPoint(event,event.currentTarget);
      const index=labelHitAt(cx,cy)>=0?labelHitAt(cx,cy):hit(x,y);
      if(index>=0){
        const id=editor.objects[index].id;
        if(state.multi.has(id))state.multi.delete(id);else state.multi.add(id);
        editor.selected=index;state.vertex=null;cancelPreview();sync();updateDeleteButton();drawDetail();
      }
      return;
    }
    state.multi.clear();cancelPreview();
    const [cx,cy]=canvasPoint(event,event.currentTarget),label=labelHitAt(cx,cy);
    // Vertex handles take priority over the label and filled region.
    const vertex=vertexAt(x,y);
    if(vertex){state.vertex=vertex;state.before=snapshot();editor.interaction={kind:"vertex",...vertex};event.currentTarget.setPointerCapture(event.pointerId);sync();drawDetail();return;}
    if(event.shiftKey){const edge=edgeAt(x,y);if(edge){const before=snapshot();selected().polygons[edge.r].splice(edge.i+1,0,edge.point);try{validatePolygons(selected().polygons);}catch(error){editor.objects=before.objects;message(error.message);return;}remember(before);state.vertex={r:edge.r,i:edge.i+1};bounds(selected());markEditorDirty();sync();drawDetail();return;}}
    if(label>=0){editor.selected=label;state.vertex=null;sync();updateDeleteButton();drawDetail();openBoxLabelPanel();return;}
    const index=hit(x,y);editor.selected=index;state.vertex=null;closeBoxLabelPanel();
    if(index>=0&&selected().kind==="polygon"){state.before=snapshot();editor.interaction={kind:"polygonMove",start:[x,y],polygons:clone(selected().polygons),bbox:[...selected().bbox]};event.currentTarget.setPointerCapture(event.pointerId);}
    sync();updateDeleteButton();drawDetail();
  }
  function pointerMove(event) {
    if(!editor.meta)return;
    const [x,y]=detailImagePoint(event);state.cursor=[x,y];const action=editor.interaction;
    if(!action){drawDetail();return;}
    if(action.kind==="cut"){
      const last=state.cutLine.at(-1);
      if(Math.hypot(x-last[0],y-last[1])>=2)state.cutLine.push([x,y]);
      drawDetail();return;
    }
    if(action.kind==="pan"){
      const c=event.currentTarget,rect=c.getBoundingClientRect(),fit=canvasFit(c,editor.view.width,editor.view.height);
      editor.view.x=action.viewX-(event.clientX-action.clientX)*c.width/rect.width/fit.scale;
      editor.view.y=action.viewY-(event.clientY-action.clientY)*c.height/rect.height/fit.scale;clampEditorView();
    }else if(action.kind==="vertex"){selected().polygons[action.r][action.i]=[x,y];bounds(selected());}
    else if(action.kind==="polygonMove"){
      const [l,t,r,b]=action.bbox,dx=Math.max(-l,Math.min(editor.meta.width-r,x-action.start[0])),dy=Math.max(-t,Math.min(editor.meta.height-b,y-action.start[1]));
      selected().polygons=action.polygons.map(p=>p.map(([px,py])=>[px+dx,py+dy]));bounds(selected());
    }else if(action.kind==="paint"){
      try{paint(selected(),action.last,[x,y],editor.tool==="erase");action.last=[x,y];}
      catch(error){cancelInteraction();message(error.message);}
    }
    drawDetail();
  }
  function pointerUp(event) {
    const action=editor.interaction;if(!action)return;
    if(action.kind==="pan")refreshEditorCrop(80);
    else if(action.kind==="cut"){
      const [x,y]=detailImagePoint(event),last=state.cutLine.at(-1);
      if(Math.hypot(x-last[0],y-last[1])>=1)state.cutLine.push([x,y]);
      message("分割线已画好；点击“拆分实例”预览结果。");
    }
    else if(action.kind!=="cut") {
      try{
        if(selected()?.kind==="polygon")validatePolygons(selected().polygons);
        if(selected()?.kind==="mask"&&!selected().mask.tiles.length){editor.objects.splice(editor.selected,1);editor.selected=-1;}
        if(state.before&&JSON.stringify(state.before.objects)!==JSON.stringify(editor.objects)){remember(state.before);markEditorDirty();}
      }catch(error){if(state.before){editor.objects=state.before.objects;editor.selected=state.before.selected;}message(error.message);}
    }
    state.before=null;editor.interaction=null;
    if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);
    sync();updateDeleteButton();drawDetail();
  }
  function cancelInteraction(){if(editor.interaction?.kind==="cut")state.cutLine=null;if(state.before){editor.objects=state.before.objects;editor.selected=state.before.selected;}state.before=null;editor.interaction=null;sync();drawDetail();}
  function keydown(event) {
    const key=event.key.toLowerCase();
    if (editor.interaction && key !== "escape") return;
    if((event.ctrlKey||event.metaKey)&&key==="z"){event.preventDefault();undo(event.shiftKey);return;}
    if((event.ctrlKey||event.metaKey)&&key==="y"){event.preventDefault();undo(true);return;}
    if(key==="enter"){event.preventDefault();finish();return;}
    if(key==="escape"){cancelInteraction();cancelDraft();state.cutLine=null;cancelPreview();sync();drawDetail();return;}
    if(key==="backspace"&&state.draft){event.preventDefault();state.draft.points.pop();sync();drawDetail();return;}
    if(key==="delete"){event.preventDefault();if(state.vertex)removeVertex();else deleteObject();return;}
    if(key==="n")setTool("draw");if(key==="v")setTool("edit");if(key==="b")setTool("brush");if(key==="e"&&selected())setTool("erase");
  }
  function canSave(){if(state.preview){message("请先应用或取消轮廓预览，再保存。");return false;}if(state.draft||editor.interaction){message("请先完成当前轮廓或笔画，再保存。");return false;}return true;}
  const dirty=()=>editor.dirty||!!state.draft||!!state.before;
  function init(){
    $("appendContour").onclick=append;$("finishContour").onclick=finish;$("cancelContour").onclick=cancelDraft;
    $("deleteVertex").onclick=()=>removeVertex();$("deleteContour").onclick=()=>removeVertex(true);
    $("segUndo").onclick=()=>undo();$("segRedo").onclick=()=>undo(true);
    $("brushMask").onclick=()=>setTool("brush");$("eraseMask").onclick=()=>setTool("erase");
    $("newMask").onclick=()=>{editor.selected=-1;setTool("brush");};
    $("brushRadius").onchange=()=>{$("brushRadius").value=radius();};
    for(const [id,op] of [["segSmooth","smooth"],["segShrink","shrink"],["segExpand","expand"],
                          ["segSnap","snap"],["segMerge","merge"],["segSplit","split"]])
      $(id).onclick=()=>startPreview(op);
    $("segCutTool").onclick=()=>setTool("cut");
    $("segApplyPreview").onclick=applyPreview;
    $("segCancelPreview").onclick=()=>{cancelPreview();sync();drawDetail();};
    for(const id of ["segEditRadius","segSnapDistance","segProtectHoles"])
      $(id).onchange=()=>{cancelPreview();sync();drawDetail();};
    $("detailCanvas").addEventListener("pointerleave",()=>{state.cursor=null;if(active())drawDetail();});
  }
  return {active,reset,configure,sync,setTool,drawObject,drawHandles,drawOverlay,pointerDown,pointerMove,pointerUp,
    cancelInteraction,keydown,deleteObject,checkpoint,canSave,dirty,init,drawPreview};
})();
