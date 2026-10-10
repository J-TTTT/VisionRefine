/* Video segmentation editor. All geometry uses pixels in the exact source frame. */
(() => {
  "use strict";
  const SIZE = 128, PIXELS = SIZE * SIZE, MAX_TILES = 4096, MAX_RUNS = 1000000;
  const copy = value => value == null ? null : JSON.parse(JSON.stringify(value));
  const colorValue = value => /^#[0-9a-f]{6}$/i.test(value || "") ? value : "#39c8ac";
  const cross = (a, b, c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  function inside(ring, x, y) {
    let found = false;
    for (let i=0, j=ring.length-1; i<ring.length; j=i++) {
      const [ax, ay]=ring[i], [bx, by]=ring[j];
      if ((ay>y)!==(by>y) && x<(bx-ax)*(y-ay)/(by-ay)+ax) found=!found;
    }
    return found;
  }
  function intersects(a, b, c, d) {
    if (Math.max(a[0],b[0])<Math.min(c[0],d[0]) || Math.max(c[0],d[0])<Math.min(a[0],b[0]) ||
        Math.max(a[1],b[1])<Math.min(c[1],d[1]) || Math.max(c[1],d[1])<Math.min(a[1],b[1])) return false;
    return cross(a,b,c)*cross(a,b,d)<=0 && cross(c,d,a)*cross(c,d,b)<=0;
  }
  function validatePolygons(parts, width, height) {
    if (!Array.isArray(parts) || !parts.length || parts.length>256 ||
        parts.some(p=>!Array.isArray(p)) || parts.reduce((n,p)=>n+p.length,0)>2048)
      throw Error("每个对象需要 1–256 段轮廓，最多 2048 个顶点。");
    const edges=[];
    parts.forEach((ring,r)=>{
      if (ring.length<3 || ring.some(p=>!Array.isArray(p) || p.length!==2 ||
          !p.every(Number.isFinite) || p[0]<0 || p[1]<0 || p[0]>width || p[1]>height))
        throw Error("轮廓至少需要三个位于原图范围内的有效顶点。");
      if (new Set(ring.map(p=>p.join(","))).size!==ring.length) throw Error("轮廓顶点不能重复，闭合点会自动补齐。");
      const area=ring.reduce((s,p,i)=>s+ring[(i+ring.length-1)%ring.length][0]*p[1]-p[0]*ring[(i+ring.length-1)%ring.length][1],0);
      if (Math.abs(area)<1e-8) throw Error("轮廓面积为零，请调整顶点。");
      ring.forEach((a,i)=>{
        const b=ring[(i+1)%ring.length], c=ring[(i+2)%ring.length];
        if (cross(a,b,c)===0 && (a[0]-b[0])*(c[0]-b[0])+(a[1]-b[1])*(c[1]-b[1])>0)
          throw Error("轮廓边不能折返重叠。");
        edges.push({r,i,a,b});
      });
    });
    for (let i=0;i<edges.length;i++) for (let j=i+1;j<edges.length;j++) {
      const a=edges[i],b=edges[j];
      if (a.r===b.r && (Math.abs(a.i-b.i)===1 || Math.abs(a.i-b.i)===parts[a.r].length-1)) continue;
      if (intersects(a.a,a.b,b.a,b.b)) throw Error("轮廓不能交叉、重叠或相接；孔洞请使用橡皮擦。");
    }
    for (let i=0;i<parts.length;i++) for (let j=i+1;j<parts.length;j++)
      if (inside(parts[i],...parts[j][0]) || inside(parts[j],...parts[i][0]))
        throw Error("追加轮廓用于分离的可见区域，孔洞请使用橡皮擦。");
  }
  function decode(tile) {
    const values=new Uint8Array(PIXELS); let at=0;
    tile.counts.forEach((n,i)=>{if(i%2)values.fill(1,at,at+n);at+=n;});
    return values;
  }
  function encode(values, x, y) {
    const counts=[]; let bit=0,n=0,filled=0;
    for (const v of values) {filled+=v;if(v===bit)n++;else{counts.push(n);n=1;bit=v;}}
    counts.push(n); return filled?{x,y,counts}:null;
  }
  function validateGeometry(value, width, height) {
    if (value==null) return null;
    if (value.kind==="polygon") {
      validatePolygons(value.polygons,width,height);
      return {kind:"polygon",polygons:copy(value.polygons)};
    }
    const mask=value.mask;
    if (value.kind!=="mask" || mask?.encoding!=="tile-rle-row-v1" || mask.tile_size!==SIZE ||
        !Array.isArray(mask.tiles) || mask.tiles.length>MAX_TILES) throw Error("不支持的分割几何或掩码格式。");
    if (!mask.tiles.length) return null;
    let runs=0; const locations=new Set();
    for (const tile of mask.tiles) {
      if (!Number.isInteger(tile.x) || !Number.isInteger(tile.y) || tile.x<0 || tile.y<0 ||
          tile.x%SIZE || tile.y%SIZE || tile.x>=width || tile.y>=height ||
          locations.has(`${tile.x},${tile.y}`)) throw Error("掩码块坐标无效或重复。");
      locations.add(`${tile.x},${tile.y}`);
      if (!Array.isArray(tile.counts) || tile.counts.length<2 || tile.counts.length>PIXELS+1 ||
          (runs+=tile.counts.length)>MAX_RUNS) throw Error("掩码超过编辑容量上限。");
      let offset=0,foreground=0;
      tile.counts.forEach((n,i)=>{
        if (!Number.isInteger(n) || n<0 || (i>0 && !n) || offset+n>PIXELS) throw Error("掩码游程无效。");
        if (i%2) {
          foreground+=n;
          for (let p=offset;p<offset+n;) {
            const row=Math.floor(p/SIZE),end=Math.min(offset+n,(row+1)*SIZE);
            if (tile.y+row>=height || tile.x+(end-1)%SIZE>=width) throw Error("掩码像素超出原图范围。");
            p=end;
          }
        }
        offset+=n;
      });
      if (offset!==PIXELS || !foreground) throw Error("掩码块必须包含前景并覆盖 128×128 像素。");
    }
    return {kind:"mask",mask:{encoding:"tile-rle-row-v1",tile_size:SIZE,
      tiles:copy(mask.tiles).sort((a,b)=>a.y-b.y||a.x-b.x)}};
  }

  function create({canvas,onChange=()=>{},onStatus=()=>{},onSelection=()=>{}}) {
    if (!(canvas instanceof HTMLCanvasElement)) throw Error("分割编辑器需要 canvas 元素。");
    const ctx=canvas.getContext("2d"),listeners=[];
    const state={width:0,height:0,image:null,url:null,key:null,geometry:null,overlays:[],color:"#39c8ac",
      ready:false,enabled:true,tool:"polygon",radius:12,draft:null,vertex:null,action:null,cursor:null,
      undo:[],redo:[],error:"",scale:1,fitScale:1,x:0,y:0,cssWidth:0,cssHeight:0,space:false};
    let generation=0,destroyed=false,tileCache=new WeakMap();
    canvas.tabIndex=0;
    canvas.setAttribute("aria-label","视频实例分割编辑画布");
    canvas.style.touchAction="none";
    const dirtyDraft=()=>!!state.draft || !!(state.action && state.action.kind!=="pan");
    const getState=()=>({ready:state.ready,enabled:state.enabled,tool:state.tool,dirtyDraft:dirtyDraft(),
      canUndo:!!state.undo.length||!!state.draft,canRedo:!!state.redo.length&&!dirtyDraft(),
      canSave:state.ready&&!dirtyDraft(),error:state.error,key:state.key,zoom:state.scale/state.fitScale,
      vertex:copy(state.vertex),draftPoints:state.draft?.length||0,
      view:{scale:state.scale,x:state.x,y:state.y,width:state.cssWidth,height:state.cssHeight}});
    const sync=()=>{if(!destroyed)onStatus(getState());};
    const report=message=>{state.error=message;sync();return false;};
    const usable=()=>!destroyed&&state.ready&&state.enabled;
    function fit() {
      if (!state.width || !state.cssWidth || !state.cssHeight) return;
      state.fitScale=Math.min(state.cssWidth/state.width,state.cssHeight/state.height);
      state.scale=state.fitScale;
      state.x=(state.cssWidth-state.width*state.scale)/2;
      state.y=(state.cssHeight-state.height*state.scale)/2;
      draw();sync();
    }
    function resize() {
      if (destroyed) return;
      const rect=canvas.getBoundingClientRect(),w=Math.max(1,rect.width),h=Math.max(1,rect.height);
      if (state.cssWidth===w && state.cssHeight===h && canvas.width===Math.round(w*(window.devicePixelRatio||1))) return;
      const center=[(state.cssWidth/2-state.x)/state.scale,(state.cssHeight/2-state.y)/state.scale];
      const zoom=state.scale/state.fitScale,hadSize=!!state.cssWidth;
      state.cssWidth=w;state.cssHeight=h;
      const dpr=window.devicePixelRatio||1;
      canvas.width=Math.max(1,Math.round(w*dpr));canvas.height=Math.max(1,Math.round(h*dpr));
      if (state.width && hadSize) {
        state.fitScale=Math.min(w/state.width,h/state.height);state.scale=state.fitScale*zoom;
        state.x=w/2-center[0]*state.scale;state.y=h/2-center[1]*state.scale;clampView();
      } else fit();
      draw();sync();
    }
    function clampView() {
      const w=state.width*state.scale,h=state.height*state.scale;
      state.x=w<=state.cssWidth?(state.cssWidth-w)/2:Math.min(0,Math.max(state.cssWidth-w,state.x));
      state.y=h<=state.cssHeight?(state.cssHeight-h)/2:Math.min(0,Math.max(state.cssHeight-h,state.y));
    }
    function tileImage(tile,color) {
      const cached=tileCache.get(tile);if(cached?.color===color)return cached.canvas;
      const target=document.createElement("canvas");target.width=target.height=SIZE;
      const c=target.getContext("2d"),pixels=c.createImageData(SIZE,SIZE),data=decode(tile);
      const rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16));
      for(let i=0;i<PIXELS;i++)if(data[i])pixels.data.set([...rgb,105],i*4);
      c.putImageData(pixels,0,0);tileCache.set(tile,{color,canvas:target});return target;
    }
    function drawGeometry(geometry,color,selected) {
      if (!geometry) return;
      ctx.save();ctx.translate(state.x,state.y);ctx.scale(state.scale,state.scale);
      if (geometry.kind==="polygon") {
        ctx.beginPath();geometry.polygons.forEach(ring=>{ring.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();});
        ctx.fillStyle=color;ctx.globalAlpha=.26;ctx.fill();ctx.globalAlpha=1;
        if(selected){ctx.lineWidth=4/state.scale;ctx.strokeStyle="#fff";ctx.stroke();}
        ctx.lineWidth=2/state.scale;ctx.strokeStyle=color;ctx.stroke();
      } else {
        ctx.imageSmoothingEnabled=false;
        for(const tile of geometry.mask.tiles) {
          if ((tile.x+SIZE)*state.scale+state.x<0 || (tile.y+SIZE)*state.scale+state.y<0 ||
              tile.x*state.scale+state.x>state.cssWidth || tile.y*state.scale+state.y>state.cssHeight) continue;
          ctx.drawImage(tileImage(tile,color),tile.x,tile.y);
        }
      }
      ctx.restore();
    }
    const screen=p=>[state.x+p[0]*state.scale,state.y+p[1]*state.scale];
    function draw() {
      if(destroyed)return;
      ctx.setTransform(canvas.width/state.cssWidth||1,0,0,canvas.height/state.cssHeight||1,0,0);
      ctx.clearRect(0,0,state.cssWidth,state.cssHeight);
      ctx.fillStyle="#101821";ctx.fillRect(0,0,state.cssWidth,state.cssHeight);
      if(!state.ready||!state.image)return;
      ctx.drawImage(state.image,state.x,state.y,state.width*state.scale,state.height*state.scale);
      for(const overlay of state.overlays)drawGeometry(overlay.geometry,colorValue(overlay.color),false);
      drawGeometry(state.geometry,state.color,true);
      if(state.geometry?.kind==="polygon"&&state.tool==="edit") {
        state.geometry.polygons.forEach((ring,r)=>ring.forEach((p,i)=>{
          const [x,y]=screen(p);ctx.fillStyle=state.vertex?.r===r&&state.vertex?.i===i?"#ffcf57":"#fff";
          ctx.strokeStyle=state.color;ctx.lineWidth=1.5;ctx.fillRect(x-4,y-4,8,8);ctx.strokeRect(x-4,y-4,8,8);
        }));
      }
      if(state.draft) {
        ctx.strokeStyle=state.color;ctx.lineWidth=2;ctx.setLineDash([5,3]);ctx.beginPath();
        state.draft.forEach((p,i)=>{const [x,y]=screen(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
        if(state.cursor&&state.draft.length)ctx.lineTo(...screen(state.cursor));
        ctx.stroke();ctx.setLineDash([]);
        state.draft.forEach((p,i)=>{ctx.beginPath();ctx.arc(...screen(p),i?3:6,0,Math.PI*2);ctx.fillStyle="#fff";ctx.fill();ctx.stroke();});
      }
      if(state.cursor&&state.enabled&&["brush","erase"].includes(state.tool)) {
        ctx.beginPath();ctx.arc(...screen(state.cursor),state.radius*state.scale,0,Math.PI*2);
        ctx.strokeStyle=state.tool==="erase"?"#ff715f":"#fff";ctx.lineWidth=1.5;ctx.stroke();
      }
    }
    function trimHistory(list) {
      while(list.length>25 || (list.length>1&&JSON.stringify(list).length>16000000))list.shift();
    }
    function changed(before) {
      if(JSON.stringify(before)===JSON.stringify(state.geometry)){draw();sync();return false;}
      state.undo.push(before);trimHistory(state.undo);state.redo=[];state.error="";
      onChange(copy(state.geometry),{key:state.key});draw();sync();return true;
    }
    function setGeometry(value,{silent=true,resetHistory=false}={}) {
      if(dirtyDraft())return report("请先完成或取消当前轮廓或笔画。");
      try {
        const next=validateGeometry(value,state.width,state.height),before=copy(state.geometry);
        state.geometry=next;state.vertex=null;tileCache=new WeakMap();state.error="";
        if(resetHistory){state.undo=[];state.redo=[];}
        if(!silent)changed(before);else{draw();sync();}
        return true;
      } catch(error){return report(error.message);}
    }
    async function load({imageUrl,width,height,geometry=null,overlays=[],key=null,color}) {
      if(destroyed)return false;
      if(dirtyDraft())return report("请先完成或取消当前轮廓或笔画，再切换帧或对象。");
      let next,others;
      try {
        if(!Number.isInteger(width)||!Number.isInteger(height)||width<=0||height<=0)throw Error("原始帧尺寸无效。");
        next=validateGeometry(geometry,width,height);
        others=overlays.map(item=>({...item,geometry:validateGeometry(item.geometry,width,height)}));
      } catch(error){++generation;state.ready=false;draw();return report(error.message);}
      const token=++generation,sameImage=state.ready&&state.url===imageUrl&&state.width===width&&state.height===height;
      const sameKey=state.key===key;
      state.ready=false;state.error="";state.vertex=null;state.cursor=null;state.key=key;state.space=false;
      if(!sameKey){state.undo=[];state.redo=[];}
      sync();draw();
      const image=sameImage?state.image:new Image();
      try {
        if(!sameImage)await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error("无法加载原始帧，请重试。"));image.src=imageUrl;});
        if(destroyed||token!==generation)return false;
        if(image.naturalWidth!==width||image.naturalHeight!==height)throw Error("帧图像尺寸与原始视频尺寸不一致，已暂停编辑。");
        const dimensionsChanged=state.width!==width||state.height!==height;
        state.image=image;state.url=imageUrl;state.width=width;state.height=height;
        state.geometry=next;state.overlays=others;state.color=colorValue(color);state.ready=true;
        tileCache=new WeakMap();resize();if(dimensionsChanged)fit();draw();sync();return true;
      } catch(error){if(!destroyed&&token===generation){state.ready=false;report(error.message);draw();}return false;}
    }
    function point(event,clamp=true) {
      const rect=canvas.getBoundingClientRect();
      const x=(event.clientX-rect.left-state.x)/state.scale,y=(event.clientY-rect.top-state.y)/state.scale;
      return clamp?[Math.max(0,Math.min(state.width,x)),Math.max(0,Math.min(state.height,y))]:[x,y];
    }
    function setTool(tool) {
      if(!["polygon","edit","brush","erase","pan"].includes(tool))return false;
      if(dirtyDraft())return report("请先完成或取消当前轮廓或笔画。");
      if(tool==="polygon"&&state.geometry?.kind==="mask")return report("像素掩码请用画笔补充；清空当前形状后可重新绘制多边形。");
      state.tool=tool;state.vertex=null;state.error="";
      canvas.style.cursor=tool==="pan"?"grab":tool==="edit"?"default":"crosshair";
      draw();sync();return true;
    }
    function finishPolygon() {
      if(!usable()||!state.draft)return false;
      if(state.draft.length<3)return report("至少绘制三个顶点后才能完成轮廓。");
      const before=copy(state.geometry);
      try {
        state.geometry=validateGeometry({kind:"polygon",polygons:[...(state.geometry?.polygons||[]),state.draft]},state.width,state.height);
        state.draft=null;state.tool="edit";canvas.style.cursor="default";state.vertex=null;changed(before);return true;
      } catch(error){return report(error.message);}
    }
    function cancelDraft() {
      if(state.action?.kind!=="pan"&&state.action)state.geometry=state.action.before;
      if(state.action?.pointerId!=null&&canvas.hasPointerCapture(state.action.pointerId))canvas.releasePointerCapture(state.action.pointerId);
      state.action=null;state.draft=null;state.error="";draw();sync();return true;
    }
    function appendContour() {
      if(!usable()||!setTool("polygon"))return false;
      state.draft=[];sync();draw();return true;
    }
    function removeVertex(contour=false) {
      if(!usable()||dirtyDraft()||!state.vertex||state.geometry?.kind!=="polygon")return false;
      const before=copy(state.geometry),{r,i}=state.vertex,parts=copy(state.geometry.polygons);
      if(contour)parts.splice(r,1);else if(parts[r].length>3)parts[r].splice(i,1);else return report("轮廓至少需要三个顶点；可使用删除轮廓。");
      try{state.geometry=parts.length?validateGeometry({kind:"polygon",polygons:parts},state.width,state.height):null;state.vertex=null;changed(before);return true;}
      catch(error){return report(error.message);}
    }
    function clear() {
      if(!usable()||dirtyDraft())return report("请先完成或取消当前轮廓或笔画。");
      const before=copy(state.geometry);state.geometry=null;state.vertex=null;changed(before);return true;
    }
    function history(redo=false) {
      if(!usable()||state.action)return false;
      if(state.draft){cancelDraft();return true;}
      const from=redo?state.redo:state.undo,to=redo?state.undo:state.redo;if(!from.length)return false;
      to.push(copy(state.geometry));trimHistory(to);state.geometry=from.pop();state.vertex=null;state.error="";
      if(state.geometry?.kind==="mask"&&state.tool==="polygon")state.tool="edit";
      onChange(copy(state.geometry),{key:state.key});draw();sync();return true;
    }
    function polygonToMask(geometry) {
      if(!geometry)return {kind:"mask",mask:{encoding:"tile-rle-row-v1",tile_size:SIZE,tiles:[]}};
      if(geometry.kind==="mask")return geometry;
      const locations=new Map();
      for(const ring of geometry.polygons) {
        const left=Math.floor(Math.min(...ring.map(p=>p[0]))/SIZE)*SIZE,top=Math.floor(Math.min(...ring.map(p=>p[1]))/SIZE)*SIZE;
        const right=Math.max(...ring.map(p=>p[0])),bottom=Math.max(...ring.map(p=>p[1]));
        if(Math.ceil((right-left)/SIZE)*Math.ceil((bottom-top)/SIZE)>MAX_TILES)throw Error("轮廓超过画笔编辑容量，请继续使用顶点编辑。");
        for(let y=top;y<bottom;y+=SIZE)for(let x=left;x<right;x+=SIZE){locations.set(`${x},${y}`,[x,y]);if(locations.size>MAX_TILES)throw Error("轮廓超过画笔编辑容量。");}
      }
      const target=document.createElement("canvas");target.width=target.height=SIZE;
      const c=target.getContext("2d",{willReadFrequently:true}),tiles=[];
      for(const [x,y] of locations.values()) {
        c.clearRect(0,0,SIZE,SIZE);c.beginPath();
        for(const ring of geometry.polygons){ring.forEach(([px,py],i)=>i?c.lineTo(px-x,py-y):c.moveTo(px-x,py-y));c.closePath();}c.fill();
        const rgba=c.getImageData(0,0,SIZE,SIZE).data,values=new Uint8Array(PIXELS);
        for(let n=0;n<PIXELS;n++)if(x+n%SIZE<state.width&&y+Math.floor(n/SIZE)<state.height)values[n]=rgba[n*4+3]>=128?1:0;
        const tile=encode(values,x,y);if(tile)tiles.push(tile);
      }
      return {kind:"mask",mask:{encoding:"tile-rle-row-v1",tile_size:SIZE,tiles}};
    }
    function paint(a,b,erase) {
      const r=state.radius,x0=Math.max(0,Math.floor((Math.min(a[0],b[0])-r)/SIZE)*SIZE),y0=Math.max(0,Math.floor((Math.min(a[1],b[1])-r)/SIZE)*SIZE);
      const x1=Math.min(state.width,Math.max(a[0],b[0])+r),y1=Math.min(state.height,Math.max(a[1],b[1])+r);
      if(Math.ceil((x1-x0)/SIZE)*Math.ceil((y1-y0)/SIZE)>MAX_TILES)throw Error("单次笔画范围过大，请放大后分段绘制。");
      const tiles=new Map(state.geometry.mask.tiles.map(t=>[`${t.x},${t.y}`,t]));
      const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
      for(let y=y0;y<y1;y+=SIZE)for(let x=x0;x<x1;x+=SIZE) {
        const key=`${x},${y}`,old=tiles.get(key);if(erase&&!old)continue;
        const data=old?decode(old):new Uint8Array(PIXELS);
        for(let py=0;py<SIZE&&y+py<state.height;py++)for(let px=0;px<SIZE&&x+px<state.width;px++) {
          const cx=x+px+.5,cy=y+py+.5,t=length?Math.max(0,Math.min(1,((cx-a[0])*dx+(cy-a[1])*dy)/length)):0;
          if((cx-a[0]-t*dx)**2+(cy-a[1]-t*dy)**2<=r*r)data[py*SIZE+px]=erase?0:1;
        }
        const tile=encode(data,x,y);if(tile)tiles.set(key,tile);else tiles.delete(key);
      }
      if(tiles.size>MAX_TILES||[...tiles.values()].reduce((n,t)=>n+t.counts.length,0)>MAX_RUNS)throw Error("此对象已达到掩码编辑容量上限。");
      state.geometry.mask.tiles=[...tiles.values()].sort((a,b)=>a.y-b.y||a.x-b.x);
    }
    function hit(geometry,p) {
      if(!geometry)return false;
      if(geometry.kind==="polygon")return geometry.polygons.some(r=>inside(r,...p));
      const x=Math.floor(p[0]),y=Math.floor(p[1]),tile=geometry.mask.tiles.find(t=>x>=t.x&&x<t.x+SIZE&&y>=t.y&&y<t.y+SIZE);
      if(!tile)return false;
      const target=(y-tile.y)*SIZE+x-tile.x;let at=0;
      for(let i=0;i<tile.counts.length;i++){at+=tile.counts[i];if(target<at)return !!(i%2);}return false;
    }
    function vertexAt(p) {
      if(state.geometry?.kind!=="polygon")return null;
      let closest=null,distance=9/state.scale;
      state.geometry.polygons.forEach((ring,r)=>ring.forEach((q,i)=>{const d=Math.hypot(p[0]-q[0],p[1]-q[1]);if(d<=distance){distance=d;closest={r,i};}}));return closest;
    }
    function edgeAt(p) {
      if(state.geometry?.kind!=="polygon")return null;
      let closest=null,distance=8/state.scale;
      state.geometry.polygons.forEach((ring,r)=>ring.forEach((a,i)=>{
        const b=ring[(i+1)%ring.length],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)));
        const q=[a[0]+t*dx,a[1]+t*dy],d=Math.hypot(q[0]-p[0],q[1]-p[1]);
        if(d<distance){closest={r,i,point:q};distance=d;}
      }));return closest;
    }
    function capture(event,action) {
      state.action={...action,pointerId:event.pointerId};canvas.setPointerCapture(event.pointerId);sync();
    }
    function down(event) {
      if(!usable()||state.action||event.button>2)return;
      canvas.focus({preventScroll:true});event.preventDefault();const p=point(event,false);state.cursor=p;state.error="";
      if(event.button===1||event.button===2||state.space||state.tool==="pan") {
        capture(event,{kind:"pan",start:[event.clientX,event.clientY],x:state.x,y:state.y});return;
      }
      if(p[0]<0||p[1]<0||p[0]>state.width||p[1]>state.height)return;
      if(state.tool==="polygon") {
        if(state.geometry?.kind==="mask"){report("像素掩码请使用画笔补充。");return;}
        if(!state.draft)state.draft=[];
        if(state.draft.length>=3&&Math.hypot(p[0]-state.draft[0][0],p[1]-state.draft[0][1])<8/state.scale){finishPolygon();return;}
        if(state.draft.length+(state.geometry?.polygons?.reduce((n,r)=>n+r.length,0)||0)>=2048){report("每个对象最多 2048 个顶点。");return;}
        state.draft.push(p);draw();sync();return;
      }
      if(["brush","erase"].includes(state.tool)) {
        if(state.tool==="erase"&&!state.geometry)return;
        const before=copy(state.geometry);
        try{state.geometry=polygonToMask(state.geometry);paint(p,p,state.tool==="erase");capture(event,{kind:"paint",before,last:p,erase:state.tool==="erase"});draw();}
        catch(error){state.geometry=before;draw();report(error.message);}return;
      }
      const vertex=vertexAt(p);
      if(vertex){state.vertex=vertex;capture(event,{kind:"vertex",before:copy(state.geometry),...vertex});draw();return;}
      if(event.shiftKey) {
        const edge=edgeAt(p);
        if(edge){const before=copy(state.geometry),next=copy(before);next.polygons[edge.r].splice(edge.i+1,0,edge.point);
          try{state.geometry=validateGeometry(next,state.width,state.height);state.vertex={r:edge.r,i:edge.i+1};changed(before);}catch(error){report(error.message);}return;}
      }
      state.vertex=null;
      if(hit(state.geometry,p)&&state.geometry.kind==="polygon") {
        const pts=state.geometry.polygons.flat();
        capture(event,{kind:"move",before:copy(state.geometry),start:p,bounds:[Math.min(...pts.map(q=>q[0])),Math.min(...pts.map(q=>q[1])),Math.max(...pts.map(q=>q[0])),Math.max(...pts.map(q=>q[1]))]});
      } else if(!hit(state.geometry,p)) {
        const other=[...state.overlays].reverse().find(item=>hit(item.geometry,p));if(other)onSelection(other.id);
      }
      draw();sync();
    }
    function move(event) {
      if(!usable())return;
      const p=point(event),action=state.action;state.cursor=p;
      if(action&&event.pointerId!==action.pointerId)return;
      if(action?.kind==="pan") {
        state.x=action.x+event.clientX-action.start[0];state.y=action.y+event.clientY-action.start[1];clampView();
      } else if(action?.kind==="vertex")state.geometry.polygons[action.r][action.i]=p;
      else if(action?.kind==="move") {
        const [l,t,r,b]=action.bounds,dx=Math.max(-l,Math.min(state.width-r,p[0]-action.start[0])),dy=Math.max(-t,Math.min(state.height-b,p[1]-action.start[1]));
        state.geometry.polygons=action.before.polygons.map(ring=>ring.map(([x,y])=>[x+dx,y+dy]));
      } else if(action?.kind==="paint") {
        try{paint(action.last,p,action.erase);action.last=p;}catch(error){cancelDraft();report(error.message);}
      }
      draw();
    }
    function up(event) {
      const action=state.action;if(!action||event.pointerId!==action.pointerId)return;
      if(action.kind!=="pan") {
        try{state.geometry=validateGeometry(state.geometry,state.width,state.height);state.action=null;changed(action.before);}
        catch(error){state.geometry=action.before;state.action=null;report(error.message);}
      } else state.action=null;
      if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
      draw();sync();
    }
    function wheel(event) {
      if(!usable()||state.action)return;
      event.preventDefault();const p=point(event,false),old=state.scale;
      state.scale=Math.max(state.fitScale,Math.min(state.fitScale*32,old*Math.exp(-event.deltaY*.0015)));
      state.x+=p[0]*(old-state.scale);state.y+=p[1]*(old-state.scale);clampView();draw();sync();
    }
    function keydown(event) {
      if(document.activeElement!==canvas||!usable())return;
      const key=event.key.toLowerCase(),mod=event.ctrlKey||event.metaKey;
      const handled=key==="escape"||key==="enter"||key===" "||key==="delete"||
        (key==="backspace"&&state.draft)||(mod&&["z","y"].includes(key))||(!mod&&["n","v","b","e"].includes(key));
      if(!handled)return;
      event.preventDefault();event.stopPropagation();
      if(key==="escape"){cancelDraft();return;}
      if(state.action)return;
      if(key===" "){state.space=true;return;}
      if(mod&&key==="z"){history(event.shiftKey);return;}
      if(mod&&key==="y"){history(true);return;}
      if(key==="enter"){finishPolygon();return;}
      if(key==="backspace"&&state.draft){state.draft.pop();draw();sync();return;}
      if(key==="delete"){if(state.vertex)removeVertex();return;}
      const tool={n:"polygon",v:"edit",b:"brush",e:"erase"}[key];if(tool)setTool(tool);
    }
    function listen(target,event,handler,options) {target.addEventListener(event,handler,options);listeners.push(()=>target.removeEventListener(event,handler,options));}
    listen(canvas,"pointerdown",down);listen(canvas,"pointermove",move);listen(canvas,"pointerup",up);
    listen(canvas,"pointercancel",()=>cancelDraft());
    listen(canvas,"lostpointercapture",()=>{if(state.action)cancelDraft();});
    listen(canvas,"pointerleave",()=>{if(!state.action){state.cursor=null;draw();}});
    listen(canvas,"wheel",wheel,{passive:false});listen(canvas,"contextmenu",event=>event.preventDefault());
    listen(canvas,"keydown",keydown);
    listen(canvas,"keyup",event=>{if(event.key===" "){state.space=false;event.preventDefault();event.stopPropagation();}});
    listen(canvas,"blur",()=>{state.space=false;});
    const observer=new ResizeObserver(resize);observer.observe(canvas);resize();
    return {load,setGeometry,setTool,finishPolygon,cancelDraft,appendContour,clear,fit,
      deleteVertex:()=>removeVertex(),deleteContour:()=>removeVertex(true),undo:()=>history(),redo:()=>history(true),
      setRadius:value=>{state.radius=Math.max(1,Math.min(256,Number(value)||12));draw();return state.radius;},
      setEnabled:enabled=>{if(!enabled&&dirtyDraft())return report("请先完成或取消当前轮廓或笔画。");state.enabled=!!enabled;state.cursor=null;draw();sync();return true;},
      canSave:()=>state.ready&&!dirtyDraft(),isDraftDirty:dirtyDraft,getGeometry:()=>copy(state.geometry),getState,
      destroy:()=>{if(destroyed)return;cancelDraft();destroyed=true;++generation;observer.disconnect();listeners.forEach(remove=>remove());state.image=null;tileCache=new WeakMap();}};
  }
  window.VideoSegmentationCanvas={create};
})();
