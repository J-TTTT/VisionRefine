document.getElementById('imageAI').onclick = async () => {
  const status = document.getElementById('workspaceStatus');
  if (!current || !editor.image || editor.loading) { status.textContent='请先在工作台打开一张图片。'; return; }
  const clean=()=>!editor.dirty && !(Segmentation.active() && Segmentation.dirty());
  if(!clean()){status.textContent='请先保存当前图片的修改，再使用 AI 辅助标注。';return;}
  const pid=current.id,image=editor.image;
  const selected=editor.objects[editor.selected];
  try{
    await window.VisionRefineAI.open({pid,image,task:current.task,labels:current.labels,width:editor.meta.width,height:editor.meta.height,
      initialLabel:selected?.label || document.getElementById('workspaceLabel').value,
      imageUrl:`/api/projects/${pid}/thumbnail/${encodedPath(image)}`,
      selectedGeometry:selected&&['mask','polygon'].includes(selected.kind)?{kind:selected.kind,mask:selected.mask,polygons:selected.polygons}:null,
      beforeApply:()=>current?.id===pid&&editor.image===image&&clean(),
      onApplied:async()=>{if(current?.id===pid&&editor.image===image){if(!clean()){status.textContent='AI 建议已保存为草稿；当前新修改仍保留在画布上。';return;}await loadEditorImage(image,editor.meta);status.textContent='AI 建议已进入草稿，检查后请点击“保存人工检查”。';}}
    });
  }catch(error){status.textContent=error.message;}
};
