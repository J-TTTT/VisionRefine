# 模型部署与自定义服务接口

## 默认服务

应用服务不依赖 PyTorch。模型运行在独立进程中；默认地址为 http://127.0.0.1:8030，面向应用后端。

需要 Python 3.10+、支持 CUDA 12.6 的驱动和足够磁盘空间。RTX 3090 24GB 用于本次部署；不同模型串行加载，共用一张卡。安装期间会同时保存 wheel 和解压后的依赖，请预留约 20GB 空间；已有 OWLv2 缓存可复用。

```bash
# 在项目根目录。VISIONREFINE_PYTHON 可指向现有的 Python 3.10+。
VISIONREFINE_PYTHON=.venv/bin/python bash scripts/setup_ai.sh

# 默认只对本机监听。多卡机器明确选择一张空闲卡。
CUDA_VISIBLE_DEVICES=0 workspace/cache/ai/venv/bin/python -m visionrefine.ai_worker --port 8030

# 另一个终端启动网页。
.venv/bin/python -m visionrefine.server --host 0.0.0.0 --port 8022
```

安装脚本固定 torch 2.8.0 + CUDA 12.6、torchvision 0.23.0、transformers 4.57.1、SAM 2 官方 Git 提交，以及三个默认模型的 Hugging Face 提交 ID。重试继续同一版本。大文件分块续传并检查官方 SHA-256。

默认视频理解采用 [Qwen3-VL 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，官方模型仓库标注为 Apache-2.0 许可。分割使用 [SAM 2.1 Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny)，检测使用 [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)。

模型目录与 Python 环境均在 workspace/cache/ai/。下载脚本优先复用当前用户缓存中的 OWLv2。model_paths.json 记录本地路径，model_revisions.json 记录权重版本。模型首次请求时加载，不在网页进程启动时下载。

默认 Qwen3-VL 使用原生视频输入，按原视频 PTS 提供采样帧时间，不在模型处理器中再次抽帧。时间提示会按模型的时间合并规则编码，事件边界仍需逐帧复核。自定义 OpenAI 兼容服务使用下文的多图加时间戳协议。

如果 Hugging Face 下载较慢，可在安装命令前设置 VISIONREFINE_MODEL_SOURCE=modelscope。Qwen 权重将从官方 ModelScope 仓库直连下载，只有文件 SHA-256 与固定的 Hugging Face 版本一致才会使用该镜像。

在 /ai 点击“测试连接”，分别查看服务可达性与权重就绪状态。默认服务不可用时任务会失败并保留错误，人工标注仍可使用。

如把 worker 部署到另一台服务器，设置 VISIONREFINE_AI_TOKEN 后使用 --host 0.0.0.0。在应用服务器设置对应密钥环境变量，并在 /ai 填写该变量名称。避免在地址或配置文件中放明文密钥。

## 选择自己的模型

1. 先部署模型服务。
2. 打开 /ai，添加服务名称、地址、协议、模型名及支持的能力。
3. 测试连接并保存。
4. 可修改全局默认，也可在某个项目的 AI 面板选择该服务。项目选择不会更改其他项目。
5. 已提交任务使用提交时的配置；后续配置修改只影响新任务。

### OpenAI 兼容视觉接口

支持图片框建议、视频描述和事件。基础地址通常以 /v1 结尾，应用调用：

- GET /models：返回 data 数组，其中包含配置的模型 id。
- POST /chat/completions：messages 中包含文本和 image_url（JPEG data URL），temperature=0，max_tokens=2048。

模型需要支持多图输入和 JSON 输出。时间戳以额外文本逐帧提供。视频“理解”由多帧视觉输入实现，不依赖服务接受原始视频文件。

分割需要下一节的专用协议。通用聊天接口不能可靠代替像素级掩码服务。

## VisionRefine HTTP 协议 1.0

基础地址不含 /v1，例如 http://model-server:8030。可选 Authorization: Bearer <token>。所有坐标使用**原始显示尺寸的像素**，所有帧索引从 0 开始，所有时间使用原视频 PTS 时间轴上的秒。

### 服务发现

GET /capabilities：

```json
{
  "protocol_version": "1.0",
  "models": [{
    "id": "my-model",
    "capabilities": ["image_segmentation", "video_segmentation"],
    "ready": true,
    "detail": "Weights are available locally"
  }]
}
```

能力名称为 image_detection、image_segmentation、video_segmentation、video_caption、video_events。

### 提交任务

POST /jobs，返回 HTTP 202 和 {"id":"worker-...","status":"queued"}。

请求示例（图片分割）：

```json
{
  "model": "my-model",
  "capability": "image_segmentation",
  "width": 1920,
  "height": 1080,
  "label": "person",
  "labels": ["person"],
  "threshold": 0.2,
  "prompt": {
    "points": [[350, 200], [800, 600]],
    "point_labels": [1, 0],
    "box": null,
    "geometry": null
  },
  "frames": [{
    "frame_index": 0,
    "timestamp": 0.0,
    "jpeg": "<base64 JPEG without a data URL prefix>"
  }]
}
```

传输图片可能缩放到最长边 2048（图片任务）或 1024（视频任务）。**width／height 和提示几何保持原始尺寸**，服务需要自行换算模型输入坐标，并将结果换算回来。

视频传播附加字段：

- start_frame、end_frame：包含端点的请求区间。
- seed_frame：起始可见关键帧。
- seeds：关键帧列表；每项包含 frame_index、visibility="visible"、geometry。
- frames：区间内每一帧，按连续索引排列，不能跳帧。
- source_sha256、segmentation_revision：输入来源及版本信息。

视频描述／事件使用采样 frames，并附加 start、end、whole_video、labels、instruction。labels 是事件标签对象数组，每个对象有 id、name、color；图片检测的 labels 则是类别名称数组。

### 查询、取消

GET /jobs/<id>：

```json
{
  "id": "worker-...",
  "status": "running",
  "progress": {"current": 24, "total": 100, "message": "Propagating"},
  "error": null
}
```

完成状态为 completed，增加 result 字段。失败为 failed，提供 error。取消使用 POST /jobs/<id>/cancel，body 为 {}；服务应尽快释放资源并返回 cancelled。允许进程重启后返回 interrupted。

应用每 0.5 秒检查一次服务任务。超时或取消时调用远端取消接口。默认超时 600 秒，可以在设置中调整到 10–3600 秒。

### 输出结构

图片检测：

```json
{"objects":[{"label":"person","bbox":[10,20,200,300],"confidence":0.85}]}
```

图片分割：

```json
{"objects":[{"label":"person","kind":"polygon","polygons":[[[10,20],[30,20],[30,40],[10,40]]]}]}
```

视频分割：

```json
{"keyframes":[
  {"frame_index":12,"visibility":"visible","geometry":{"kind":"polygon","polygons":[[[10,20],[30,20],[30,40],[10,40]]]}},
  {"frame_index":13,"visibility":"occluded","geometry":null}
]}
```

有孔洞或复杂轮廓请返回 kind="mask"，使用 [实例分割文档](instance-segmentation.md) 中的 tile-rle-row-v1：128×128 原图像素对齐块，按行 RLE，首段为背景，块内计数总和 16384。禁止用填实多边形掩盖真实孔洞。

视频描述：

```json
{"captions":[{"text":"一个人走向桌边，拿起杯子。"}]}
```

视频事件：

```json
{"events":[{"kind":"interval","start":1.2,"end":2.5,"label_id":null,"text":"拿起杯子"}]}
```

若命中固定类别，label_id 必须来自请求。点事件使用 kind="point" 和 start，end 省略或为 null。空数组代表未产生建议。不要为了返回非空结果而猜测内容。

可在 result 顶层返回 model_revision 字符串，应用会将它写入建议来源。

### 校验与限制

应用校验几何、原图边界、类别、时间范围、帧索引唯一性及响应大小。单次建议最多 2000 条、32MB；传播最多 1000 帧，帧载荷合计最多 96MB。默认 worker 限制请求体 128MB，并限制队列为 4 个任务。

模型生成的 reviewed、id、provenance 等字段不能决定人工状态：应用重新分配建议 ID 和来源，并强制未审核。采纳还会检查当前标注版本，并再次保护人工关键帧及明确可见性状态。

部署是单进程应用 + 单进程 worker。不要用多个 uvicorn workers 共享同一个项目目录。
