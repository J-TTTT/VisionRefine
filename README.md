# VisionRefine

[中文](#中文) | [English](#english)

VisionRefine is a local-first, AI-assisted annotation workspace for images of any resolution, from ordinary datasets to gigapixel scenes.

![VisionRefine annotation workspace](docs/images/annotation-workspace.png)

## 中文

VisionRefine 是一个本地优先的 AI 辅助多模态数据标注工具。它面向不同尺寸的图像数据，自动选择整图、缩放、重叠切片或标注引导裁剪等处理方式，将“快速模型初标、视觉大模型复核、人工校正、AI 最终检查”组织为可追踪的工作流。

当前原型重点实现了目标检测和超高分辨率图像标注。AI 输出始终作为独立建议版本保存，不会静默覆盖人工确认结果。

### 界面预览

**数据分析与处理路由**

系统扫描数据集的尺寸和已有标注，并为每张图像选择合适的模型输入策略。下图展示了平均 550 MP 数据集的重叠切片路由结果。

![项目分析与处理路由](docs/images/project-overview.png)

**超高分辨率标注工作台**

工作台按需读取原始分辨率局部区域。可使用滚轮缩放、右键拖动画面、左键新增框，并对选中框进行移动、八方向缩放或删除。所有框均使用原图坐标保存。

![超高分辨率标注工作台](docs/images/annotation-workspace.png)

### 当前能力

- 支持目标检测、实例分割、视觉定位、图像描述、VQA、OCR 和图像分类项目类型。
- 自动分析图像数量、分辨率、已有粗标注及推荐输入策略。
- 支持直接输入、整图缩放、重叠切片和标注引导裁剪。
- 对接 OpenAI-compatible API 和 vLLM，可发现模型并运行单切片或整图切片测试。
- 提供超高分辨率画布，支持缩放、平移、新增、选择、移动、缩放和删除检测框。
- AI 建议与人工版本分开保存，重新加载时优先读取人工确认结果。
- 原始图像保留在本地，删除项目不会删除源数据。

> VisionRefine 仍处于早期原型阶段。快速检测器、AI 候选框复核、分割编辑器和通用导入导出仍在开发中。

### 本地运行

```bash
git clone https://github.com/Soleilor/VisionRefine.git
cd VisionRefine
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
visionrefine --port 8020
```

打开 <http://127.0.0.1:8020>。

### 基本流程

1. 新建项目并选择任务类型和候选标签。
2. 指定本地图像目录；已有粗标注时可同时填写标注文件。
3. 查看自动生成的分辨率路由，并配置兼容 OpenAI API 的视觉模型。
4. 运行 AI 测试或打开标注工作台进行人工校正。
5. 保存人工检查结果；后续 AI 复核只生成建议，不覆盖人工版本。

## English

VisionRefine is a local-first, AI-assisted multimodal annotation tool for images of any size. It automatically routes each image through whole-image input, resizing, overlapping tiles, or annotation-guided crops, and organizes fast model proposals, VLM review, human correction, and final AI auditing into a traceable workflow.

The current prototype focuses on object detection and ultra-high-resolution imagery. AI output is always stored as a separate suggestion revision and never silently overwrites human-reviewed annotations.

### Interface

**Dataset analysis and resolution routing**

VisionRefine scans image dimensions and existing annotations, then selects an appropriate model-input strategy for every image.

![Dataset analysis and resolution routing](docs/images/project-overview.png)

**Ultra-high-resolution annotation workspace**

The workspace loads original-resolution regions on demand. Use the mouse wheel to zoom, right-drag to pan, and left-drag to create a box. A selected box can be moved, resized from eight handles, or deleted. Coordinates are always stored in the original image space.

![Ultra-high-resolution annotation workspace](docs/images/annotation-workspace.png)

### Current capabilities

- Project types for detection, instance segmentation, visual grounding, captioning, VQA, OCR, and image classification.
- Automatic dataset inspection covering image counts, resolution, coarse annotations, and recommended input routing.
- Direct input, whole-image resize, overlapping tiles, and annotation-guided crops.
- OpenAI-compatible and vLLM adapters with model discovery and tile-based pilot inference.
- A gigapixel-capable canvas for zooming, panning, creating, selecting, moving, resizing, and deleting boxes.
- Separate AI suggestion and human-reviewed revisions, with human revisions taking precedence when reloaded.
- Local source images remain untouched when a VisionRefine project is deleted.

> VisionRefine is an early prototype. Fast detector adapters, selective VLM review, segmentation editors, and general-purpose import/export are still under development.

### Run locally

```bash
git clone https://github.com/Soleilor/VisionRefine.git
cd VisionRefine
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
visionrefine --port 8020
```

Open <http://127.0.0.1:8020>.

### Basic workflow

1. Create a project and choose a task type and candidate labels.
2. Select a local image directory and, optionally, an existing coarse annotation file.
3. Review the generated resolution routes and configure an OpenAI-compatible vision model.
4. Run an AI pilot or open the annotation workspace for human correction.
5. Save the human-reviewed revision. Later AI passes remain suggestions and do not overwrite it.

## Development

```bash
pytest -q
```

## License

Apache-2.0.
