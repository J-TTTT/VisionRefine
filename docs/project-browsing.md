# 大项目浏览与下一步

## 仓库定位

VisionRefine 以本地数据集和原图坐标为基础，把导入粗标注、AI 建议和人工确认版本接到同一个工作台。
目前主要可用链路是目标检测和实例分割：前者支持检测框编辑与 VLM 切片初标，后者已有多边形、
稀疏掩码、画笔、边界调整、合并拆分以及 COCO 分割导入导出。

- `core/dataset_io/`：数据契约、格式适配、导入预览/追加、版本选择和后台导出。
- `core/store.py`、`server.py`：项目持久化、人工/AI 版本、HTTP API 和图像访问。
- `core/router.py`、`pilot.py`、`initial_detection.py`：分辨率路由与检测模型调用。
- `static/app.js`、`segmentation.js`：检测与分割编辑；`image-browser.js` 负责分页浏览。

人工确认结果（包括空标注）优先于 AI 建议和导入粗标注，是后续模型接入必须保持的约束。

## 本次改动

此前 `project.json` 的 `analysis.images` 保存全部图像记录，首页就会携带所有项目的图像清单；
项目页一次渲染全部表格，工作台再次下载全量列表并生成全部下拉选项。缩略图、裁剪和标注接口
还会解析整个 JSON manifest，包含所有分割几何的校验。

现在项目摘要与图像清单分开：

- `project.json` 只保存分析汇总及 `analysis.image_index`。
- `analysis/<id>.sqlite3` 按原始顺序保存图像元数据，供分页、跳页和路径搜索使用。
- `datasets/import-<id>.sqlite3` 为同名 JSON manifest 建立索引，支持分别读取文件位置和单图标注。
- 新索引先写临时文件，完成后原子发布，项目随后发布对应指针。追加和重新分析生成新版本。
- JSON manifest 继续用于完整导入、导出和历史记录；人工版本仍保存在原来的标注文档中。

SQLite 使用 Python 标准库，不增加依赖。单图路径仍检查目录边界和符号链接。页面只保留当前
50 条记录；加载失败保留原画布并允许重试，过期请求不会覆盖新结果，跨页检查未保存修改。
AI 初标在开始时绑定项目和图像，完成时检查当前页面与未保存编辑。

## 兼容性和恢复

旧项目首次读取时，将 `analysis.images` 转成索引并保存摘要。缺少数据集索引的旧项目首次访问
图像时，会从既有 manifest 建立索引。这一首次转换仍随项目大小耗时，后续访问及重启复用磁盘索引；
不重新扫描来源目录，也不读取或散列全部源图片。

新导入在提交期间建立数据集索引。旧分析索引随项目保留，以免影响正在使用旧版本的请求。
缺失分析索引时，列表返回 409 并提示重新分析。删除项目时索引随项目移入回收目录。

**API 返回结构有变化**：项目详情和分析响应不再包含 `analysis.images`，`GET /images` 改为分页对象。
调用方应读取 `items` 并依据 `total` 翻页；完整 manifest 仍可通过 `/dataset` 显式获取。

```text
GET /api/projects/{id}/images?offset=0&limit=50&q=frame&split=train
{
  "items": [...],
  "total": 10000,
  "offset": 0,
  "limit": 50,
  "revision": "<analysis image_index>"
}
```

`offset` 至少为 0，`limit` 默认为 50、最大为 200；`q` 为最长 256 字符的路径子串，
`split` 可选 `train/val/test/unspecified`。空结果和越界页返回空 `items`。
`GET /api/projects/{id}/dataset/report` 只读取导入报告，查看报告无需下载全量图像和分割数据。

## 验证与基准

```bash
.venv/bin/python -m pytest -q
VISIONREFINE_BROWSER_TESTS=1 .venv/bin/python -m pytest -q \
  tests/test_image_paging.py tests/test_segmentation_browser.py tests/test_dataset_io_browser.py
.venv/bin/python scripts/benchmark_project_browsing.py --images 10000 50000 --repeats 5
```

接口测试覆盖 10,003 条记录的全量遍历、末页、筛选、参数校验、重启、旧项目转换、索引恢复、
新版本可见性、单图读取与人工空标注。测试禁止浏览请求读取完整 JSON manifest，防止退回
“先读取全部数据，再切一页”的实现。真实浏览器验收检查表格和选项数、请求乱序、失败重试、
跨页未保存确认和人工保存后重新打开。

本机 Python 3.13.13，每项请求 5 次，下表为中位耗时：

| 指标 | 10,000 条 | 50,000 条 |
|---|---:|---:|
| 项目详情响应 | 806 B | 806 B |
| 首个 50 条列表响应 | 12,193 B | 12,193 B |
| 项目详情 | 1.78 ms | 2.10 ms |
| 第一页 | 2.76 ms | 2.84 ms |
| 最后 50 条 | 3.01 ms | 3.21 ms |
| 单图分割标注 | 3.03 ms | 2.86 ms |
| 路径子串搜索 | 11.48 ms | 29.68 ms |

基准生成独立临时项目和一张 128×128 PNG，各逻辑图像引用这张生成图，每条记录包含一个简单
多边形。计时通过本机 TestClient 完成，开始前新建 ProjectStore，包含响应序列化。它验证记录数量
增加时的浏览开销，不测量网络延迟、冷磁盘扫描、真实大图解码或复杂掩码的单图编辑性能。
本次没有访问 OCEAN 数据集。路径子串搜索仍扫描索引文本，耗时随记录数量增长。

## 下一步顺序

1. **分割模型适配接口**：输入图像/局部区域和正负点或框，输出原图坐标下的候选掩码及来源。
   先用可控的假适配器验证坐标变换、失败恢复、取消和人工优先，再接真实模型。
2. **接通预览和应用**：候选结果进入现有分割预览流程，明确应用后加入编辑并可撤销，AI 建议
   独立保存。复杂结果遵守已有掩码块数和内存预算。
3. **再处理导入耗时**：预览、提交和重新分析仍是同步操作；如果瓶颈落在这些步骤，再增加后台
   进度、取消和恢复。分页不会消除首次导入的文件枚举、尺寸读取或格式解析成本。
